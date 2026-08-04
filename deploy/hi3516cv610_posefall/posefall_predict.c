#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ot_common_video.h"
#include "ot_common_vpss.h"
#include "ot_common_region.h"
#include "ss_mpi_region.h"
#include "ss_mpi_sys.h"
#include "ss_mpi_sys_mem.h"
#include "ss_mpi_vpss.h"
#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "svp_acl_rt.h"

#define VPSS_GRP 0
#define VPSS_TEMPLATE_CHN 1
#define VPSS_AI_CHN 2
#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640
#define NV12_BYTES (MODEL_WIDTH * MODEL_HEIGHT * 3 / 2)
#define RGB_FP32_BYTES (MODEL_WIDTH * MODEL_HEIGHT * 3 * sizeof(float))
#define POSE_CHANNELS 56
#define POSE_CANDIDATES 8400
#define KEYPOINTS 17
#define BOX_RGN_BASE 20
#define MAX_PERSONS 32
#define MAX_OVERLAYS 16
#define MAX_NMS_CANDIDATES 256
#define MAX_TRACKS 32
#define DET_CONF_THRESHOLD 0.1f
#define TRACK_HIGH_THRESHOLD 0.25f
#define NEW_TRACK_THRESHOLD 0.25f
#define NMS_IOU_THRESHOLD 0.7f
#define TRACK_MATCH_THRESHOLD 0.8f
#define SECOND_MATCH_THRESHOLD 0.5f
#define UNCONFIRMED_MATCH_THRESHOLD 0.7f
#define TRACK_BUFFER 30
#define FALL_FEATURE_DIM 56
#define FALL_WINDOW 60
#define FALL_THRESHOLD 0.5f
#define FALL_MIN_CONF 0.2f
#define CAMERA_FRAME_US 33333ULL
#define TRACK_GAP_US 500000ULL

typedef struct {
    int found;
    float confidence;
    float x1, y1, x2, y2;
    float kx[KEYPOINTS], ky[KEYPOINTS], kc[KEYPOINTS];
} pose_result;

typedef struct {
    float rows[FALL_WINDOW][FALL_FEATURE_DIM];
    int count;
    int last_tick;
    int has_last_tick;
} pose_history;

typedef enum {
    TRACK_STATE_NEW = 0,
    TRACK_STATE_TRACKED = 1,
    TRACK_STATE_LOST = 2,
    TRACK_STATE_REMOVED = 3,
} track_state;

typedef struct {
    int used;
    int id;
    int current;
    int is_activated;
    int frame_id;
    int start_frame;
    track_state state;
    pose_result pose;
    pose_history history;
    float probability;
    int probability_valid;
    int is_fall;
    float mean[8];
    float covariance[8][8];
} person_track;

typedef struct {
    person_track items[MAX_TRACKS];
    int next_id;
    int frame_id;
} person_tracker;

typedef struct {
    const char *source;
    const char *pose_model;
    const char *head_model;
    const char *trace_path;
    int frame_limit;
    float source_fps;
} runtime_options;

static td_u32 g_video_width = 1920;
static td_u32 g_video_height = 1080;

typedef struct {
    void *memory;
    size_t size;
    size_t stride;
    svp_acl_data_buffer *buffer;
} tensor_buffer;

typedef struct {
    uint32_t id;
    void *memory;
    size_t size;
    svp_acl_mdl_desc *desc;
    int loaded;
} model_handle;

static volatile sig_atomic_t g_stop;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static double monotonic_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int acl_ok(svp_acl_error rc, const char *operation)
{
    if (rc == SVP_ACL_SUCCESS) return 0;
    fprintf(stderr, "ERROR %s rc=%d\n", operation, rc);
    return -1;
}

static int load_model(const char *path, model_handle *model)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) goto file_error;
    long bytes = ftell(fp);
    if (bytes <= 0 || fseek(fp, 0, SEEK_SET) != 0) goto file_error;
    model->size = (size_t)bytes;
    if (acl_ok(svp_acl_rt_malloc(&model->memory, model->size,
                                 SVP_ACL_MEM_MALLOC_NORMAL_ONLY),
               "allocate model") != 0) {
        fclose(fp);
        return -1;
    }
    if (fread(model->memory, 1, model->size, fp) != model->size) goto file_error;
    fclose(fp);
    if (acl_ok(svp_acl_mdl_load_from_mem(model->memory, model->size, &model->id),
               "load model") != 0) return -1;
    model->loaded = 1;
    model->desc = svp_acl_mdl_create_desc();
    if (!model->desc ||
        acl_ok(svp_acl_mdl_get_desc(model->desc, model->id), "get model desc") != 0) {
        return -1;
    }
    return 0;

file_error:
    fprintf(stderr, "ERROR reading model %s\n", path);
    fclose(fp);
    return -1;
}

static int allocate_tensor(tensor_buffer *tensor, size_t size, size_t stride)
{
    tensor->size = size;
    tensor->stride = stride;
    if (acl_ok(svp_acl_rt_malloc_cached(&tensor->memory, size,
                                        SVP_ACL_MEM_MALLOC_NORMAL_ONLY),
               "allocate tensor") != 0) return -1;
    memset(tensor->memory, 0, size);
    if (acl_ok(svp_acl_rt_mem_flush(tensor->memory, size),
               "flush initialized tensor") != 0) return -1;
    tensor->buffer = svp_acl_create_data_buffer(tensor->memory, size, stride);
    return tensor->buffer ? 0 : -1;
}

static void free_tensors(tensor_buffer *tensors, size_t count)
{
    if (!tensors) return;
    for (size_t i = 0; i < count; ++i) {
        if (tensors[i].buffer) svp_acl_destroy_data_buffer(tensors[i].buffer);
        if (tensors[i].memory) svp_acl_rt_free(tensors[i].memory);
    }
}

static int create_datasets(const model_handle *model,
                           svp_acl_mdl_dataset **inputs,
                           tensor_buffer **input_buffers,
                           size_t *input_count,
                           svp_acl_mdl_dataset **outputs,
                           tensor_buffer **output_buffers,
                           size_t *output_count,
                           size_t expected_user_input_size)
{
    *input_count = svp_acl_mdl_get_num_inputs(model->desc);
    *output_count = svp_acl_mdl_get_num_outputs(model->desc);
    if (*input_count < 3 || *output_count != 1) {
        fprintf(stderr, "ERROR unexpected model io inputs=%zu outputs=%zu\n",
                *input_count, *output_count);
        return -1;
    }
    *inputs = svp_acl_mdl_create_dataset();
    *outputs = svp_acl_mdl_create_dataset();
    *input_buffers = calloc(*input_count, sizeof(**input_buffers));
    *output_buffers = calloc(*output_count, sizeof(**output_buffers));
    if (!*inputs || !*outputs || !*input_buffers || !*output_buffers) return -1;

    for (size_t i = 0; i < *input_count; ++i) {
        size_t size = svp_acl_mdl_get_input_size_by_index(model->desc, i);
        size_t stride = svp_acl_mdl_get_input_default_stride(model->desc, i);
        svp_acl_data_type dtype = svp_acl_mdl_get_input_data_type(model->desc, i);
        if (allocate_tensor(&(*input_buffers)[i], size, stride) != 0 ||
            acl_ok(svp_acl_mdl_add_dataset_buffer(*inputs,
                                                  (*input_buffers)[i].buffer),
                   "add input buffer") != 0) return -1;
        printf("MODEL_INPUT index=%zu bytes=%zu stride=%zu dtype=%d\n",
               i, size, stride, (int)dtype);
    }
    if ((*input_buffers)[0].size != expected_user_input_size) {
        fprintf(stderr, "ERROR model input bytes=%zu expected=%zu\n",
                (*input_buffers)[0].size, expected_user_input_size);
        return -1;
    }

    for (size_t i = 0; i < *output_count; ++i) {
        size_t size = svp_acl_mdl_get_output_size_by_index(model->desc, i);
        size_t stride = svp_acl_mdl_get_output_default_stride(model->desc, i);
        svp_acl_data_type dtype = svp_acl_mdl_get_output_data_type(model->desc, i);
        if (allocate_tensor(&(*output_buffers)[i], size, stride) != 0 ||
            acl_ok(svp_acl_mdl_add_dataset_buffer(*outputs,
                                                  (*output_buffers)[i].buffer),
                   "add output buffer") != 0) return -1;
        printf("MODEL_OUTPUT index=%zu bytes=%zu stride=%zu dtype=%d\n",
               i, size, stride, (int)dtype);
    }
    return 0;
}

static int start_ai_channel(void)
{
    ot_vpss_chn_attr attr;
    memset(&attr, 0, sizeof(attr));
    td_s32 rc = ss_mpi_vpss_get_chn_attr(VPSS_GRP, VPSS_TEMPLATE_CHN, &attr);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "ERROR get VPSS template rc=0x%08x\n", (unsigned)rc);
        return -1;
    }
    g_video_width = attr.width;
    g_video_height = attr.height;
    attr.width = MODEL_WIDTH;
    attr.height = MODEL_HEIGHT;
    attr.depth = 1;
    attr.compress_mode = OT_COMPRESS_MODE_NONE;
    attr.frame_rate.src_frame_rate = -1;
    attr.frame_rate.dst_frame_rate = -1;
    rc = ss_mpi_vpss_set_chn_attr(VPSS_GRP, VPSS_AI_CHN, &attr);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "ERROR set VPSS AI channel rc=0x%08x\n", (unsigned)rc);
        return -1;
    }
    rc = ss_mpi_vpss_enable_chn(VPSS_GRP, VPSS_AI_CHN);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "ERROR enable VPSS AI channel rc=0x%08x\n", (unsigned)rc);
        return -1;
    }
    return 0;
}

static int clamp_u8(int value)
{
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

static void nv21_to_rgb_fp32(const uint8_t *nv21, float *rgb)
{
    const uint8_t *y_plane = nv21;
    const uint8_t *vu_plane = nv21 + MODEL_WIDTH * MODEL_HEIGHT;
    float *r_plane = rgb;
    float *g_plane = rgb + MODEL_WIDTH * MODEL_HEIGHT;
    float *b_plane = rgb + MODEL_WIDTH * MODEL_HEIGHT * 2;
    const float scale = 1.0f / 255.0f;

    for (int y = 0; y < MODEL_HEIGHT; ++y) {
        const uint8_t *y_row = y_plane + y * MODEL_WIDTH;
        const uint8_t *vu_row = vu_plane + (y / 2) * MODEL_WIDTH;
        size_t row = (size_t)y * MODEL_WIDTH;
        for (int x = 0; x < MODEL_WIDTH; x += 2) {
            int v = (int)vu_row[x] - 128;
            int u = (int)vu_row[x + 1] - 128;
            int red_delta = (1613 * v) >> 10;
            int green_delta = -((192 * u + 479 * v) >> 10);
            int blue_delta = (1900 * u) >> 10;
            for (int dx = 0; dx < 2; ++dx) {
                int luma = y_row[x + dx];
                size_t p = row + (size_t)x + dx;
                r_plane[p] = clamp_u8(luma + red_delta) * scale;
                g_plane[p] = clamp_u8(luma + green_delta) * scale;
                b_plane[p] = clamp_u8(luma + blue_delta) * scale;
            }
        }
    }
}

static int copy_camera_frame(void *model_input, size_t input_size, uint64_t *pts)
{
    ot_video_frame_info info;
    memset(&info, 0, sizeof(info));
    td_s32 rc = ss_mpi_vpss_get_chn_frame(VPSS_GRP, VPSS_AI_CHN, &info, 1000);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "ERROR get VPSS frame rc=0x%08x\n", (unsigned)rc);
        return -1;
    }

    const ot_video_frame *frame = &info.video_frame;
    int status = 0;
    if (frame->width != MODEL_WIDTH || frame->height != MODEL_HEIGHT ||
        frame->stride[0] != MODEL_WIDTH || frame->compress_mode != OT_COMPRESS_MODE_NONE) {
        fprintf(stderr, "ERROR unexpected frame %ux%u stride=%u compress=%d\n",
                frame->width, frame->height, frame->stride[0],
                (int)frame->compress_mode);
        status = -1;
    } else {
        void *mapped = ss_mpi_sys_mmap(frame->phys_addr[0], NV12_BYTES);
        if (!mapped) {
            fprintf(stderr, "ERROR mmap frame\n");
            status = -1;
        } else {
            if (input_size == NV12_BYTES) {
                memcpy(model_input, mapped, NV12_BYTES);
            } else {
                nv21_to_rgb_fp32((const uint8_t *)mapped, (float *)model_input);
            }
            ss_mpi_sys_munmap(mapped, NV12_BYTES);
            /* The AIPP CSC matrix is configured for the product's native
             * YVU420SP (NV21) order, so the frame can be copied unchanged. */
            *pts = frame->pts;
        }
    }
    rc = ss_mpi_vpss_release_chn_frame(VPSS_GRP, VPSS_AI_CHN, &info);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "ERROR release VPSS frame rc=0x%08x\n", (unsigned)rc);
        status = -1;
    }
    return status;
}

static float output_value(const tensor_buffer *output, int channel, int candidate)
{
    const uint8_t *row = (const uint8_t *)output->memory +
                         (size_t)channel * output->stride;
    return ((const float *)row)[candidate];
}

static float pose_iou(const pose_result *a, const pose_result *b)
{
    float left = fmaxf(a->x1, b->x1);
    float top = fmaxf(a->y1, b->y1);
    float right = fminf(a->x2, b->x2);
    float bottom = fminf(a->y2, b->y2);
    float intersection = fmaxf(0.0f, right - left) * fmaxf(0.0f, bottom - top);
    float area_a = fmaxf(0.0f, a->x2 - a->x1) * fmaxf(0.0f, a->y2 - a->y1);
    float area_b = fmaxf(0.0f, b->x2 - b->x1) * fmaxf(0.0f, b->y2 - b->y1);
    float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static int compare_pose_confidence(const void *left, const void *right)
{
    const pose_result *a = (const pose_result *)left;
    const pose_result *b = (const pose_result *)right;
    return a->confidence < b->confidence ? 1 :
           (a->confidence > b->confidence ? -1 : 0);
}

static void decode_pose_candidate(const tensor_buffer *output, int candidate,
                                  pose_result *result)
{
    memset(result, 0, sizeof(*result));
    float x = output_value(output, 0, candidate);
    float y = output_value(output, 1, candidate);
    float w = output_value(output, 2, candidate);
    float h = output_value(output, 3, candidate);
    result->found = 1;
    result->confidence = output_value(output, 4, candidate);
    result->x1 = x - w * 0.5f;
    result->y1 = y - h * 0.5f;
    result->x2 = x + w * 0.5f;
    result->y2 = y + h * 0.5f;
    for (int keypoint = 0; keypoint < KEYPOINTS; ++keypoint) {
        result->kx[keypoint] = output_value(output, 5 + keypoint * 3, candidate);
        result->ky[keypoint] = output_value(output, 6 + keypoint * 3, candidate);
        result->kc[keypoint] = output_value(output, 7 + keypoint * 3, candidate);
    }
}

static int decode_pose_detections(const tensor_buffer *output,
                                  pose_result detections[MAX_PERSONS])
{
    pose_result candidates[MAX_NMS_CANDIDATES];
    int candidate_count = 0;
    int weakest = 0;
    for (int index = 0; index < POSE_CANDIDATES; ++index) {
        float confidence = output_value(output, 4, index);
        if (!isfinite(confidence) || confidence < DET_CONF_THRESHOLD) continue;
        if (candidate_count < MAX_NMS_CANDIDATES) {
            decode_pose_candidate(output, index, &candidates[candidate_count++]);
            continue;
        }
        for (int item = 1; item < candidate_count; ++item) {
            if (candidates[item].confidence < candidates[weakest].confidence) weakest = item;
        }
        if (confidence > candidates[weakest].confidence) {
            decode_pose_candidate(output, index, &candidates[weakest]);
        }
    }
    qsort(candidates, (size_t)candidate_count, sizeof(candidates[0]),
          compare_pose_confidence);

    int kept = 0;
    for (int candidate = 0;
         candidate < candidate_count && kept < MAX_PERSONS;
         ++candidate) {
        int suppressed = 0;
        for (int previous = 0; previous < kept; ++previous) {
            if (pose_iou(&candidates[candidate], &detections[previous]) >
                NMS_IOU_THRESHOLD) {
                suppressed = 1;
                break;
            }
        }
        if (!suppressed) detections[kept++] = candidates[candidate];
    }
    return kept;
}

static void pose_to_features(const pose_result *pose, float features[FALL_FEATURE_DIM])
{
    float width = pose->x2 - pose->x1;
    float height = pose->y2 - pose->y1;
    memset(features, 0, sizeof(float) * FALL_FEATURE_DIM);
    features[0] = ((pose->x1 + pose->x2) * 0.5f) / MODEL_WIDTH;
    features[1] = ((pose->y1 + pose->y2) * 0.5f) / MODEL_HEIGHT;
    features[2] = width / MODEL_WIDTH;
    features[3] = height / MODEL_HEIGHT;
    features[4] = pose->confidence;
    for (int k = 0; k < KEYPOINTS; ++k) {
        features[5 + k * 3] = pose->kx[k] / MODEL_WIDTH;
        features[6 + k * 3] = pose->ky[k] / MODEL_HEIGHT;
        features[7 + k * 3] = pose->kc[k] < 0.0f ? 0.0f :
                               (pose->kc[k] > 1.0f ? 1.0f : pose->kc[k]);
    }
}

static void reset_pose_history(pose_history *history)
{
    memset(history, 0, sizeof(*history));
}

static void append_history_row(pose_history *history,
                               const float features[FALL_FEATURE_DIM])
{
    if (history->count < FALL_WINDOW) {
        memcpy(history->rows[history->count++], features,
               sizeof(float) * FALL_FEATURE_DIM);
    } else {
        memmove(history->rows[0], history->rows[1],
                sizeof(float) * (FALL_WINDOW - 1) * FALL_FEATURE_DIM);
        memcpy(history->rows[FALL_WINDOW - 1], features,
               sizeof(float) * FALL_FEATURE_DIM);
    }
}

static void append_resampled_history(pose_history *history,
                                     const float features[FALL_FEATURE_DIM],
                                     int tick)
{
    int max_gap_ticks = (int)lroundf(0.5f * 30.0f);
    if (!history->has_last_tick || tick - history->last_tick > max_gap_ticks) {
        reset_pose_history(history);
        append_history_row(history, features);
        history->last_tick = tick;
        history->has_last_tick = 1;
        return;
    }
    if (tick <= history->last_tick) {
        if (history->count > 0) {
            memcpy(history->rows[history->count - 1], features,
                   sizeof(float) * FALL_FEATURE_DIM);
        } else {
            append_history_row(history, features);
        }
        return;
    }
    int missing = tick - history->last_tick - 1;
    for (int sample = 0; sample < missing && history->count > 0; ++sample) {
        float previous[FALL_FEATURE_DIM];
        memcpy(previous, history->rows[history->count - 1], sizeof(previous));
        append_history_row(history, previous);
    }
    append_history_row(history, features);
    history->last_tick = tick;
}

static float mean_keypoint_confidence(const pose_history *history)
{
    if (history->count <= 0) return 0.0f;

    double total = 0.0;
    int padding = FALL_WINDOW - history->count;
    for (int row = 0; row < FALL_WINDOW; ++row) {
        int source = row < padding ? 0 : row - padding;
        for (int keypoint = 0; keypoint < KEYPOINTS; ++keypoint) {
            total += history->rows[source][7 + keypoint * 3];
        }
    }
    return (float)(total / (FALL_WINDOW * KEYPOINTS));
}

static void detection_xywh(const pose_result *detection, float measurement[4])
{
    measurement[0] = (detection->x1 + detection->x2) * 0.5f;
    measurement[1] = (detection->y1 + detection->y2) * 0.5f;
    measurement[2] = detection->x2 - detection->x1;
    measurement[3] = detection->y2 - detection->y1;
}

static void track_box(const person_track *track, pose_result *box)
{
    *box = track->pose;
    box->x1 = track->mean[0] - track->mean[2] * 0.5f;
    box->y1 = track->mean[1] - track->mean[3] * 0.5f;
    box->x2 = track->mean[0] + track->mean[2] * 0.5f;
    box->y2 = track->mean[1] + track->mean[3] * 0.5f;
}

static void kalman_initiate(person_track *track, const pose_result *detection)
{
    float measurement[4];
    detection_xywh(detection, measurement);
    memset(track->mean, 0, sizeof(track->mean));
    memset(track->covariance, 0, sizeof(track->covariance));
    for (int index = 0; index < 4; ++index) track->mean[index] = measurement[index];
    const float position_weight = 1.0f / 20.0f;
    const float velocity_weight = 1.0f / 160.0f;
    float deviations[8] = {
        2 * position_weight * measurement[2],
        2 * position_weight * measurement[3],
        2 * position_weight * measurement[2],
        2 * position_weight * measurement[3],
        10 * velocity_weight * measurement[2],
        10 * velocity_weight * measurement[3],
        10 * velocity_weight * measurement[2],
        10 * velocity_weight * measurement[3],
    };
    for (int index = 0; index < 8; ++index) {
        track->covariance[index][index] = deviations[index] * deviations[index];
    }
}

static void kalman_predict(person_track *track)
{
    if (track->state != TRACK_STATE_TRACKED) {
        track->mean[6] = 0.0f;
        track->mean[7] = 0.0f;
    }
    for (int index = 0; index < 4; ++index) {
        track->mean[index] += track->mean[index + 4];
    }

    float predicted[8][8];
    memset(predicted, 0, sizeof(predicted));
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 8; ++column) {
            float value = track->covariance[row][column];
            if (row < 4) value += track->covariance[row + 4][column];
            if (column < 4) value += track->covariance[row][column + 4];
            if (row < 4 && column < 4) {
                value += track->covariance[row + 4][column + 4];
            }
            predicted[row][column] = value;
        }
    }
    const float position_weight = 1.0f / 20.0f;
    const float velocity_weight = 1.0f / 160.0f;
    float deviations[8] = {
        position_weight * track->mean[2],
        position_weight * track->mean[3],
        position_weight * track->mean[2],
        position_weight * track->mean[3],
        velocity_weight * track->mean[2],
        velocity_weight * track->mean[3],
        velocity_weight * track->mean[2],
        velocity_weight * track->mean[3],
    };
    for (int index = 0; index < 8; ++index) {
        predicted[index][index] += deviations[index] * deviations[index];
    }
    memcpy(track->covariance, predicted, sizeof(predicted));
}

static int invert_4x4(const float input[4][4], float inverse[4][4])
{
    float augmented[4][8];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            augmented[row][column] = input[row][column];
            augmented[row][column + 4] = row == column ? 1.0f : 0.0f;
        }
    }
    for (int pivot = 0; pivot < 4; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < 4; ++row) {
            if (fabsf(augmented[row][pivot]) > fabsf(augmented[best][pivot])) best = row;
        }
        if (fabsf(augmented[best][pivot]) < 1e-12f) return -1;
        if (best != pivot) {
            for (int column = 0; column < 8; ++column) {
                float swap = augmented[pivot][column];
                augmented[pivot][column] = augmented[best][column];
                augmented[best][column] = swap;
            }
        }
        float scale = augmented[pivot][pivot];
        for (int column = 0; column < 8; ++column) augmented[pivot][column] /= scale;
        for (int row = 0; row < 4; ++row) {
            if (row == pivot) continue;
            float factor = augmented[row][pivot];
            for (int column = 0; column < 8; ++column) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            inverse[row][column] = augmented[row][column + 4];
        }
    }
    return 0;
}

static void kalman_update(person_track *track, const pose_result *detection)
{
    float measurement[4];
    detection_xywh(detection, measurement);
    const float position_weight = 1.0f / 20.0f;
    float projected[4][4];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            projected[row][column] = track->covariance[row][column];
        }
    }
    float deviations[4] = {
        position_weight * track->mean[2],
        position_weight * track->mean[3],
        position_weight * track->mean[2],
        position_weight * track->mean[3],
    };
    for (int index = 0; index < 4; ++index) {
        projected[index][index] += deviations[index] * deviations[index];
    }
    float inverse[4][4];
    if (invert_4x4(projected, inverse) != 0) return;
    float gain[8][4];
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 4; ++column) {
            gain[row][column] = 0.0f;
            for (int inner = 0; inner < 4; ++inner) {
                gain[row][column] += track->covariance[row][inner] *
                                     inverse[inner][column];
            }
        }
    }
    float innovation[4];
    for (int index = 0; index < 4; ++index) {
        innovation[index] = measurement[index] - track->mean[index];
    }
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 4; ++column) {
            track->mean[row] += gain[row][column] * innovation[column];
        }
    }
    float old_covariance[8][8];
    memcpy(old_covariance, track->covariance, sizeof(old_covariance));
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 8; ++column) {
            float correction = 0.0f;
            for (int inner = 0; inner < 4; ++inner) {
                correction += gain[row][inner] * old_covariance[inner][column];
            }
            track->covariance[row][column] -= correction;
        }
    }
}

static void update_track_pose(person_track *track, const pose_result *detection,
                              int frame_id, int reactivate)
{
    kalman_update(track, detection);
    track->pose = *detection;
    track->pose.x1 = track->mean[0] - track->mean[2] * 0.5f;
    track->pose.y1 = track->mean[1] - track->mean[3] * 0.5f;
    track->pose.x2 = track->mean[0] + track->mean[2] * 0.5f;
    track->pose.y2 = track->mean[1] + track->mean[3] * 0.5f;
    track->frame_id = frame_id;
    track->state = TRACK_STATE_TRACKED;
    track->is_activated = 1;
    track->current = 1;
    (void)reactivate;
}

/*
 * Rectangular partial assignment matching lap.lapjv(..., extend_cost=True,
 * cost_limit=threshold). Real rows/columns are augmented with dummy nodes whose
 * half-threshold costs make a real edge preferable exactly when cost<threshold.
 */
static int linear_assignment(const float costs[MAX_TRACKS][MAX_PERSONS],
                             int rows, int columns, float threshold,
                             int row_match[MAX_TRACKS],
                             int column_match[MAX_PERSONS])
{
    int size = rows + columns;
    if (size == 0) return 0;
    float matrix[MAX_TRACKS + MAX_PERSONS][MAX_TRACKS + MAX_PERSONS];
    const float infinity = 1000000.0f;
    float unmatched = threshold * 0.5f;
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) matrix[row][column] = infinity;
    }
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (costs[row][column] <= threshold) matrix[row][column] = costs[row][column];
        }
        matrix[row][columns + row] = unmatched;
    }
    for (int column = 0; column < columns; ++column) {
        matrix[rows + column][column] = unmatched;
        for (int dummy = 0; dummy < rows; ++dummy) {
            matrix[rows + column][columns + dummy] = 0.0f;
        }
    }

    float u[MAX_TRACKS + MAX_PERSONS + 1] = {0};
    float v[MAX_TRACKS + MAX_PERSONS + 1] = {0};
    int p[MAX_TRACKS + MAX_PERSONS + 1] = {0};
    int way[MAX_TRACKS + MAX_PERSONS + 1] = {0};
    for (int i = 1; i <= size; ++i) {
        p[0] = i;
        float minimum[MAX_TRACKS + MAX_PERSONS + 1];
        int used[MAX_TRACKS + MAX_PERSONS + 1] = {0};
        for (int j = 0; j <= size; ++j) minimum[j] = infinity;
        int j0 = 0;
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            float delta = infinity;
            for (int j = 1; j <= size; ++j) {
                if (used[j]) continue;
                float current_cost = matrix[i0 - 1][j - 1] - u[i0] - v[j];
                if (current_cost < minimum[j]) {
                    minimum[j] = current_cost;
                    way[j] = j0;
                }
                if (minimum[j] < delta) {
                    delta = minimum[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= size; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minimum[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    for (int row = 0; row < rows; ++row) row_match[row] = -1;
    for (int column = 0; column < columns; ++column) column_match[column] = -1;
    int matches = 0;
    for (int column = 1; column <= size; ++column) {
        int row = p[column] - 1;
        int col = column - 1;
        if (row >= 0 && row < rows && col >= 0 && col < columns &&
            costs[row][col] <= threshold) {
            row_match[row] = col;
            column_match[col] = row;
            ++matches;
        }
    }
    return matches;
}

static void build_fused_cost(person_track *const tracks[MAX_TRACKS], int track_count,
                             const pose_result *const detections[MAX_PERSONS],
                             int detection_count,
                             float costs[MAX_TRACKS][MAX_PERSONS],
                             int fuse_score)
{
    for (int row = 0; row < track_count; ++row) {
        pose_result predicted;
        track_box(tracks[row], &predicted);
        for (int column = 0; column < detection_count; ++column) {
            float similarity = pose_iou(&predicted, detections[column]);
            if (fuse_score) similarity *= detections[column]->confidence;
            costs[row][column] = 1.0f - similarity;
        }
    }
}

static int update_person_tracker(person_tracker *tracker,
                                 const pose_result detections[MAX_PERSONS],
                                 int detection_count,
                                 person_track *current[MAX_PERSONS])
{
    ++tracker->frame_id;
    int frame_id = tracker->frame_id;
    for (int index = 0; index < MAX_TRACKS; ++index) {
        tracker->items[index].current = 0;
    }
    for (int index = 0; index < detection_count; ++index) current[index] = NULL;

    int high_indices[MAX_PERSONS], low_indices[MAX_PERSONS];
    int high_count = 0, low_count = 0;
    for (int index = 0; index < detection_count; ++index) {
        if (detections[index].confidence >= TRACK_HIGH_THRESHOLD) {
            high_indices[high_count++] = index;
        } else if (detections[index].confidence > DET_CONF_THRESHOLD) {
            low_indices[low_count++] = index;
        }
    }

    person_track *pool[MAX_TRACKS], *unconfirmed[MAX_TRACKS];
    int pool_count = 0, unconfirmed_count = 0;
    for (int index = 0; index < MAX_TRACKS; ++index) {
        person_track *track = &tracker->items[index];
        if (!track->used) continue;
        if (track->state == TRACK_STATE_TRACKED && !track->is_activated) {
            unconfirmed[unconfirmed_count++] = track;
        } else if (track->state == TRACK_STATE_TRACKED ||
                   track->state == TRACK_STATE_LOST) {
            pool[pool_count++] = track;
        }
    }
    for (int index = 0; index < pool_count; ++index) kalman_predict(pool[index]);

    const pose_result *high[MAX_PERSONS], *low[MAX_PERSONS];
    for (int index = 0; index < high_count; ++index) high[index] = &detections[high_indices[index]];
    for (int index = 0; index < low_count; ++index) low[index] = &detections[low_indices[index]];

    float costs[MAX_TRACKS][MAX_PERSONS] = {{0}};
    int pool_match[MAX_TRACKS], high_match[MAX_PERSONS];
    build_fused_cost(pool, pool_count, high, high_count, costs, 1);
    linear_assignment(costs, pool_count, high_count, TRACK_MATCH_THRESHOLD,
                      pool_match, high_match);
    for (int row = 0; row < pool_count; ++row) {
        if (pool_match[row] < 0) continue;
        int high_slot = pool_match[row];
        int detection_index = high_indices[high_slot];
        update_track_pose(pool[row], &detections[detection_index], frame_id,
                          pool[row]->state == TRACK_STATE_LOST);
        current[detection_index] = pool[row];
    }

    person_track *second_tracks[MAX_TRACKS];
    int second_count = 0;
    for (int row = 0; row < pool_count; ++row) {
        if (pool_match[row] < 0 && pool[row]->state == TRACK_STATE_TRACKED) {
            second_tracks[second_count++] = pool[row];
        }
    }
    int second_match[MAX_TRACKS], low_match[MAX_PERSONS];
    build_fused_cost(second_tracks, second_count, low, low_count, costs, 1);
    linear_assignment(costs, second_count, low_count, SECOND_MATCH_THRESHOLD,
                      second_match, low_match);
    for (int row = 0; row < second_count; ++row) {
        person_track *track = second_tracks[row];
        if (second_match[row] >= 0) {
            int detection_index = low_indices[second_match[row]];
            update_track_pose(track, &detections[detection_index], frame_id, 0);
            current[detection_index] = track;
        } else {
            track->state = TRACK_STATE_LOST;
        }
    }

    int remaining_high_indices[MAX_PERSONS];
    const pose_result *remaining_high[MAX_PERSONS];
    int remaining_high_count = 0;
    for (int column = 0; column < high_count; ++column) {
        if (high_match[column] < 0) {
            remaining_high_indices[remaining_high_count] = high_indices[column];
            remaining_high[remaining_high_count++] = high[column];
        }
    }
    int unconfirmed_match[MAX_TRACKS], remaining_match[MAX_PERSONS];
    build_fused_cost(unconfirmed, unconfirmed_count, remaining_high,
                     remaining_high_count, costs, 1);
    linear_assignment(costs, unconfirmed_count, remaining_high_count,
                      UNCONFIRMED_MATCH_THRESHOLD,
                      unconfirmed_match, remaining_match);
    for (int row = 0; row < unconfirmed_count; ++row) {
        person_track *track = unconfirmed[row];
        if (unconfirmed_match[row] >= 0) {
            int detection_index =
                remaining_high_indices[unconfirmed_match[row]];
            update_track_pose(track, &detections[detection_index], frame_id, 0);
            current[detection_index] = track;
        } else {
            track->state = TRACK_STATE_REMOVED;
        }
    }

    for (int column = 0; column < remaining_high_count; ++column) {
        if (remaining_match[column] >= 0) continue;
        int detection_index = remaining_high_indices[column];
        if (detections[detection_index].confidence < NEW_TRACK_THRESHOLD) continue;
        int slot = -1;
        for (int track_index = 0; track_index < MAX_TRACKS; ++track_index) {
            if (!tracker->items[track_index].used ||
                tracker->items[track_index].state == TRACK_STATE_REMOVED) {
                slot = track_index;
                break;
            }
        }
        if (slot < 0) continue;
        person_track *track = &tracker->items[slot];
        memset(track, 0, sizeof(*track));
        track->used = 1;
        track->id = ++tracker->next_id;
        track->pose = detections[detection_index];
        kalman_initiate(track, &detections[detection_index]);
        track->state = TRACK_STATE_TRACKED;
        track->is_activated = frame_id == 1;
        track->frame_id = frame_id;
        track->start_frame = frame_id;
        track->current = 1;
        if (track->is_activated) current[detection_index] = track;
    }

    for (int index = 0; index < MAX_TRACKS; ++index) {
        person_track *track = &tracker->items[index];
        if (track->used && track->state == TRACK_STATE_LOST &&
            frame_id - track->frame_id > TRACK_BUFFER) {
            track->state = TRACK_STATE_REMOVED;
        }
    }
    for (int tracked = 0; tracked < MAX_TRACKS; ++tracked) {
        person_track *a = &tracker->items[tracked];
        if (!a->used || a->state != TRACK_STATE_TRACKED) continue;
        pose_result a_box;
        track_box(a, &a_box);
        for (int lost = 0; lost < MAX_TRACKS; ++lost) {
            person_track *b = &tracker->items[lost];
            if (!b->used || b->state != TRACK_STATE_LOST) continue;
            pose_result b_box;
            track_box(b, &b_box);
            if (pose_iou(&a_box, &b_box) <= 0.85f) continue;
            int a_age = a->frame_id - a->start_frame;
            int b_age = b->frame_id - b->start_frame;
            if (a_age > b_age) b->state = TRACK_STATE_REMOVED;
            else a->state = TRACK_STATE_REMOVED;
        }
    }
    return detection_count;
}

static void prepare_head_input(const pose_history *history, tensor_buffer *input)
{
    int padding = FALL_WINDOW - history->count;
    for (int row = 0; row < FALL_WINDOW; ++row) {
        int source = row < padding ? 0 : row - padding;
        memcpy((uint8_t *)input->memory + (size_t)row * input->stride,
               history->rows[source], sizeof(float) * FALL_FEATURE_DIM);
    }
}

static int execute_fall_head(const model_handle *model,
                             svp_acl_mdl_dataset *inputs,
                             tensor_buffer *input_buffers,
                             svp_acl_mdl_dataset *outputs,
                             tensor_buffer *output_buffers,
                             const pose_history *history,
                             float *probability, double *elapsed_ms)
{
    prepare_head_input(history, &input_buffers[0]);
    if (acl_ok(svp_acl_rt_mem_flush(input_buffers[0].memory,
                                    input_buffers[0].size),
               "flush head input") != 0) return -1;
    double begin = monotonic_ms();
    if (acl_ok(svp_acl_mdl_execute(model->id, inputs, outputs),
               "fall head execute") != 0) return -1;
    if (acl_ok(svp_acl_rt_mem_invalidate(output_buffers[0].memory,
                                         output_buffers[0].size),
               "invalidate head output") != 0) return -1;
    *elapsed_ms = monotonic_ms() - begin;
    *probability = *(const float *)output_buffers[0].memory;
    return isfinite(*probability) ? 0 : -1;
}

static int start_box_overlays(const ot_mpp_chn *target)
{
    int created = 0;
    for (int index = 0; index < MAX_OVERLAYS; ++index) {
        ot_rgn_handle handle = BOX_RGN_BASE + index;
        ot_rgn_attr region;
        memset(&region, 0, sizeof(region));
        region.type = OT_RGN_COVER;
        td_s32 rc = ss_mpi_rgn_create(handle, &region);
        if (rc != TD_SUCCESS) {
            fprintf(stderr, "WARN create box overlay[%d] rc=0x%08x\n",
                    index, (unsigned)rc);
            break;
        }

        ot_rgn_chn_attr display;
        memset(&display, 0, sizeof(display));
        display.is_show = TD_FALSE;
        display.type = OT_RGN_COVER;
        display.attr.cover_chn.coord = OT_COORD_ABS;
        display.attr.cover_chn.layer = (td_u32)index;
        display.attr.cover_chn.cover.type = OT_COVER_RECT;
        display.attr.cover_chn.cover.color = 0x00ff00;
        display.attr.cover_chn.cover.rect_attr.rect.x = 0;
        display.attr.cover_chn.cover.rect_attr.rect.y = 0;
        display.attr.cover_chn.cover.rect_attr.rect.width = 100;
        display.attr.cover_chn.cover.rect_attr.rect.height = 100;
        display.attr.cover_chn.cover.rect_attr.is_solid = TD_FALSE;
        display.attr.cover_chn.cover.rect_attr.thick = 4;
        rc = ss_mpi_rgn_attach_to_chn(handle, target, &display);
        if (rc != TD_SUCCESS) {
            fprintf(stderr, "WARN attach box overlay[%d] rc=0x%08x\n",
                    index, (unsigned)rc);
            ss_mpi_rgn_destroy(handle);
            break;
        }
        ++created;
    }
    printf("OVERLAY_READY target=VPSS:0:1 slots=%d video=%ux%u\n",
           created, g_video_width, g_video_height);
    return created;
}

static void update_box_overlay(const ot_mpp_chn *target, int slot,
                               const pose_result *pose, int is_fall)
{
    ot_rgn_handle handle = BOX_RGN_BASE + slot;
    ot_rgn_chn_attr display;
    memset(&display, 0, sizeof(display));
    display.type = OT_RGN_COVER;
    display.attr.cover_chn.coord = OT_COORD_ABS;
    display.attr.cover_chn.layer = (td_u32)slot;
    display.attr.cover_chn.cover.type = OT_COVER_RECT;
    display.attr.cover_chn.cover.color = is_fall ? 0xff0000 : 0x00ff00;
    display.attr.cover_chn.cover.rect_attr.is_solid = TD_FALSE;
    display.attr.cover_chn.cover.rect_attr.thick = is_fall ? 8 : 4;
    display.attr.cover_chn.cover.rect_attr.rect.x = 0;
    display.attr.cover_chn.cover.rect_attr.rect.y = 0;
    display.attr.cover_chn.cover.rect_attr.rect.width = 2;
    display.attr.cover_chn.cover.rect_attr.rect.height = 2;
    display.is_show = pose && pose->found ? TD_TRUE : TD_FALSE;

    if (pose && pose->found) {
        int x1 = (int)(pose->x1 * g_video_width / MODEL_WIDTH);
        int y1 = (int)(pose->y1 * g_video_height / MODEL_HEIGHT);
        int x2 = (int)(pose->x2 * g_video_width / MODEL_WIDTH);
        int y2 = (int)(pose->y2 * g_video_height / MODEL_HEIGHT);
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x1 >= (int)g_video_width - 2) x1 = (int)g_video_width - 4;
        if (y1 >= (int)g_video_height - 2) y1 = (int)g_video_height - 4;
        if (x2 < 2) x2 = 2;
        if (y2 < 2) y2 = 2;
        if (x2 >= (int)g_video_width) x2 = (int)g_video_width - 2;
        if (y2 >= (int)g_video_height) y2 = (int)g_video_height - 2;
        x1 &= ~1;
        y1 &= ~1;
        x2 &= ~1;
        y2 &= ~1;
        if (x2 <= x1 + 2 || y2 <= y1 + 2) {
            display.is_show = TD_FALSE;
            display.attr.cover_chn.cover.rect_attr.rect.x = 0;
            display.attr.cover_chn.cover.rect_attr.rect.y = 0;
            display.attr.cover_chn.cover.rect_attr.rect.width = 2;
            display.attr.cover_chn.cover.rect_attr.rect.height = 2;
        } else {
            display.attr.cover_chn.cover.rect_attr.rect.x = x1;
            display.attr.cover_chn.cover.rect_attr.rect.y = y1;
            display.attr.cover_chn.cover.rect_attr.rect.width = (td_u32)(x2 - x1);
            display.attr.cover_chn.cover.rect_attr.rect.height = (td_u32)(y2 - y1);
        }
    }
    td_s32 rc = ss_mpi_rgn_set_chn_display_attr(handle, target, &display);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "WARN update box overlay rc=0x%08x show=%d rect=%d,%d,%u,%u thick=%u\n",
                (unsigned)rc, (int)display.is_show,
                display.attr.cover_chn.cover.rect_attr.rect.x,
                display.attr.cover_chn.cover.rect_attr.rect.y,
                display.attr.cover_chn.cover.rect_attr.rect.width,
                display.attr.cover_chn.cover.rect_attr.rect.height,
                display.attr.cover_chn.cover.rect_attr.thick);
    }
}

static void print_output_stats(const tensor_buffer *output)
{
    float min_value = INFINITY, max_value = -INFINITY;
    size_t nonzero = 0, finite = 0;
    for (int c = 0; c < POSE_CHANNELS; ++c) {
        for (int i = 0; i < POSE_CANDIDATES; ++i) {
            float value = output_value(output, c, i);
            if (isfinite(value)) {
                ++finite;
                if (value < min_value) min_value = value;
                if (value > max_value) max_value = value;
                if (value != 0.0f) ++nonzero;
            }
        }
    }
    printf("OUTPUT_STATS min=%g max=%g finite=%zu nonzero=%zu first=%g,%g,%g,%g,%g\n",
           min_value, max_value, finite, nonzero,
           output_value(output, 0, 0), output_value(output, 1, 0),
           output_value(output, 2, 0), output_value(output, 3, 0),
           output_value(output, 4, 0));
}

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --source camera --pose pose.om --head head.om [--frames N]\n"
            "  %s --source video.mp4 --source-fps FPS "
            "--pose pose.om --head head.om [--frames N] [--trace trace.csv]\n",
            program, program);
}

static int parse_runtime_options(int argc, char **argv, runtime_options *options)
{
    memset(options, 0, sizeof(*options));
    options->source = "camera";

    if (argc >= 3 && argv[1][0] != '-') {
        options->pose_model = argv[1];
        options->head_model = argv[2];
        options->frame_limit = argc >= 4 ? atoi(argv[3]) : 0;
        return argc <= 4 ? 0 : -1;
    }

    for (int index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        if ((!strcmp(argument, "--source") ||
             !strcmp(argument, "--pose") ||
             !strcmp(argument, "--head") ||
             !strcmp(argument, "--trace") ||
             !strcmp(argument, "--frames") ||
             !strcmp(argument, "--source-fps")) &&
            index + 1 >= argc) {
            return -1;
        }
        if (!strcmp(argument, "--source")) {
            options->source = argv[++index];
        } else if (!strcmp(argument, "--pose")) {
            options->pose_model = argv[++index];
        } else if (!strcmp(argument, "--head")) {
            options->head_model = argv[++index];
        } else if (!strcmp(argument, "--trace")) {
            options->trace_path = argv[++index];
        } else if (!strcmp(argument, "--frames")) {
            options->frame_limit = atoi(argv[++index]);
        } else if (!strcmp(argument, "--source-fps")) {
            options->source_fps = strtof(argv[++index], NULL);
        } else if (!strcmp(argument, "--help") || !strcmp(argument, "-h")) {
            print_usage(argv[0]);
            return 1;
        } else {
            fprintf(stderr, "ERROR unknown argument: %s\n", argument);
            return -1;
        }
    }
    if (!options->pose_model || !options->head_model ||
        options->frame_limit < 0 || options->source_fps < 0.0f) {
        return -1;
    }
    return 0;
}

static int is_camera_source(const char *source)
{
    return !strcmp(source, "camera") || !strcmp(source, "0");
}

static float probe_video_fps(const char *ffmpeg, const char *source)
{
    char command[4096];
    int length = snprintf(command, sizeof(command),
                          "'%s' -hide_banner -i '%s' 2>&1",
                          ffmpeg, source);
    if (length < 0 || (size_t)length >= sizeof(command)) return 0.0f;
    FILE *probe = popen(command, "r");
    if (!probe) return 0.0f;
    float result = 0.0f;
    char line[1024];
    while (fgets(line, sizeof(line), probe)) {
        char *fps = strstr(line, " fps,");
        if (!fps) continue;
        char *begin = fps;
        while (begin > line && begin[-1] == ' ') --begin;
        while (begin > line && begin[-1] != ' ' && begin[-1] != ',') --begin;
        char *end = NULL;
        float value = strtof(begin, &end);
        if (end == fps && isfinite(value) && value > 0.0f) {
            result = value;
            break;
        }
    }
    pclose(probe);
    return result;
}

static int read_nv21_frame(FILE *stream, void *model_input, size_t input_size,
                           int frame_index, float source_fps, uint64_t *pts)
{
    if (input_size < NV12_BYTES) return -1;
    size_t read_bytes = fread(model_input, 1, NV12_BYTES, stream);
    if (read_bytes == 0 && feof(stream)) return 1;
    if (read_bytes != NV12_BYTES) {
        fprintf(stderr, "ERROR truncated NV21 frame: got=%zu expected=%d\n",
                read_bytes, NV12_BYTES);
        return -1;
    }
    *pts = (uint64_t)llround((frame_index - 1) * 1000000.0 / source_fps);
    return 0;
}

int main(int argc, char **argv)
{
    runtime_options options;
    int parse_status = parse_runtime_options(argc, argv, &options);
    if (parse_status != 0) {
        if (parse_status < 0) print_usage(argv[0]);
        return parse_status > 0 ? 0 : 2;
    }
    int camera_source = is_camera_source(options.source);
    FILE *source_file = NULL;
    FILE *trace_file = NULL;
    char generated_source_path[2048] = {0};
    int generated_source = 0;
    if (camera_source && options.source_fps <= 0.0f) options.source_fps = 30.0f;
    if (!camera_source) {
        const char *extension = strrchr(options.source, '.');
        if (extension && !strcmp(extension, ".nv21")) {
            if (options.source_fps <= 0.0f) options.source_fps = 30.0f;
            source_file = fopen(options.source, "rb");
        } else {
            const char *ffmpeg = getenv("POSEFALL_FFMPEG");
            if (!ffmpeg || !*ffmpeg) ffmpeg = "/root/posefall/ffmpeg";
            if (strpbrk(options.source, "'\r\n") || strpbrk(ffmpeg, "'\r\n")) {
                fprintf(stderr, "ERROR unsupported quote/newline in source path\n");
                return 2;
            }
            if (options.source_fps <= 0.0f) {
                options.source_fps = probe_video_fps(ffmpeg, options.source);
                if (options.source_fps <= 0.0f) {
                    fprintf(stderr,
                            "ERROR unable to determine source FPS; "
                            "pass --source-fps explicitly\n");
                    return 2;
                }
                printf("SOURCE_FPS %.6g\n", options.source_fps);
            }
            int path_length = snprintf(generated_source_path,
                                       sizeof(generated_source_path),
                                       "%s.posefall_640.nv21", options.source);
            if (path_length < 0 ||
                (size_t)path_length >= sizeof(generated_source_path)) {
                fprintf(stderr, "ERROR generated source path is too long\n");
                return 2;
            }
            char frame_limit_option[64] = {0};
            if (options.frame_limit > 0) {
                snprintf(frame_limit_option, sizeof(frame_limit_option),
                         "-frames:v %d", options.frame_limit);
            }
            char command[4096];
            int length = snprintf(
                command, sizeof(command),
                "'%s' -y -hide_banner -loglevel error -nostdin "
                "-threads 1 -filter_threads 1 -i '%s' -an "
                "-vf \"scale=640:640:force_original_aspect_ratio=decrease,"
                "pad=640:640:(ow-iw)/2:(oh-ih)/2:color=0x727272\" "
                "%s -pix_fmt nv21 -f rawvideo '%s'",
                ffmpeg, options.source, frame_limit_option,
                generated_source_path);
            if (length < 0 || (size_t)length >= sizeof(command)) {
                fprintf(stderr, "ERROR source command is too long\n");
                return 2;
            }
            printf("PREPARING_SOURCE source=%s cache=%s\n",
                   options.source, generated_source_path);
            int command_status = system(command);
            if (command_status != 0) {
                fprintf(stderr, "ERROR ffmpeg source preparation failed status=%d\n",
                        command_status);
                unlink(generated_source_path);
                return 2;
            }
            generated_source = 1;
            source_file = fopen(generated_source_path, "rb");
        }
        if (!source_file) {
            fprintf(stderr, "ERROR unable to open source %s\n", options.source);
            return 2;
        }
    }
    int frame_limit = options.frame_limit;
    if (!options.pose_model || !options.head_model) {
        if (source_file) fclose(source_file);
        if (generated_source) unlink(generated_source_path);
        return 2;
    }
    if (options.trace_path) {
        trace_file = fopen(options.trace_path, "w");
        if (!trace_file) {
            fprintf(stderr, "ERROR unable to open trace %s\n", options.trace_path);
            if (source_file) fclose(source_file);
            if (generated_source) unlink(generated_source_path);
            return 2;
        }
        fprintf(trace_file,
                "frame,track_id,det_conf,det_x1,det_y1,det_x2,det_y2,"
                "track_x1,track_y1,track_x2,track_y2,probability,is_fall\n");
    }
    int status = 1;
    int ai_channel_started = 0;
    int overlay_count = 0;
    int acl_initialized = 0;
    int device_set = 0;
    model_handle pose_model = {0}, head_model = {0};
    svp_acl_mdl_dataset *pose_inputs = NULL, *pose_outputs = NULL;
    svp_acl_mdl_dataset *head_inputs = NULL, *head_outputs = NULL;
    tensor_buffer *pose_input_buffers = NULL, *pose_output_buffers = NULL;
    tensor_buffer *head_input_buffers = NULL, *head_output_buffers = NULL;
    size_t pose_input_count = 0, pose_output_count = 0;
    size_t head_input_count = 0, head_output_count = 0;
    person_tracker tracker = {0};
    ot_mpp_chn overlay_target = {
        .mod_id = OT_ID_VPSS,
        .dev_id = VPSS_GRP,
        .chn_id = VPSS_TEMPLATE_CHN,
    };

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (acl_ok(svp_acl_init(NULL), "svp_acl_init") != 0) goto cleanup;
    acl_initialized = 1;
    if (acl_ok(svp_acl_rt_set_device(0), "set device") != 0) goto cleanup;
    device_set = 1;
    if (load_model(options.pose_model, &pose_model) != 0) goto cleanup;
    if (create_datasets(&pose_model, &pose_inputs, &pose_input_buffers,
                        &pose_input_count, &pose_outputs, &pose_output_buffers,
                        &pose_output_count, NV12_BYTES) != 0) goto cleanup;
    if (load_model(options.head_model, &head_model) != 0) goto cleanup;
    if (create_datasets(&head_model, &head_inputs, &head_input_buffers,
                        &head_input_count, &head_outputs, &head_output_buffers,
                        &head_output_count,
                        FALL_WINDOW * FALL_FEATURE_DIM * sizeof(float)) != 0) goto cleanup;
    if (camera_source) {
        if (start_ai_channel() != 0) goto cleanup;
        ai_channel_started = 1;
        overlay_count = start_box_overlays(&overlay_target);
    }
    printf("POSEFALL_READY pose=%s head=%s channel=%d size=%dx%d window=%d "
           "min_conf=%.2f max_persons=%d source=%s source_fps=%.3f\n",
           options.pose_model, options.head_model, VPSS_AI_CHN,
           MODEL_WIDTH, MODEL_HEIGHT, FALL_WINDOW, FALL_MIN_CONF,
           MAX_PERSONS, options.source, options.source_fps);

    for (int frame = 1; !g_stop && (frame_limit <= 0 || frame <= frame_limit); ++frame) {
        uint64_t pts = 0;
        int source_status = camera_source ?
            copy_camera_frame(pose_input_buffers[0].memory,
                              pose_input_buffers[0].size, &pts) :
            read_nv21_frame(source_file, pose_input_buffers[0].memory,
                            pose_input_buffers[0].size, frame,
                            options.source_fps, &pts);
        if (source_status > 0) break;
        if (source_status < 0) {
            if (camera_source) continue;
            break;
        }
        double frame_begin = monotonic_ms();
        if (acl_ok(svp_acl_rt_mem_flush(pose_input_buffers[0].memory,
                                        pose_input_buffers[0].size),
                   "flush model input") != 0) break;
        if (frame == 1 && getenv("POSEFALL_DUMP")) {
            FILE *dump = fopen("/run/posefall_frame_640.nv21", "wb");
            if (dump) {
                fwrite(pose_input_buffers[0].memory, 1, NV12_BYTES, dump);
                fclose(dump);
                printf("DUMPED /run/posefall_frame_640.nv21 bytes=%d\n", NV12_BYTES);
            }
        }
        double begin = monotonic_ms();
        if (acl_ok(svp_acl_mdl_execute(pose_model.id, pose_inputs, pose_outputs),
                   "model execute") != 0) break;
        if (acl_ok(svp_acl_rt_mem_invalidate(pose_output_buffers[0].memory,
                                             pose_output_buffers[0].size),
                   "invalidate model output") != 0) break;
        double elapsed = monotonic_ms() - begin;
        if (frame == 1 && getenv("POSEFALL_DEBUG")) {
            print_output_stats(&pose_output_buffers[0]);
        }
        if (frame == 1 && getenv("POSEFALL_DUMP_OUTPUT")) {
            FILE *dump = fopen("/run/posefall_output.bin", "wb");
            if (dump) {
                fwrite(pose_output_buffers[0].memory, 1,
                       pose_output_buffers[0].size, dump);
                fclose(dump);
                printf("DUMPED /run/posefall_output.bin bytes=%zu\n",
                       pose_output_buffers[0].size);
            }
        }
        (void)elapsed;
        pose_result detections[MAX_PERSONS];
        person_track *current[MAX_PERSONS] = {0};
        int detection_count = decode_pose_detections(&pose_output_buffers[0],
                                                      detections);
        update_person_tracker(&tracker, detections, detection_count, current);
        int visible_count = 0;
        int head_failed = 0;
        for (int detection = 0; detection < detection_count; ++detection) {
            person_track *track = current[detection];
            if (!track) continue;
            float features[FALL_FEATURE_DIM];
            pose_to_features(&track->pose, features);
            int tick = (int)lround((frame - 1) * 30.0 / options.source_fps);
            append_resampled_history(&track->history, features, tick);
            if (mean_keypoint_confidence(&track->history) >= FALL_MIN_CONF) {
                double head_ms = 0.0;
                if (execute_fall_head(&head_model, head_inputs,
                                      head_input_buffers, head_outputs,
                                      head_output_buffers, &track->history,
                                      &track->probability, &head_ms) != 0) {
                    head_failed = 1;
                    break;
                }
                track->probability_valid = 1;
                track->is_fall = track->probability >= FALL_THRESHOLD;
            } else {
                track->probability_valid = 0;
                track->is_fall = 0;
            }
            ++visible_count;
        }
        if (head_failed) break;
        if (trace_file) {
            if (detection_count == 0) {
                fprintf(trace_file, "%d,,,,,,,,,,,,\n", frame);
            }
            for (int detection = 0; detection < detection_count; ++detection) {
                person_track *track = current[detection];
                fprintf(trace_file,
                        "%d,", frame);
                if (track) fprintf(trace_file, "%d", track->id);
                fprintf(trace_file,
                        ",%.9g,%.9g,%.9g,%.9g,%.9g,",
                        detections[detection].confidence,
                        detections[detection].x1, detections[detection].y1,
                        detections[detection].x2, detections[detection].y2);
                if (track) {
                    fprintf(trace_file, "%.9g,%.9g,%.9g,%.9g,",
                            track->pose.x1, track->pose.y1,
                            track->pose.x2, track->pose.y2);
                    if (track->probability_valid) {
                        fprintf(trace_file, "%.9g,%d\n",
                                track->probability, track->is_fall);
                    } else {
                        fprintf(trace_file, ",0\n");
                    }
                } else {
                    fprintf(trace_file, ",,,,,0\n");
                }
            }
            fflush(trace_file);
        }
        double end_to_end_ms = monotonic_ms() - frame_begin;
        if (visible_count == 0) {
            printf("frame %d: (no detections), %.2fms\n",
                   frame, end_to_end_ms);
        } else {
            printf("frame %d: %d %s (", frame, visible_count,
                   visible_count == 1 ? "person" : "persons");
            int printed = 0;
            for (int detection = 0; detection < detection_count; ++detection) {
                person_track *track = current[detection];
                if (!track) continue;
                printf("%s%s", printed++ ? ", " : "",
                       track->is_fall ? "FALL" : "nofall");
            }
            printf("), %.2fms\n", end_to_end_ms);
        }
        for (int slot = 0; slot < overlay_count; ++slot) {
            person_track *track = slot < detection_count ? current[slot] : NULL;
            update_box_overlay(&overlay_target, slot,
                               track ? &track->pose : NULL,
                               track ? track->is_fall : 0);
        }
    }
    status = 0;

cleanup:
    if (trace_file) fclose(trace_file);
    if (source_file) fclose(source_file);
    if (generated_source) unlink(generated_source_path);
    for (int slot = 0; slot < overlay_count; ++slot) {
        ot_rgn_handle handle = BOX_RGN_BASE + slot;
        ss_mpi_rgn_detach_from_chn(handle, &overlay_target);
        ss_mpi_rgn_destroy(handle);
    }
    if (ai_channel_started) ss_mpi_vpss_disable_chn(VPSS_GRP, VPSS_AI_CHN);
    free_tensors(head_output_buffers, head_output_count);
    free_tensors(head_input_buffers, head_input_count);
    free(head_output_buffers);
    free(head_input_buffers);
    if (head_outputs) svp_acl_mdl_destroy_dataset(head_outputs);
    if (head_inputs) svp_acl_mdl_destroy_dataset(head_inputs);
    if (head_model.loaded) svp_acl_mdl_unload(head_model.id);
    if (head_model.desc) svp_acl_mdl_destroy_desc(head_model.desc);
    if (head_model.memory) svp_acl_rt_free(head_model.memory);
    free_tensors(pose_output_buffers, pose_output_count);
    free_tensors(pose_input_buffers, pose_input_count);
    free(pose_output_buffers);
    free(pose_input_buffers);
    if (pose_outputs) svp_acl_mdl_destroy_dataset(pose_outputs);
    if (pose_inputs) svp_acl_mdl_destroy_dataset(pose_inputs);
    if (pose_model.loaded) svp_acl_mdl_unload(pose_model.id);
    if (pose_model.desc) svp_acl_mdl_destroy_desc(pose_model.desc);
    if (pose_model.memory) svp_acl_rt_free(pose_model.memory);
    if (device_set) svp_acl_rt_reset_device(0);
    if (acl_initialized) svp_acl_finalize();
    return status;
}
