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
#define BOX_RGN_HANDLE 20
#define FALL_FEATURE_DIM 56
#define FALL_WINDOW 60
#define FALL_STRIDE 15
#define FALL_THRESHOLD 0.5f
#define CAMERA_FRAME_US 33333ULL
#define TRACK_GAP_US 500000ULL

typedef struct {
    int found;
    float confidence;
    float x1, y1, x2, y2;
    float kx[KEYPOINTS], ky[KEYPOINTS], kc[KEYPOINTS];
} pose_result;

typedef struct {
    int fall_frames;
    int normal_frames;
    int is_fall;
} fall_state;

typedef struct {
    float rows[FALL_WINDOW][FALL_FEATURE_DIM];
    int count;
    int samples_since_head;
    uint64_t last_pose_pts;
} pose_history;

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

static void decode_best_pose(const tensor_buffer *output, int frame_index,
                             uint64_t pts, double inference_ms,
                             pose_result *result, int print_details)
{
    memset(result, 0, sizeof(*result));
    int best = -1;
    float best_conf = 0.0f;
    for (int i = 0; i < POSE_CANDIDATES; ++i) {
        float conf = output_value(output, 4, i);
        if (isfinite(conf) && conf > best_conf) {
            best_conf = conf;
            best = i;
        }
    }
    if (best < 0 || best_conf < 0.25f) {
        if (print_details) {
            printf("POSE frame=%d pts=%llu infer_ms=%.2f person=none max_conf=%.6f\n",
                   frame_index, (unsigned long long)pts, inference_ms, best_conf);
        }
        return;
    }

    float x = output_value(output, 0, best);
    float y = output_value(output, 1, best);
    float w = output_value(output, 2, best);
    float h = output_value(output, 3, best);
    result->found = 1;
    result->confidence = best_conf;
    result->x1 = x - w * 0.5f;
    result->y1 = y - h * 0.5f;
    result->x2 = x + w * 0.5f;
    result->y2 = y + h * 0.5f;
    if (print_details) {
        printf("POSE frame=%d pts=%llu infer_ms=%.2f conf=%.3f "
               "box=%.1f,%.1f,%.1f,%.1f",
               frame_index, (unsigned long long)pts, inference_ms, best_conf,
               x - w * 0.5f, y - h * 0.5f, x + w * 0.5f, y + h * 0.5f);
    }
    for (int k = 0; k < KEYPOINTS; ++k) {
        float kx = output_value(output, 5 + k * 3, best);
        float ky = output_value(output, 6 + k * 3, best);
        float kc = output_value(output, 7 + k * 3, best);
        result->kx[k] = kx;
        result->ky[k] = ky;
        result->kc[k] = kc;
        if (print_details && kc >= 0.25f) {
            printf(" k%d=%.0f,%.0f,%.2f", k, kx, ky, kc);
        }
    }
    if (print_details) {
        putchar('\n');
        fflush(stdout);
    }
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
    ++history->samples_since_head;
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

static int update_model_fall_state(float probability, fall_state *state)
{
    if (probability >= FALL_THRESHOLD) {
        if (state->fall_frames < 1000) ++state->fall_frames;
        state->normal_frames = 0;
        if (state->fall_frames >= 2) state->is_fall = 1;
    } else {
        state->fall_frames = 0;
        if (state->normal_frames < 1000) ++state->normal_frames;
        if (state->normal_frames >= 3) state->is_fall = 0;
    }
    return state->is_fall;
}

static int start_box_overlay(const ot_mpp_chn *target)
{
    ot_rgn_attr region;
    memset(&region, 0, sizeof(region));
    region.type = OT_RGN_COVER;
    td_s32 rc = ss_mpi_rgn_create(BOX_RGN_HANDLE, &region);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "WARN create box overlay rc=0x%08x\n", (unsigned)rc);
        return -1;
    }

    ot_rgn_chn_attr display;
    memset(&display, 0, sizeof(display));
    display.is_show = TD_FALSE;
    display.type = OT_RGN_COVER;
    display.attr.cover_chn.coord = OT_COORD_ABS;
    display.attr.cover_chn.layer = 0;
    display.attr.cover_chn.cover.type = OT_COVER_RECT;
    display.attr.cover_chn.cover.color = 0x00ff00;
    display.attr.cover_chn.cover.rect_attr.rect.x = 0;
    display.attr.cover_chn.cover.rect_attr.rect.y = 0;
    display.attr.cover_chn.cover.rect_attr.rect.width = 100;
    display.attr.cover_chn.cover.rect_attr.rect.height = 100;
    display.attr.cover_chn.cover.rect_attr.is_solid = TD_FALSE;
    display.attr.cover_chn.cover.rect_attr.thick = 4;
    rc = ss_mpi_rgn_attach_to_chn(BOX_RGN_HANDLE, target, &display);
    if (rc != TD_SUCCESS) {
        fprintf(stderr, "WARN attach box overlay rc=0x%08x\n", (unsigned)rc);
        ss_mpi_rgn_destroy(BOX_RGN_HANDLE);
        return -1;
    }
    printf("OVERLAY_READY target=VPSS:0:1 video=%ux%u\n",
           g_video_width, g_video_height);
    return 0;
}

static void update_box_overlay(const ot_mpp_chn *target, const pose_result *pose,
                               int is_fall)
{
    ot_rgn_chn_attr display;
    memset(&display, 0, sizeof(display));
    display.type = OT_RGN_COVER;
    display.attr.cover_chn.coord = OT_COORD_ABS;
    display.attr.cover_chn.layer = 0;
    display.attr.cover_chn.cover.type = OT_COVER_RECT;
    display.attr.cover_chn.cover.color = is_fall ? 0xff0000 : 0x00ff00;
    display.attr.cover_chn.cover.rect_attr.is_solid = TD_FALSE;
    display.attr.cover_chn.cover.rect_attr.thick = is_fall ? 8 : 4;
    display.attr.cover_chn.cover.rect_attr.rect.x = 0;
    display.attr.cover_chn.cover.rect_attr.rect.y = 0;
    display.attr.cover_chn.cover.rect_attr.rect.width = 2;
    display.attr.cover_chn.cover.rect_attr.rect.height = 2;
    display.is_show = pose->found ? TD_TRUE : TD_FALSE;

    if (pose->found) {
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
    td_s32 rc = ss_mpi_rgn_set_chn_display_attr(BOX_RGN_HANDLE, target, &display);
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

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s pose_camera.om posefall_head.om [frame_count]\n", argv[0]);
        return 2;
    }
    int frame_limit = argc == 4 ? atoi(argv[3]) : 0;
    int status = 1;
    int ai_channel_started = 0;
    int overlay_started = 0;
    int acl_initialized = 0;
    int device_set = 0;
    model_handle pose_model = {0}, head_model = {0};
    svp_acl_mdl_dataset *pose_inputs = NULL, *pose_outputs = NULL;
    svp_acl_mdl_dataset *head_inputs = NULL, *head_outputs = NULL;
    tensor_buffer *pose_input_buffers = NULL, *pose_output_buffers = NULL;
    tensor_buffer *head_input_buffers = NULL, *head_output_buffers = NULL;
    size_t pose_input_count = 0, pose_output_count = 0;
    size_t head_input_count = 0, head_output_count = 0;
    fall_state detector_state = {0};
    pose_history history = {0};
    int previous_fall_state = 0;
    float fall_probability = 0.0f;
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
    if (load_model(argv[1], &pose_model) != 0) goto cleanup;
    if (create_datasets(&pose_model, &pose_inputs, &pose_input_buffers,
                        &pose_input_count, &pose_outputs, &pose_output_buffers,
                        &pose_output_count, NV12_BYTES) != 0) goto cleanup;
    if (load_model(argv[2], &head_model) != 0) goto cleanup;
    if (create_datasets(&head_model, &head_inputs, &head_input_buffers,
                        &head_input_count, &head_outputs, &head_output_buffers,
                        &head_output_count,
                        FALL_WINDOW * FALL_FEATURE_DIM * sizeof(float)) != 0) goto cleanup;
    if (start_ai_channel() != 0) goto cleanup;
    ai_channel_started = 1;
    if (start_box_overlay(&overlay_target) == 0) overlay_started = 1;
    printf("POSEFALL_READY pose=%s head=%s channel=%d size=%dx%d window=%d stride=%d\n",
           argv[1], argv[2], VPSS_AI_CHN, MODEL_WIDTH, MODEL_HEIGHT,
           FALL_WINDOW, FALL_STRIDE);

    for (int frame = 1; !g_stop && (frame_limit <= 0 || frame <= frame_limit); ++frame) {
        uint64_t pts = 0;
        double preprocess_begin = monotonic_ms();
        if (copy_camera_frame(pose_input_buffers[0].memory,
                              pose_input_buffers[0].size, &pts) != 0) continue;
        double preprocess_ms = monotonic_ms() - preprocess_begin;
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
        if (frame == 1) print_output_stats(&pose_output_buffers[0]);
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
        int report = frame_limit > 0 || frame == 1 || frame % 30 == 0;
        if (report) {
            printf("TIMING frame=%d preprocess_ms=%.2f infer_ms=%.2f\n",
                   frame, preprocess_ms, elapsed);
        }
        pose_result pose;
        decode_best_pose(&pose_output_buffers[0], frame, pts, elapsed, &pose, report);
        int head_ran = 0;
        double head_ms = 0.0;
        if (pose.found) {
            float features[FALL_FEATURE_DIM];
            pose_to_features(&pose, features);
            int repeats = 1;
            if (history.last_pose_pts != 0 && pts > history.last_pose_pts) {
                uint64_t gap = pts - history.last_pose_pts;
                if (gap > TRACK_GAP_US) {
                    reset_pose_history(&history);
                } else {
                    repeats = (int)((gap + CAMERA_FRAME_US / 2) / CAMERA_FRAME_US);
                    if (repeats < 1) repeats = 1;
                    if (repeats > 4) repeats = 4;
                }
            }
            for (int sample = 0; sample < repeats; ++sample) {
                append_history_row(&history, features);
            }
            history.last_pose_pts = pts;
            if (history.samples_since_head >= FALL_STRIDE) {
                if (execute_fall_head(&head_model, head_inputs,
                                      head_input_buffers, head_outputs,
                                      head_output_buffers, &history,
                                      &fall_probability, &head_ms) != 0) break;
                history.samples_since_head %= FALL_STRIDE;
                head_ran = 1;
                update_model_fall_state(fall_probability, &detector_state);
            }
        } else if (history.last_pose_pts != 0 &&
                   pts > history.last_pose_pts + TRACK_GAP_US) {
            reset_pose_history(&history);
            memset(&detector_state, 0, sizeof(detector_state));
            fall_probability = 0.0f;
        }
        int is_fall = detector_state.is_fall;
        if (head_ran) {
            printf("HEAD frame=%d prob=%.4f infer_ms=%.2f history=%d state=%s\n",
                   frame, fall_probability, head_ms, history.count,
                   is_fall ? "FALL" : "NORMAL");
        }
        if (report || is_fall != previous_fall_state) {
            printf("STATE frame=%d state=%s probability=%.4f history=%d "
                   "fall_votes=%d normal_votes=%d\n",
                   frame, is_fall ? "FALL" : "NORMAL", fall_probability,
                   history.count, detector_state.fall_frames,
                   detector_state.normal_frames);
        }
        previous_fall_state = is_fall;
        if (overlay_started) {
            update_box_overlay(&overlay_target, &pose, is_fall);
        }
    }
    status = 0;

cleanup:
    if (overlay_started) {
        ss_mpi_rgn_detach_from_chn(BOX_RGN_HANDLE, &overlay_target);
        ss_mpi_rgn_destroy(BOX_RGN_HANDLE);
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
