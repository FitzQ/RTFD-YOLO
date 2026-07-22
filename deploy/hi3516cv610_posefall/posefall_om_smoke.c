#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "svp_acl_rt.h"

#define USER_INPUTS 1
#define EXTRA_INPUTS 2

typedef struct {
    void *memory;
    size_t size;
    size_t stride;
    svp_acl_data_buffer *buffer;
} owned_buffer;

typedef struct {
    uint32_t id;
    void *memory;
    size_t size;
    svp_acl_mdl_desc *desc;
    int loaded;
} loaded_model;

static int check(svp_acl_error code, const char *what)
{
    if (code == SVP_ACL_SUCCESS) {
        return 0;
    }
    fprintf(stderr, "ERROR: %s failed, code=%d\n", what, code);
    return -1;
}

static double monotonic_ms(void)
{
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int load_model(const char *path, loaded_model *model)
{
    FILE *file = fopen(path, "rb");
    long file_size;
    if (file == NULL) {
        perror(path);
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: unable to determine model size: %s\n", path);
        fclose(file);
        return -1;
    }
    model->size = (size_t)file_size;
    if (check(svp_acl_rt_malloc(&model->memory, model->size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY),
        "allocate model memory") != 0) {
        fclose(file);
        return -1;
    }
    if (fread(model->memory, 1, model->size, file) != model->size) {
        fprintf(stderr, "ERROR: unable to read model: %s\n", path);
        fclose(file);
        return -1;
    }
    fclose(file);
    if (check(svp_acl_mdl_load_from_mem(model->memory, model->size, &model->id), "load model") != 0) {
        return -1;
    }
    model->loaded = 1;
    model->desc = svp_acl_mdl_create_desc();
    if (model->desc == NULL || check(svp_acl_mdl_get_desc(model->desc, model->id), "get model desc") != 0) {
        return -1;
    }
    return 0;
}

static size_t data_type_bytes(svp_acl_data_type type)
{
    size_t bits = svp_acl_data_type_size(type);
    return (bits + 7U) / 8U;
}

static void print_dims(const char *kind, size_t index, const char *name, const svp_acl_mdl_io_dims *dims,
    svp_acl_data_type type, size_t size, size_t stride)
{
    size_t i;
    printf("%s[%zu] name=%s shape=", kind, index, name == NULL ? "?" : name);
    for (i = 0; i < dims->dim_count; ++i) {
        printf("%s%lld", i == 0 ? "[" : ",", (long long)dims->dims[i]);
    }
    printf("] dtype=%d bytes=%zu stride=%zu\n", (int)type, size, stride);
}

static int allocate_buffer(owned_buffer *owned, size_t size, size_t stride)
{
    owned->size = size;
    owned->stride = stride;
    if (check(svp_acl_rt_malloc(&owned->memory, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY), "allocate tensor") != 0) {
        return -1;
    }
    memset(owned->memory, 0, size);
    owned->buffer = svp_acl_create_data_buffer(owned->memory, size, stride);
    if (owned->buffer == NULL) {
        fprintf(stderr, "ERROR: create data buffer failed\n");
        return -1;
    }
    return 0;
}

static int load_strided_input(const char *path, void *memory, size_t buffer_size,
    const svp_acl_mdl_io_dims *dims, svp_acl_data_type type, size_t stride)
{
    size_t rows = 1;
    size_t line_bytes;
    size_t i;
    FILE *file;
    if (dims->dim_count == 0) {
        return -1;
    }
    for (i = 0; i + 1 < dims->dim_count; ++i) {
        rows *= (size_t)dims->dims[i];
    }
    line_bytes = (size_t)dims->dims[dims->dim_count - 1] * data_type_bytes(type);
    if (line_bytes > stride) {
        fprintf(stderr, "ERROR: input line %zu is larger than stride %zu\n", line_bytes, stride);
        return -1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return -1;
    }
    /* Static AIPP exposes the logical NCHW shape but its actual user buffer is
     * one NV12 frame (width * height * 3 / 2). Prefer an exact-size raw copy;
     * retain the row-wise path below for tensors whose last axis is padded. */
    if (fseek(file, 0, SEEK_END) == 0 && (size_t)ftell(file) == buffer_size &&
        fseek(file, 0, SEEK_SET) == 0) {
        if (fread(memory, 1, buffer_size, file) != buffer_size) {
            fclose(file);
            return -1;
        }
        fclose(file);
        return 0;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    for (i = 0; i < rows; ++i) {
        if (fread((uint8_t *)memory + i * stride, 1, line_bytes, file) != line_bytes) {
            fprintf(stderr, "ERROR: input ended early at row %zu/%zu\n", i, rows);
            fclose(file);
            return -1;
        }
    }
    if (fgetc(file) != EOF) {
        fprintf(stderr, "ERROR: input has trailing data; expected %zu bytes\n", rows * line_bytes);
        fclose(file);
        return -1;
    }
    fclose(file);
    return 0;
}

static int dump_strided_output(const char *path, const void *memory, const svp_acl_mdl_io_dims *dims,
    svp_acl_data_type type, size_t stride)
{
    size_t rows = 1;
    size_t line_bytes;
    size_t i;
    FILE *file;
    for (i = 0; i + 1 < dims->dim_count; ++i) {
        rows *= (size_t)dims->dims[i];
    }
    line_bytes = (size_t)dims->dims[dims->dim_count - 1] * data_type_bytes(type);
    file = fopen(path, "wb");
    if (file == NULL) {
        perror(path);
        return -1;
    }
    for (i = 0; i < rows; ++i) {
        if (fwrite((const uint8_t *)memory + i * stride, 1, line_bytes, file) != line_bytes) {
            fclose(file);
            return -1;
        }
    }
    fclose(file);
    printf("wrote %s (%zu bytes)\n", path, rows * line_bytes);
    return 0;
}

static void free_buffers(svp_acl_mdl_dataset *dataset, owned_buffer *buffers, size_t count)
{
    size_t i;
    (void)dataset;
    for (i = 0; i < count; ++i) {
        if (buffers[i].buffer != NULL) {
            (void)svp_acl_destroy_data_buffer(buffers[i].buffer);
        }
        if (buffers[i].memory != NULL) {
            (void)svp_acl_rt_free(buffers[i].memory);
        }
    }
}

int main(int argc, char **argv)
{
    loaded_model model = {0};
    svp_acl_mdl_dataset *inputs = NULL;
    svp_acl_mdl_dataset *outputs = NULL;
    owned_buffer *input_buffers = NULL;
    owned_buffer *output_buffers = NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    size_t i;
    int iterations = 1;
    int warmup = 0;
    int run;
    int status = 1;

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s model.om input.bin [iterations]\n", argv[0]);
        return 2;
    }
    if (argc == 4) {
        iterations = atoi(argv[3]);
        if (iterations < 1 || iterations > 100000) {
            fprintf(stderr, "ERROR: iterations must be in 1..100000\n");
            return 2;
        }
        warmup = iterations > 1 ? 5 : 0;
    }
    if (check(svp_acl_init(NULL), "svp_acl_init") != 0 || check(svp_acl_rt_set_device(0), "set device") != 0) {
        goto cleanup;
    }
    if (load_model(argv[1], &model) != 0) {
        goto cleanup;
    }
    input_count = svp_acl_mdl_get_num_inputs(model.desc);
    output_count = svp_acl_mdl_get_num_outputs(model.desc);
    printf("model=%s inputs=%zu outputs=%zu\n", argv[1], input_count, output_count);
    if (input_count < USER_INPUTS + EXTRA_INPUTS || output_count == 0) {
        fprintf(stderr, "ERROR: expected at least one user input plus two runtime inputs\n");
        goto cleanup;
    }

    input_buffers = calloc(input_count, sizeof(*input_buffers));
    output_buffers = calloc(output_count, sizeof(*output_buffers));
    inputs = svp_acl_mdl_create_dataset();
    outputs = svp_acl_mdl_create_dataset();
    if (input_buffers == NULL || output_buffers == NULL || inputs == NULL || outputs == NULL) {
        fprintf(stderr, "ERROR: allocate dataset failed\n");
        goto cleanup;
    }

    for (i = 0; i < input_count; ++i) {
        svp_acl_mdl_io_dims dims = {0};
        svp_acl_data_type type = svp_acl_mdl_get_input_data_type(model.desc, i);
        size_t size = svp_acl_mdl_get_input_size_by_index(model.desc, i);
        size_t stride = svp_acl_mdl_get_input_default_stride(model.desc, i);
        if (check(svp_acl_mdl_get_input_dims(model.desc, i, &dims), "get input dims") != 0 ||
            allocate_buffer(&input_buffers[i], size, stride) != 0 ||
            check(svp_acl_mdl_add_dataset_buffer(inputs, input_buffers[i].buffer), "add input") != 0) {
            goto cleanup;
        }
        print_dims("input", i, svp_acl_mdl_get_input_name_by_index(model.desc, i), &dims, type, size, stride);
        if (i == 0 && load_strided_input(argv[2], input_buffers[i].memory, size, &dims, type, stride) != 0) {
            goto cleanup;
        }
    }

    for (i = 0; i < output_count; ++i) {
        svp_acl_mdl_io_dims dims = {0};
        svp_acl_data_type type = svp_acl_mdl_get_output_data_type(model.desc, i);
        size_t size = svp_acl_mdl_get_output_size_by_index(model.desc, i);
        size_t stride = svp_acl_mdl_get_output_default_stride(model.desc, i);
        if (check(svp_acl_mdl_get_output_dims(model.desc, i, &dims), "get output dims") != 0 ||
            allocate_buffer(&output_buffers[i], size, stride) != 0 ||
            check(svp_acl_mdl_add_dataset_buffer(outputs, output_buffers[i].buffer), "add output") != 0) {
            goto cleanup;
        }
        print_dims("output", i, svp_acl_mdl_get_output_name_by_index(model.desc, i), &dims, type, size, stride);
    }

    for (run = 0; run < warmup; ++run) {
        if (check(svp_acl_mdl_execute(model.id, inputs, outputs), "model warmup") != 0) {
            goto cleanup;
        }
    }
    {
        double started_ms = monotonic_ms();
        double elapsed_ms;
        for (run = 0; run < iterations; ++run) {
            if (check(svp_acl_mdl_execute(model.id, inputs, outputs), "model execute") != 0) {
                goto cleanup;
            }
        }
        elapsed_ms = monotonic_ms() - started_ms;
        printf("MODEL_BENCH iterations=%d warmup=%d total_ms=%.3f mean_ms=%.6f\n",
            iterations, warmup, elapsed_ms, elapsed_ms / (double)iterations);
    }
    for (i = 0; i < output_count; ++i) {
        svp_acl_mdl_io_dims dims = {0};
        svp_acl_data_type type = svp_acl_mdl_get_output_data_type(model.desc, i);
        char output_path[64];
        (void)svp_acl_mdl_get_output_dims(model.desc, i, &dims);
        snprintf(output_path, sizeof(output_path), "output_%zu.bin", i);
        if (dump_strided_output(output_path, output_buffers[i].memory, &dims, type, output_buffers[i].stride) != 0) {
            goto cleanup;
        }
    }
    printf("MODEL_SMOKE_OK\n");
    status = 0;

cleanup:
    free_buffers(outputs, output_buffers, output_count);
    free_buffers(inputs, input_buffers, input_count);
    free(output_buffers);
    free(input_buffers);
    if (outputs != NULL) {
        (void)svp_acl_mdl_destroy_dataset(outputs);
    }
    if (inputs != NULL) {
        (void)svp_acl_mdl_destroy_dataset(inputs);
    }
    if (model.loaded) {
        (void)svp_acl_mdl_unload(model.id);
    }
    if (model.desc != NULL) {
        (void)svp_acl_mdl_destroy_desc(model.desc);
    }
    if (model.memory != NULL) {
        (void)svp_acl_rt_free(model.memory);
    }
    (void)svp_acl_rt_reset_device(0);
    (void)svp_acl_finalize();
    return status;
}
