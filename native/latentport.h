#pragma once
#include <stdint.h>
#ifdef _WIN32
#define LP_API __declspec(dllexport)
#else
#define LP_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Canonical tensors are contiguous CUDA:0. K/V=BF16, R=FP32;
// H/C accept bridge BF16 or lossless native FP32, distinguished by byte count.
// kind: 0=H, 1=K, 2=V, 3=C, 4=R. Shape is checked against the loaded model.
typedef struct { int32_t kind, layer; void * data; uint64_t bytes; } lp_field;
typedef struct {
    int32_t abi, layers, hidden, kv_heads, head_dim, conv_channels, recurrent_heads;
    int32_t mtp_heads, max_drafts, context_size, vocabulary;
    uint64_t auxiliary_bytes;
} lp_info;
typedef struct {
    int32_t count, proposed, accepted, warmup, clock;
    int32_t rolled_back;
} lp_receipt;
typedef struct {
    uint64_t prefill_tokens, committed_tokens, verified_tokens, head_calls, installs, exports;
    uint64_t model_bytes, context_bytes, compute_bytes, auxiliary_bytes;
    double last_install_s, last_head_s;
} lp_trace;
LP_API int lp_abi(void);
LP_API const char * lp_error(void);
LP_API void * lp_open(const char * path, int side, int context, int batch, int threads, int drafts);
LP_API void lp_close(void * handle);
LP_API int lp_get_info(void * handle, lp_info * out);
LP_API int lp_tokenize(void * handle, const char * text, int bytes, int32_t * out, int capacity);
LP_API int lp_reset(void * handle);
LP_API int lp_install(void * handle, const lp_field * fields, int count, int clock);
LP_API int lp_export(void * handle, const lp_field * fields, int count);
LP_API int lp_prefill(void * handle, const int32_t * ids, int count);
LP_API int lp_logits(void * handle, float * host_output, int capacity);
LP_API int lp_head(void * handle, const void * h, uint64_t bytes);
LP_API int lp_commit(void * handle, int32_t token);
LP_API int lp_greedy(void * handle, int limit, const int32_t * eos, int eos_count,
                     int32_t * output, lp_receipt * receipt);
LP_API int lp_clock(void * handle);
LP_API int lp_sync(void * handle);
LP_API int lp_get_trace(void * handle, lp_trace * out);
// Explicit correctness-test hook: replace proposals, never target verification.
LP_API int lp_test_force_drafts(void * handle, const int32_t * ids, int count);
#ifdef __cplusplus
}
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
// Called only while building the graph, not once per evaluated layer.
void lp_graph_tap(void * owner, ggml_context *, ggml_cgraph *, ggml_tensor *, int kind, int layer);
#endif
