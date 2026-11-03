#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define LP_HD __host__ __device__
#else
#define LP_HD
#endif
// Canonical [heads,time,dim] -> native [time,heads,dim].
LP_HD inline int64_t lp_kv_index(int64_t i, int64_t time, int heads, int dim) {
    const int64_t d = i % dim, t = (i / dim) % time, h = i / (dim * time);
    return (t * heads + h) * dim + d;
}
// GGML delta-net stores S transposed: native [head,value,key].
LP_HD inline int64_t lp_r_index(int64_t i, int dim) {
    return (i / (dim * dim)) * dim * dim + (i % dim) * dim + (i / dim) % dim;
}
// Pinned Qwen GGUF conversion changes grouped V heads to tiled V heads.
LP_HD inline int lp_v_head(int h,int key_heads,int value_heads) {
    const int ratio=value_heads/key_heads;
    return (h%ratio)*key_heads+h/ratio;
}
LP_HD inline int lp_c_channel(int channel,int key_heads,int value_heads,int dim) {
    const int base=2*key_heads*dim;
    return channel<base?channel:base+lp_v_head((channel-base)/dim,key_heads,value_heads)*dim+(channel-base)%dim;
}
LP_HD inline int64_t lp_recurrent_index(int64_t i,int key_heads,int value_heads,int dim) {
    return (int64_t)lp_v_head((int)(i/(dim*dim)),key_heads,value_heads)*dim*dim+(i%dim)*dim+(i/dim)%dim;
}
// Index into [old four positions | freshly projected tokens].
LP_HD inline int lp_conv_time(int committed, int lane) { return committed + lane; }
void lp_copy(void * dst, const void * src, uint64_t bytes);
void lp_zero(void * dst, uint64_t bytes);
void lp_convert_kv(void * dst, const void * src, int time, int heads, int dim, bool importing);
void lp_convert_r(void * dst, const void * src, int heads, int dim, bool importing);
void lp_bf16_to_float(float * dst, const void * src, int count);
void lp_float_to_bf16(void * dst, const float * src, int count);
void lp_conv_import(float * c4, float * c3, const void * src, int channels, bool fp32);
void lp_conv_export(void * dst, const float * c4, int channels, bool fp32);
void lp_conv_update(float * dst, const float * old, const float * raw,
                    int channels, int raw_count, int committed, int raw_start);
void lp_device_sync();
