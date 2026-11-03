#include "layout.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void lp_copy(void * d, const void * s, uint64_t n) { check(cudaMemcpy(d,s,n,cudaMemcpyDeviceToDevice)); }
void lp_zero(void * d, uint64_t n) { check(cudaMemset(d,0,n)); }
void lp_device_sync() { check(cudaDeviceSynchronize()); }
template<bool Import> __global__ void kv_kernel(uint16_t * d,const uint16_t * s,int t,int h,int dim,int64_t n) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { auto j=lp_kv_index(i,t,h,dim); if(Import)d[j]=s[i];else d[i]=s[j]; }
}
__global__ void r_kernel(float * d,const float * s,int heads,int dim,int64_t n,bool importing) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){auto j=lp_recurrent_index(i,16,heads,dim);if(importing)d[j]=s[i];else d[i]=s[j];}
}
template<bool Import> __global__ void cast_kernel(void * out,const void * in,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        if(Import)((float*)out)[i]=__bfloat162float(((const __nv_bfloat16*)in)[i]);
        else ((__nv_bfloat16*)out)[i]=__float2bfloat16(((const float*)in)[i]);
    }
}
__global__ void conv_import_kernel(float * c4,float * c3,const void * s,int channels,bool fp32) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    int n=lp_c_channel(c,16,(channels-4096)/128,128);
    if(c<channels)for(int j=0;j<4;++j){float v=fp32?((const float*)s)[c*4+j]:__bfloat162float(((const __nv_bfloat16*)s)[c*4+j]);c4[n*4+j]=v;if(j)c3[n*3+j-1]=v;}
}
__global__ void conv_export_kernel(void * d,const float * s,int channels,bool fp32) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    int n=lp_c_channel(c,16,(channels-4096)/128,128);
    if(c<channels)for(int j=0;j<4;++j){float v=s[n*4+j];if(fp32)((float*)d)[c*4+j]=v;else ((__nv_bfloat16*)d)[c*4+j]=__float2bfloat16(v);}
}
__global__ void conv_update_kernel(float * d,const float * old,const float * raw,int channels,int committed,int start) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<channels)for(int j=0;j<4;++j){int t=lp_conv_time(committed,j);d[c*4+j]=t<4?old[c*4+t]:raw[(t-4-start)*channels+c];}
}
void lp_convert_kv(void * d,const void * s,int t,int h,int dim,bool in) {
    int64_t n=(int64_t)t*h*dim;
    if(in)kv_kernel<true><<<(n+255)/256,256>>>((uint16_t*)d,(const uint16_t*)s,t,h,dim,n);
    else kv_kernel<false><<<(n+255)/256,256>>>((uint16_t*)d,(const uint16_t*)s,t,h,dim,n);
    check(cudaGetLastError());
}
void lp_convert_r(void*d,const void*s,int h,int dim,bool importing){int n=h*dim*dim;r_kernel<<<(n+255)/256,256>>>((float*)d,(const float*)s,h,dim,n,importing);check(cudaGetLastError());}
void lp_bf16_to_float(float*d,const void*s,int n){cast_kernel<true><<<(n+255)/256,256>>>(d,s,n);check(cudaGetLastError());}
void lp_float_to_bf16(void*d,const float*s,int n){cast_kernel<false><<<(n+255)/256,256>>>(d,s,n);check(cudaGetLastError());}
void lp_conv_import(float*c4,float*c3,const void*s,int c,bool fp32){conv_import_kernel<<<(c+255)/256,256>>>(c4,c3,s,c,fp32);check(cudaGetLastError());}
void lp_conv_export(void*d,const float*s,int c,bool fp32){conv_export_kernel<<<(c+255)/256,256>>>(d,s,c,fp32);check(cudaGetLastError());}
void lp_conv_update(float*d,const float*o,const float*r,int c,int raw_count,int committed,int start){
    if(committed<1||committed>start+raw_count||committed-4>start+raw_count)throw std::runtime_error("invalid convolution boundary");
    conv_update_kernel<<<(c+255)/256,256>>>(d,o,r,c,committed,start);check(cudaGetLastError());
}
