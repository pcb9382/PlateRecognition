#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix {
  float value[6];
};


void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,cudaStream_t stream);

void preprocess_kernel_img_yolov5_plate(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,cudaStream_t stream);

void preprocess_kernel_img_ppliteseg(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,cudaStream_t stream);

void postprocess_kernel_img_ppliteseg(uint32_t* src, int src_width, int src_height,uint32_t* dst, int dst_width, int dst_height,cudaStream_t stream);

void preprocess_kernel_img_simple(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream);

void preprocess_kernel_img_reorder(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream);

void maxindex_kernel(float* src, unsigned char* dst, int src_width, int src_height, int num_class, cudaStream_t stream);

#endif  // __PREPROCESS_H
