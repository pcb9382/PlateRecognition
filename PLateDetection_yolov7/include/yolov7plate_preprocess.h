#ifndef __YOLOV7_PREPROCESS_H
#define __YOLOV7_PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


// struct AffineMatrix{
//     float value[6];
// };


void yolov7plate_preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float*d2i,cudaStream_t stream);
#endif  // __PREPROCESS_H
