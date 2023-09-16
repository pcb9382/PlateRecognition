#include "yolov5plate_preprocess.h"
#include <opencv2/opencv.hpp>

__global__ void warpaffine_kernel( 
    uint8_t* src, int src_line_size, int src_width, 
    int src_height, float* dst, int dst_width, 
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void preprocess_kernel_img(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {
    AffineMatrix s2d,d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);

}

__global__ void warpaffine_kernel_ppliteseg_pre( uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, 
    int dst_height, uint8_t const_value_st,AffineMatrix d2s, int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) 
    {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } 
    else 
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) 
        {
            if (x_low >= 0)
            {
                v1 = src + y_low * src_line_size + x_low * 3;
            }
            if (x_high < src_width)
            {
                v2 = src + y_low * src_line_size + x_high * 3;
            }
        }
        if (y_high < src_height) 
        {
            if (x_low >= 0)
            {
                v3 = src + y_high * src_line_size + x_low * 3;
            }
            if (x_high < src_width)
            {   
                v4 = src + y_high * src_line_size + x_high * 3;
            }
        }
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization,mean,var
    c0 = ((c0 / 255.0f)-0.5f)/0.5f;
    c1 = ((c1 / 255.0f)-0.5f)/0.5f;
    c2 = ((c2 / 255.0f)-0.5f)/0.5f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}
__global__ void warpaffine_kernel_ppliteseg_post(float* src, int src_line_size, int src_width, int src_height, uint32_t* dst, int dst_width, 
    int dst_height, float const_value_st,AffineMatrix d2s, int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];  //0.8
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];  //1.06667
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;     //477
    int dy = position / dst_width;     //430
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;//382.1
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;//459.16
    uint32_t c0; //c1, c2

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) 
    {
        c0 = const_value_st;
    } 
    else 
    {
        int y_low = floorf(src_y);   //459
        int x_low = floorf(src_x);   //382
        int y_high = y_low + 1;      //460
        int x_high = x_low + 1;      //383

        float const_value[] = {const_value_st};//, const_value_st, const_value_st
        float ly = src_y - y_low;   //0.16
        float lx = src_x - x_low;   //0.1
        float hy = 1 - ly;          //0.84
        float hx = 1 - lx;          //0.9
        float w1 = hy * hx;         //0.756
        float w2 = hy * lx;         //0.084
        float w3 = ly * hx;         //0.144
        float w4 = ly * lx;         //0.016
        float* v1 = const_value;
        float* v2 = const_value;
        float* v3 = const_value;
        float* v4 = const_value;

        if (y_low >= 0) 
        {
            if (x_low >= 0)
            {
                v1 = src + y_low * src_line_size + x_low ;//* 3;
            }
            if (x_high < src_width)
            {
                v2 = src + y_low * src_line_size + x_high ;//* 3;
            }
        }
        if (y_high < src_height) 
        {
            if (x_low >= 0)
            {
                v3 = src + y_high * src_line_size + x_low ;//* 3;
            }
            if (x_high < src_width)
            {   
                v4 = src + y_high * src_line_size + x_high ;//* 3;
            }
        }
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    }

    uint32_t* pdst_c0 = dst + dy * dst_width + dx;
    *pdst_c0 = c0;
}
__global__ void warpaffine_kernel_ppliteseg_post1(uint32_t* src, int src_line_size, int src_width, int src_height, uint32_t* dst, int dst_width, 
    int dst_height, uint32_t const_value_st,AffineMatrix d2s, int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];  //0.8
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];  //1.06667
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;     //477
    int dy = position / dst_width;     //430
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;//382.1
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;//459.16
    uint32_t c0; //c1, c2

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) 
    {
        c0 = const_value_st;
    } 
    else 
    {
        int y_low = floorf(src_y);   //459
        int x_low = floorf(src_x);   //382
        //int y_high = y_low + 1;      //460
        //int x_high = x_low + 1;      //383

        uint32_t const_value[] = {const_value_st};//, const_value_st, const_value_st
        //float ly = src_y - y_low;   //0.16
        //float lx = src_x - x_low;   //0.1
        //float hy = 1 - ly;          //0.84
        //float hx = 1 - lx;          //0.9
        //float w1 = hy * hx;         //0.756
        //float w2 = hy * lx;         //0.084
        //float w3 = ly * hx;         //0.144
        //float w4 = ly * lx;         //0.016
        uint32_t* v1 = const_value;
        uint32_t* v2 = const_value;
        uint32_t* v3 = const_value;
        uint32_t* v4 = const_value;

        if (y_low >= 0&&y_low<src_height) 
        {
            if (x_low >= 0&&x_low<src_width)
            {
                v1 = src + y_low * src_line_size + x_low ;//* 3;
                c0=v1[0];
            }
            else
            {
                c0=0;
            }
        }
        else
        {
            c0=0;
        }
    }

    uint32_t* pdst_c0 = dst + dy * dst_width + dx;
    *pdst_c0 = c0;
}
void preprocess_kernel_img_ppliteseg(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,cudaStream_t stream)
{
    AffineMatrix s2d,d2s;
    //float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);
    s2d.value[0] = (float)dst_width/src_width;//scale;
    s2d.value[1] = 0;
    s2d.value[2] = 0;//-scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = (float)dst_height/src_height;//scale;
    s2d.value[5] = 0;//-scale * src_height * 0.5 + dst_height * 0.5;
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));
    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel_ppliteseg_pre<<<blocks, threads, 0, stream>>>(src,src_width*3, src_width,src_height,dst,dst_width,dst_height, 128, d2s, jobs);
}

void postprocess_kernel_img_ppliteseg(uint32_t* src, int src_width, int src_height,uint32_t* dst, int dst_width, int dst_height,cudaStream_t stream)
{
    AffineMatrix s2d,d2s;
    //float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);
    s2d.value[0] = (float)dst_width/src_width;//scale;
    s2d.value[1] = 0;
    s2d.value[2] = 0;//-scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = (float)dst_height/src_height;//scale;
    s2d.value[5] = 0;//-scale * src_height * 0.5 + dst_height * 0.5;
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    //std::cout<< "m_x1:"<<d2s.value[0]<<" m_y1:"<<d2s.value[1]<<" m_z1:"<<d2s.value[2]
    //        <<" m_x2:"<<d2s.value[3]<<" m_y2:"<<d2s.value[4]<<" m_z2:"<<d2s.value[5]<<std::endl;
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));
    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel_ppliteseg_post1<<<blocks, threads, 0, stream>>>(src,src_width, src_width,src_height,dst,dst_width,dst_height,0.0, d2s, jobs);
}

//=================yolov5_face==========================================================================================================
__global__ void warpaffine_kernel_yolov5_plate(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, 
    int dst_height, uint8_t const_value_st,AffineMatrix d2s, int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void preprocess_kernel_img_yolov5_plate(uint8_t* src, int src_width, int src_height,float* dst, int dst_width, int dst_height,cudaStream_t stream)
{
    AffineMatrix s2d,d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel_yolov5_plate<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width,
        src_height, dst, dst_width,
        dst_height, 114, d2s, jobs);
}
//=================yolov5_face==========================================================================================================

__global__ void warpaffine_kernel_simple( 
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % src_width;
    int dy = position / src_width;

    float c0, c1, c2;

    uint8_t* v = src + dy * src_line_size + dx * 3;

    c0 = v[0];
    c1 = v[1];
    c2 = v[2];

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = (c0 / 255.0f - 0.485) / 0.229;
    c1 = (c1 / 255.0f - 0.456) / 0.224;
    c2 = (c2 / 255.0f - 0.406) / 0.225;

    //rgbrgbrgb to rrrgggbbb
    int area = src_width * src_height;
    float* pdst_c0 = dst + dy * src_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}
__global__ void warpaffine_kernel_reorder( 
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % src_width;
    int dy = position / src_width;

    float c0, c1, c2;

    uint8_t* v = src + dy * src_line_size + dx * 3;

    c0 = v[0];
    c1 = v[1];
    c2 = v[2];

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    // c0 = (c0 / 255.0f - 0.485) / 0.229;
    // c1 = (c1 / 255.0f - 0.456) / 0.224;
    // c2 = (c2 / 255.0f - 0.406) / 0.225;

    //rgbrgbrgb to rrrgggbbb
    int area = src_width * src_height;
    float* pdst_c0 = dst + dy * src_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

__global__ void warpmax_kernel
( 
    float* src, int src_width, int src_height, unsigned char* dst, int edge, int num_class) 
    {
    unsigned int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % src_width;
    int dy = position / src_width;

    float* v = src + dy * src_width + dx;
    unsigned char* dv = dst + dy * src_width + dx;
    float max = v[0];
    *dv = 0;

    for(unsigned char idx = 1; idx < num_class; idx++)
    {
        if(v[0 + idx * edge] > max) 
        {
            max = v[0 + idx * edge];
            *dv = idx;
        }   
    }
}

void preprocess_kernel_img_simple(uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream)
{

    int jobs = src_width * src_height;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel_simple<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width, src_height, dst, jobs);
}

void preprocess_kernel_img_reorder(
    uint8_t* src, float* dst, int src_width, int src_height, cudaStream_t stream) {

    int jobs = src_width * src_height;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel_reorder<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width, src_height, dst, jobs);
}

void maxindex_kernel(float* src, unsigned char* dst, int src_width, int src_height, int num_class, cudaStream_t stream) {
    int jobs = src_width * src_height;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpmax_kernel<<<blocks, threads, 0, stream>>>(
        src, src_width, src_height, dst, jobs, num_class);
}