#pragma once
#ifndef __DETECTOR_YOLOV7Plate_H__
#define __DETECTOR_YOLOV7Plate_H__
#include <memory>
#include <vector>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include <dirent.h>
#include "NvInfer.h"
#include "yolov7plate_preprocess.h"
#include "yolov7plate_postprocess.h"
#include "ONNX2TRT.h"
#include "DataTypes_Plate.h"
#define USE_FP16
#define MAX_IMAGE_INPUT_SIZE_THRESH 8000 * 6000
#define MAX_OBJECTS 2048
#define NMS_THRESH 0.1
#define BBOX_CONF_THRESH 0.4
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;
// struct affineMatrix  //letter_box  仿射变换矩阵
// {
//     float i2d[6];       //仿射变换正变换
//     float d2i[6];       //仿射变换逆变换
// };
// struct bbox 
// {
//     float x1,x2,y1,y2;
//     float landmarks[10]; //5个关键点
//     float score;
// };
// const float color_list[5][3] =
// {
//     {255, 0, 0},
//     {0, 255, 0},
//     {0, 0, 255},
//     {0, 255, 255},
//     {255,255,0},
// };
class Detector_Yolov7Plate
{ 
public:
    Detector_Yolov7Plate();
    ~Detector_Yolov7Plate();
    HZFLAG InitDetector_Yolov7Plate(Config*config);
    HZFLAG Detect_Yolov7Plate(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>&dets);
    HZFLAG ReleaseDetector_Yolov7Plate();

private:
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* buffers[2];
    int inputIndex;
    int outputIndex;
    cudaStream_t stream;
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    float *decode_ptr_device = nullptr;
    float **decode_ptr_host = nullptr;
    float *pre_predict=nullptr;
    float **affine_matrix_d2i_device = nullptr;         //放射变换矩阵 device
    float *affine_matrix_d2i_host = nullptr;            //仿射变换矩阵 host
    int gpu_id;                                         // GPU id
    float conf_thresh;
    float nms_thresh;
    int INPUT_H;                                        // H, W must be able to  be divided by 32.
    int INPUT_W;
    int OUTPUT_SIZE;
    char* INPUT_BLOB_NAME;
    char* OUTPUT_BLOB_NAME;
    char *trtModelStream{nullptr};
    int CKPT_NUM;
    int batch_size;
    int NUM_BOX_ELEMENT;
    int OUTPUT_CANDIDATES;
    int NUM_CLASSES;
private:
    void affine_project(float *d2i,float x,float y,float *ox,float *oy); //通过仿射变换逆矩阵，恢复成原图的坐标
    void getd2i(affineMatrix &afmt,cv::Size  to,cv::Size from);
    //cv::Rect get_rect_adapt_landmark1(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]);
    inline bool model_exists (const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
};
#endif 