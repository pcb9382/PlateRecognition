#pragma once
#ifndef __DETECTOR_YOLOV5PLATE_H__
#define __DETECTOR_YOLOV5PLATE_H__
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
#include "yolov5plate_preprocess.h"
#include "ONNX2TRT.h"
#include "DataTypes_Plate.h"
#define USE_FP16
#define MAX_IMAGE_INPUT_SIZE_THRESH 8000 * 6000
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

namespace decodeplugin_yolov5plate
{
  struct  Detection
  {
    float bbox[4];  //x1 y1 x2 y2
    float class_confidence;
    int label;
    float landmark[8];
  };
}


class Detector_Yolov5plate
{ 
public:
    Detector_Yolov5plate();
    ~Detector_Yolov5plate();
    HZFLAG InitDetector_Yolov5plate(Config*config);
    HZFLAG Detect_Yolov5plate(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>&dets);
    HZFLAG ReleaseDetector_Yolov5plate();

public:
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
    int gpu_id;  // GPU id
    float conf_thresh;
    float nms_thresh;
    int INPUT_H;  // H, W must be able to  be divided by 32.
    int INPUT_W;
    int OUTPUT_SIZE;
    int OUTPUT_CANDIDATES;
    int OUTPUT_TENSOR;
    char* INPUT_BLOB_NAME;
    char* OUTPUT_BLOB_NAME;
    char *trtModelStream{nullptr};
    float *prob=nullptr;


public:
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize);
    cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
    float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<decodeplugin_yolov5plate::Detection>& res, float *output,float confidence=0.1,float nms_thresh = 0.4);
    void get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det);
    //cv::Rect get_rect_adapt_landmark1(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]);
    inline bool model_exists (const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
};
#endif 