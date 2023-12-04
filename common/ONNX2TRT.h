#pragma once
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "logging.h"


using namespace nvinfer1;
class Onnx2Ttr
{
private:
    //Logger gLogger;
    int gUseDLACore;
public:
    Onnx2Ttr(/* args */);
    ~Onnx2Ttr();
    void enableDLA(IBuilderConfig* b, int useDLACore);
    int get_stream_from_file(const char* filename, unsigned char* buf, size_t* size);
    void onnxToTRTModel(const char* modelFile,         // name of the onnx model
                    unsigned int maxBatchSize,     // batch size - NB must be at least as large as the batch we want to run with
                    const char* out_trtfile,Logger*gLogger);

};


