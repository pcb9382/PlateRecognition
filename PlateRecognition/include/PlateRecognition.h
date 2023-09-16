#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include "dirent.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "ONNX2TRT.h"
#include "DataTypes_Plate.h"
using namespace nvinfer1;
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


class PlateRecognition
{
public:
	PlateRecognition();
	~PlateRecognition();
	HZFLAG PlateRecognitionInit(Config*config);
	HZFLAG PlateRecognitionRun(cv::Mat&img,std::string&plate_str,std::string&plate_color);
	HZFLAG PlateRecognitionRelease();
private:
	char* INPUT_BLOB_NAME;
	char* OUTPUT_BLOB_NAME1;
	char* OUTPUT_BLOB_NAME2;
	int INPUT_H;
	int INPUT_W;
	int OUTPUT_SIZE1;
	int OUTPUT_SIZE2;
	float *data ;
	float *prob1;
	float *prob2;
	int inputIndex1;
    int outputIndex1;
    int outputIndex2;
	Logger gLogger;
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
	void* buffers[3];
	// Create stream
	cudaStream_t stream;
	int bs;
	std::vector<std::string>plate_color_list={"黑色","蓝色","绿色","白色","黄色"};
	std::string plate_chr[78]={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁",
	"新","学","警","港","澳","挂","使","领","民","航","危","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品"};
	cv::Mat pr_img;
private:
	cv::Mat CenterCrop(cv::Mat img);
	std::vector<float> softmax(float *prob);
	void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	bool model_exists(const std::string& name);
	template<class ForwardIterator>
	inline size_t argmin(ForwardIterator first, ForwardIterator last)
	{
		return std::distance(first, std::min_element(first, last));
	}
	
	template<class ForwardIterator>
	inline size_t argmax(ForwardIterator first, ForwardIterator last)
	{
		return std::distance(first, std::max_element(first, last));
	}

};


