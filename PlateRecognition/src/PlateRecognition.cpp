#include "PlateRecognition.h"

PlateRecognition::PlateRecognition()
{
}

PlateRecognition::~PlateRecognition()
{
}
HZFLAG PlateRecognition:: PlateRecognitionInit(Config*config)
{
	INPUT_BLOB_NAME = "images";
	OUTPUT_BLOB_NAME1 = "output_1";
	OUTPUT_BLOB_NAME2 = "output_2";
	INPUT_H = 48;
	INPUT_W = 168;
	OUTPUT_SIZE1 = 21*78;
	OUTPUT_SIZE2 = 5;
	this->bs=config->plate_recognition_bs;
	cudaSetDevice(config->gpu_id);
	// create a model using the API directly and serialize it to a stream
	char *trtModelStream{ nullptr };
	size_t size{0};
	std::string directory;
	std::string model_path=config->PlateReconitionModelPath;
	const size_t last_slash_idx = model_path.rfind(".onnx");
	if (std::string::npos != last_slash_idx)
	{
		directory = model_path.substr(0, last_slash_idx);
	}
	std::string out_engine = directory +"_batch="+ std::to_string(config->plate_recognition_bs) + ".engine";
	bool enginemodel = model_exists(out_engine);
	if (!enginemodel)
	{
		std::cout << "Building engine, please wait for a while..." << std::endl;
		bool wts_model = model_exists(model_path);//config->classs_path
		if (!wts_model)
		{
			std::cout << "ONNX file is not Exist!Please Check!" << std::endl;
			return HZ_ERROR;
		}
		Onnx2Ttr onnx2trt;
		onnx2trt.onnxToTRTModel(gLogger,model_path.c_str(),config->plate_recognition_bs,out_engine.c_str());//config->classs_path
	}
	std::ifstream file(out_engine, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}

	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	assert(engine->getNbBindings() == 3);
	inputIndex1 = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
    outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);
    assert(inputIndex1 == 0);
    assert(outputIndex1 == 2);
    assert(outputIndex2 == 1);

	CHECK(cudaMalloc(&buffers[inputIndex1], config->plate_recognition_bs * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex1], config->plate_recognition_bs * OUTPUT_SIZE1 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex2], config->plate_recognition_bs * OUTPUT_SIZE2 * sizeof(float)));
	CHECK(cudaStreamCreate(&stream));

	data = new float[3 * INPUT_H * INPUT_W];
	prob1 = new float[OUTPUT_SIZE1];
	prob2 = new float[OUTPUT_SIZE2];
	return HZ_SUCCESS;
}
HZFLAG PlateRecognition::PlateRecognitionRun(cv::Mat&img,std::string&plate_str,std::string&plate_color)
{
	
	if (img.empty())
		return HZ_IMGEMPTY;
	cv::resize(img,pr_img,cv::Size(168,48));
	int i = 0;
	for (int row = 0; row < INPUT_H; ++row)
	{
		uchar* uc_pixel = pr_img.data + row * pr_img.step;
		for (int col = 0; col < INPUT_W; ++col)
		{
			data[i] = ((float)uc_pixel[0] / 255.0 -0.588) / 0.193; // R-0.485
			data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 -0.588) / 0.193;
			data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[2] / 255.0 -0.588) / 0.193;
			uc_pixel += 3;
			++i;
		}
	}
	// Run inference  
	
	CHECK(cudaMemcpyAsync(buffers[inputIndex1], data, this->bs * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	(*context).enqueueV2(buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(prob1, buffers[outputIndex1], this->bs * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(prob2, buffers[outputIndex2], this->bs * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	//颜色
	std::vector<float>plate_color_vec(5);
	memcpy(plate_color_vec.data(),prob2,5*sizeof(float));
	int max_Index=argmax(plate_color_vec.begin(),plate_color_vec.end());
	plate_color=plate_color_list[max_Index];
	//车牌
	std::vector<int>plate_index;
	std::vector<float>plate_tensor(78);
	float* prob1_temp=prob1;
	for (size_t j = 0; j < 21; j++)
	{
		memcpy(plate_tensor.data(),prob1_temp,78*sizeof(float));
		int max_Index=argmax(plate_tensor.begin(),plate_tensor.end());
		plate_index.push_back(max_Index);
		prob1_temp=prob1_temp+78;
	}
	int pre=0;
	for (size_t j = 0; j < plate_index.size(); j++)
	{
		if(plate_index[j]!=0&&plate_index[j]!=pre)
		{
			plate_str+=plate_chr[plate_index[j]];
		}
        pre=plate_index[j];
	}
	return HZ_SUCCESS;
}
HZFLAG PlateRecognition::PlateRecognitionRelease()
{
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex1]));
	CHECK(cudaFree(buffers[outputIndex1]));
	CHECK(cudaFree(buffers[outputIndex2]));
	delete[]data;
	delete[]prob1;
	delete[]prob2;
	data = NULL;
	prob1 = NULL;
	prob2 = NULL;
	return HZ_SUCCESS;
}
bool  PlateRecognition::model_exists(const std::string& name)
{
	std::ifstream f(name.c_str());
	return f.good();
}