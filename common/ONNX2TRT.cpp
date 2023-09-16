#include "ONNX2TRT.h"


Onnx2Ttr::Onnx2Ttr(/* args */)
{
    gUseDLACore=-1;
}

Onnx2Ttr::~Onnx2Ttr()
{
}
void Onnx2Ttr::enableDLA(IBuilderConfig* b, int useDLACore) 
{
  if (useDLACore >= 0) 
  {
    // b->allowGPUFallback(true);
    b->setFlag(BuilderFlag::kFP16);
    b->setDefaultDeviceType(DeviceType::kDLA);
    b->setDLACore(useDLACore);
  }
}

int Onnx2Ttr::get_stream_from_file(const char* filename, unsigned char* buf, size_t* size) 
{
  FILE* fp = fopen(filename, "rb");
  if(fp == NULL) 
  {
    printf("Can not open trt file\n");
    fclose(fp);
    return -1;
  } 
  else 
  {
    fseek(fp, 0L, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    int ret = fread(buf, 1, *size, fp);
    fclose(fp);
    return ret;
  }
}

void Onnx2Ttr::onnxToTRTModel(Logger gLogger,const char* modelFile,         // name of the onnx model
    unsigned int maxBatchSize,                               // batch size - NB must be at least as large as the batch we want to run with
    const char* out_trtfile) 
{
  int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

  // init plugin
  bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

  int cnt = 0;
  auto creator_list = getPluginRegistry()-> getPluginCreatorList(&cnt);
  std::cout << "creator_list : " << cnt << std::endl;
  for(int i = 0 ; i < cnt; i++)
  {
    std::cout << creator_list[i]->getPluginName() << " " << creator_list[i]->getPluginVersion() << std::endl;
  }
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);

  auto parser = nvonnxparser::createParser(*network, gLogger);

  if (!parser->parseFromFile(modelFile, verbosity))
  {
    std::string msg("failed to parse onnx file");
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }

  std::cout << "Net Name : " << network->getName() << std::endl;
  std::cout << "Net layer number : " << network->getNbLayers() << std::endl;
  std::cout << "Net input/output : " << network->getNbInputs() << " " << network->getNbOutputs() << std::endl;
  for(int i = 0; i < network->getNbInputs(); i++) 
  {
    auto tensor = network->getInput(i);
    std::cout << "Input Name : " << tensor->getName() << std::endl;
    for(int n = 0; n < tensor->getDimensions().nbDims; n++)
      std::cout << tensor->getDimensions().d[n] << " ";
    std::cout << std::endl;
  }
  for(int i = 0; i < network->getNbLayers(); i++) 
  {
    auto layer = network->getLayer(i);
    if(layer == nullptr) std::cout << "layer == nullptr " << i << std::endl;
    std::cout << "layer: " << i << "    Name : " << layer->getName() << "   Type:" << int(layer->getType()) << std::endl;
    for(int n = 0; n < layer->getOutput(0)->getDimensions().nbDims; n++)
      std::cout << layer->getOutput(0)->getDimensions().d[n] << " ";
    std::cout << std::endl;
  }

  for(int i = 0; i < network->getNbOutputs(); i++) 
  {
    auto tensor = network->getOutput(i);
    std::cout << "Output Name : " << tensor->getName() << std::endl;
    for(int n = 0; n < tensor->getDimensions().nbDims; n++)
      std::cout << tensor->getDimensions().d[n] << " ";
    std::cout << std::endl;
  }

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(1 << 30);
  config->setFlag(BuilderFlag::kFP16);

  std::cout << "Total DLA Core : " << builder->getNbDLACores() << std::endl;

  enableDLA(config, gUseDLACore);
  auto* engine = builder->buildSerializedNetwork(*network, *config);
  assert(engine);

  // we can destroy the parser
  parser->destroy();

  // serialize the engine, then close everything down
  assert(engine != nullptr && "engine == nullptr");

  std::ofstream p(out_trtfile, std::ios::binary);
  if (!p) 
  {
    std::cerr << "could not open plan output file" << std::endl;
    return ;
  }
  p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
  return;
}