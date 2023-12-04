#include "platedetector_yolov7.h"
using namespace std;

Detector_Yolov7Plate::Detector_Yolov7Plate()
{
}
Detector_Yolov7Plate::~Detector_Yolov7Plate()
{
}
HZFLAG Detector_Yolov7Plate::InitDetector_Yolov7Plate(Config*config)
{

  this->conf_thresh=config->yolov7plate_confidence_thresh;
  this->nms_thresh=config->yolov7plate_nms_thresh;
  this->batch_size=config->yolov7plate_detect_bs;
  // H, W must be able to  be divided by 32.
  this->INPUT_W = 640;
  this->INPUT_H = 640;
  this->NUM_CLASSES=2;        //单层车牌和双层车牌
  this->CKPT_NUM=4;           //4个关键点
  this->NUM_BOX_ELEMENT=7+CKPT_NUM*2;
  this->OUTPUT_SIZE=(INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 3 * 19;
  this->INPUT_BLOB_NAME = "images";
  this->OUTPUT_BLOB_NAME = "output";
  cudaSetDevice(config->gpu_id);
  std::string directory; 
  std::string model_path=config->Yolov7PlateDetectModelPath;
  const size_t last_slash_idx=model_path.rfind(".onnx");
  if (std::string::npos != last_slash_idx)
  {
      directory = model_path.substr(0, last_slash_idx);
  }
  std::string out_engine=directory+"_batch="+std::to_string(config->yolov7plate_detect_bs)+".engine";
  bool enginemodel=model_exists(out_engine);
  if (!enginemodel)
  {
    std::cout << "Building engine, please wait for a while..." << std::endl;
    bool wts_model=model_exists(model_path);
    if (!wts_model)
    {
      std::cout<<"yolov7s-Plate.onnx is not Exist!!!Please Check!"<<std::endl;
      return HZ_WITHOUTMODEL;
    }
    Onnx2Ttr onnx2trt;
    //IHostMemory* modelStream{ nullptr };
    onnx2trt.onnxToTRTModel(model_path.c_str(),config->yolov5plate_detect_bs,out_engine.c_str(),&gLogger);
  }
  size_t size{0};
  std::ifstream file(out_engine, std::ios::binary);//out_engine"/home/pcb/PlateRecognition_Linux_Release/yolov7Plate_test/yolov7-Plate-tensorrt/yolov7s-Plate_batch=1.engine"
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
  else
  {
    std::cout<<"yolov7s-Plate.engine model file not exist!"<<std::endl;
    return HZ_WITHOUTMODEL;
  }

  this->runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  this->context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  assert(engine->getNbBindings() == 2);
  this->inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  this->outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  assert(inputIndex == 0);
  assert(outputIndex == 1);

  auto out_dims = engine->getBindingDimensions(1);
  OUTPUT_CANDIDATES = out_dims.d[1];

  // Create GPU buffers on device
  CHECK(cudaMalloc(&this->buffers[inputIndex], config->yolov7plate_detect_bs * 3 * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&this->buffers[outputIndex], config->yolov7plate_detect_bs * OUTPUT_SIZE * sizeof(float)));
  // Create stream
  CHECK(cudaStreamCreate(&stream));
    // prepare input data cache in pinned memory 
  CHECK(cudaMallocHost((void**)&img_host, config->yolov7plate_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
  // prepare input data cache in device memory
  CHECK(cudaMalloc((void**)&img_device, config->yolov7plate_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
  
  //postprocess input data cache in device memory
  CHECK(cudaMalloc(&decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT)));
  
  CHECK(cudaMalloc((void**)&pre_predict, OUTPUT_SIZE * sizeof(float)));

  CHECK(cudaMallocHost(&affine_matrix_d2i_host,sizeof(float)*6));

  this->affine_matrix_d2i_device=new float*[batch_size];
  this->decode_ptr_host=new float*[batch_size];
  for (size_t i = 0; i < batch_size; i++)
  {
    this->decode_ptr_host[i]= new float[(1+MAX_OBJECTS*NUM_BOX_ELEMENT)];
    CHECK(cudaMalloc(&this->affine_matrix_d2i_device[i],sizeof(float)*6));
  }
  return HZ_SUCCESS;
}

HZFLAG Detector_Yolov7Plate::Detect_Yolov7Plate(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>& dets)
{
  // prepare input data ---------------------------
  int detector_batchsize=ImgVec.size();
  float* buffer_idx = (float*)this->buffers[inputIndex];
  for (int b = 0; b < detector_batchsize; b++)
  {
    if (ImgVec[b].empty()||ImgVec[b].data==NULL) 
    {
      continue;
    }
    //proprecess
    affineMatrix afmt;
    getd2i(afmt,cv::Size(INPUT_W,INPUT_H),cv::Size(ImgVec[b].cols,ImgVec[b].rows));
    size_t size_image = ImgVec[b].cols * ImgVec[b].rows * 3*sizeof(uint8_t);
    size_t size_image_dst = INPUT_H * INPUT_W * 3*sizeof(uint8_t);
    memcpy(affine_matrix_d2i_host,afmt.d2i,sizeof(afmt.d2i));
    memcpy(img_host, ImgVec[b].data, size_image);
    CHECK(cudaMemcpy(img_device, img_host, size_image, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(affine_matrix_d2i_device[b],affine_matrix_d2i_host,sizeof(afmt.d2i),cudaMemcpyHostToDevice));
    yolov7plate_preprocess_kernel_img(img_device, ImgVec[b].cols, ImgVec[b].rows, buffer_idx, INPUT_W, INPUT_H,affine_matrix_d2i_device[b], stream);
    buffer_idx += size_image_dst;
  }
  //inference
  (*context).enqueueV2((void**)this->buffers, stream, nullptr);

  //postprocess
  float *predict = (float *)this->buffers[outputIndex];
  for (size_t i = 0; i < detector_batchsize; i++)
  {
    CHECK(cudaMemsetAsync(decode_ptr_device,0,sizeof(int),stream));
    CHECK(cudaMemcpyAsync(pre_predict,predict,OUTPUT_SIZE * sizeof(float),cudaMemcpyDeviceToDevice, stream));
    yolov7plate_decode_kernel_invoker(pre_predict,NUM_BOX_ELEMENT,OUTPUT_CANDIDATES,NUM_CLASSES,CKPT_NUM,
                          this->conf_thresh,affine_matrix_d2i_device[i],decode_ptr_device,MAX_OBJECTS,stream);  //cuda decode
    yolov7plate_nms_kernel_invoker(decode_ptr_device,this->nms_thresh, MAX_OBJECTS, stream,NUM_BOX_ELEMENT);                //cuda nms
    CHECK(cudaMemcpyAsync(decode_ptr_host[i],decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT),cudaMemcpyDeviceToHost,stream));
    predict+=OUTPUT_SIZE;
  }
  cudaStreamSynchronize(stream);
  for (size_t k = 0; k < detector_batchsize; k++)
  {
    std::vector<Det>det;
    int count = std::min((int)*decode_ptr_host[k],MAX_OBJECTS);
    for (int i = 0; i<count;i++)
    {
      int basic_pos = 1+i*NUM_BOX_ELEMENT;
      int keep_flag= decode_ptr_host[k][basic_pos+6];
      if (keep_flag==1)
      {
        Det det_temp;
        det_temp.bbox.xmin =  decode_ptr_host[k][basic_pos+0];    
        det_temp.bbox.ymin =  decode_ptr_host[k][basic_pos+1];    
        det_temp.bbox.xmax =  decode_ptr_host[k][basic_pos+2];    
        det_temp.bbox.ymax =  decode_ptr_host[k][basic_pos+3];    
        det_temp.confidence= decode_ptr_host[k][basic_pos+4];
        det_temp.label= decode_ptr_host[k][basic_pos+5];   
        int landmark_pos = basic_pos+7;
        for (int id = 0; id<CKPT_NUM; id+=1)
        {
          det_temp.key_points.push_back(decode_ptr_host[k][landmark_pos+2*id]);
          det_temp.key_points.push_back(decode_ptr_host[k][landmark_pos+2*id+1]);
        }
        det.push_back(det_temp);
      }
    }
    dets.push_back(det);
  }
  return HZ_SUCCESS;
}
HZFLAG Detector_Yolov7Plate::ReleaseDetector_Yolov7Plate()
{
  context->destroy();
  engine->destroy();
  runtime->destroy();
  for (size_t i = 0; i < batch_size; i++)
  {
    CHECK(cudaFree(affine_matrix_d2i_device[i]));
    delete decode_ptr_host[i];
  }
  delete [] decode_ptr_host;
  delete [] affine_matrix_d2i_device;
  CHECK(cudaFreeHost(affine_matrix_d2i_host));
  CHECK(cudaFree(img_device));
  CHECK(cudaFreeHost(img_host));
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
  CHECK(cudaFree(decode_ptr_device));
  CHECK(cudaFree(pre_predict));
  return HZ_SUCCESS;
}

void Detector_Yolov7Plate::affine_project(float *d2i,float x,float y,float *ox,float *oy) //通过仿射变换逆矩阵，恢复成原图的坐标
{
  *ox = d2i[0]*x+d2i[1]*y+d2i[2];
  *oy = d2i[3]*x+d2i[4]*y+d2i[5];
}

void Detector_Yolov7Plate::getd2i(affineMatrix &afmt,cv::Size  to,cv::Size from) //计算仿射变换的矩阵和逆矩阵
{
  float scale = std::min(1.0*to.width/from.width, 1.0*to.height/from.height);
  afmt.i2d[0]=scale;
  afmt.i2d[1]=0;
  afmt.i2d[2]=-scale*from.width*0.5+to.width*0.5;
  afmt.i2d[3]=0;
  afmt.i2d[4]=scale;
  afmt.i2d[5]=-scale*from.height*0.5+to.height*0.5;
  cv::Mat i2d_mat(2,3,CV_32F,afmt.i2d);
  cv::Mat d2i_mat(2,3,CV_32F,afmt.d2i);
  cv::invertAffineTransform(i2d_mat,d2i_mat);
  memcpy(afmt.d2i, d2i_mat.ptr<float>(0), sizeof(afmt.d2i));
}