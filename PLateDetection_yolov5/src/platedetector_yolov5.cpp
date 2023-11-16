#include "platedetector_yolov5.h"
using namespace std;

Detector_Yolov5plate::Detector_Yolov5plate()
{

}
Detector_Yolov5plate::~Detector_Yolov5plate()
{


}

HZFLAG Detector_Yolov5plate::InitDetector_Yolov5plate(Config*config)
{

  this->conf_thresh=config->yolov5plate_confidence_thresh;
  this->nms_thresh=config->yolov5plate_nms_thresh;
  // H, W must be able to  be divided by 32.
  // INPUT_W = 640;
  // INPUT_H = 640;  
  // OUTPUT_SIZE=(INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 3 * 15;
  INPUT_BLOB_NAME = "input";
  OUTPUT_BLOB_NAME = "output";
  std::string model_path=config->Yolov5PlateDetectModelPath;
  std::string directory; 
  const size_t last_slash_idx=model_path.rfind(".onnx");
  if (std::string::npos != last_slash_idx)
  {
    directory = model_path.substr(0, last_slash_idx);
  }
  std::string out_engine=directory+"_batch="+std::to_string(config->yolov5plate_detect_bs)+".engine";
  bool enginemodel=model_exists(out_engine);
  if (!enginemodel)
  {
    std::cout << "Building engine, please wait for a while..." << std::endl;
    bool wts_model=model_exists(model_path);
    if (!wts_model)
    {
        std::cout<<"yolov5s-plate.onnx is not Exist!!!Please Check!"<<std::endl;
        return HZ_WITHOUTMODEL;
    }
    Onnx2Ttr onnx2trt;
    onnx2trt.onnxToTRTModel(gLogger,model_path.c_str(),config->yolov5plate_detect_bs,out_engine.c_str());
  }
  size_t size{0};
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
  else
  {
    std::cout<<"yolov5s-plate.engine model file not exist!"<<std::endl;
    return HZ_WITHOUTMODEL;
  }
  
  runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  assert(engine->getNbBindings() == 2);
  inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  assert(inputIndex == 0);
  assert(outputIndex == 1);

    //input nchw
  auto input_dims = engine->getBindingDimensions(0);
  this->INPUT_W = input_dims.d[3];
  this->INPUT_H = input_dims.d[2];

  //1*20*8400
  auto output_dims = engine->getBindingDimensions(1);
  this->OUTPUT_TENSOR=output_dims.d[1];                         //25200
  this->OUTPUT_CANDIDATES = output_dims.d[2];                   //15
  this->OUTPUT_SIZE=this->OUTPUT_TENSOR*this->OUTPUT_CANDIDATES;//2500*15


  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], config->yolov5plate_detect_bs * 3 * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], config->yolov5plate_detect_bs * OUTPUT_SIZE * sizeof(float)));
  CHECK(cudaStreamCreate(&stream));
    // prepare input data cache in pinned memory 
  CHECK(cudaMallocHost((void**)&img_host, config->yolov5plate_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
  // prepare input data cache in device memory
  CHECK(cudaMalloc((void**)&img_device, config->yolov5plate_detect_bs*MAX_IMAGE_INPUT_SIZE_THRESH * 3*sizeof(uint8_t)));
  prob=new float[config->yolov5plate_detect_bs * OUTPUT_SIZE];
  return HZ_SUCCESS;
}

HZFLAG Detector_Yolov5plate::Detect_Yolov5plate(std::vector<cv::Mat>&ImgVec,std::vector<std::vector<Det>>& dets)
{
    
    // prepare input data ---------------------------
    int detector_batchsize=ImgVec.size();
    float* buffer_idx = (float*)buffers[inputIndex];
    for (int b = 0; b < detector_batchsize; b++)
    {
      if (ImgVec[b].empty()||ImgVec[b].data==NULL) 
      {
        continue;
      }
      ImgVec[b] = ImgVec[b].clone();
      size_t  size_image = ImgVec[b].cols * ImgVec[b].rows * 3*sizeof(uint8_t);
      size_t  size_image_dst = INPUT_H * INPUT_W * 3*sizeof(uint8_t);
      //copy data to pinned memory
      memcpy(img_host,ImgVec[b].data,size_image);
      //copy data to device memory
      CHECK(cudaMemcpy(img_device,img_host,size_image,cudaMemcpyHostToDevice));
      preprocess_kernel_img_yolov5_plate(img_device,ImgVec[b].cols,ImgVec[b].rows, buffer_idx, INPUT_W, INPUT_H, stream);       
      buffer_idx += size_image_dst;
    }
    // Run inference
    (*context).enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(prob, buffers[1], detector_batchsize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    for (int b = 0; b < detector_batchsize; b++) 
    {
        std::vector<decodeplugin_yolov5plate::Detection> res;
        nms(res, &prob[b * OUTPUT_SIZE],this->conf_thresh,this->nms_thresh);
        std::vector<Det>Imgdet;
        for (size_t j = 0; j < res.size(); j++) 
        {
            if (res[j].class_confidence < conf_thresh) 
            {
                continue;
            }
            Det det;
            det.confidence=res[j].class_confidence;
            det.label=res[j].label;
            get_rect_adapt_landmark(ImgVec[b], INPUT_W, INPUT_H, res[j].bbox, res[j].landmark,det);
            Imgdet.push_back(det);
        }
        dets.push_back(Imgdet);
    }
   
    return HZ_SUCCESS;

}
HZFLAG Detector_Yolov5plate::ReleaseDetector_Yolov5plate()
{
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete []prob;
    prob=NULL;
    return HZ_SUCCESS;
}

void Detector_Yolov5plate::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) 
{
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
cv::Mat Detector_Yolov5plate::preprocess_img(cv::Mat& img, int input_w, int input_h) 
{
  int w, h, x, y;
  float r_w = input_w / (img.cols*1.0);
  float r_h = input_h / (img.rows*1.0);
  if (r_h > r_w) 
  {
    w = input_w;
    h = r_w * img.rows;
    x = 0;
    y = (input_h - h) / 2;
  }
  else 
  {
    w = r_h * img.cols;
    h = input_h;
    x = (input_w - w) / 2;
    y = 0;
  }
  cv::Mat re(h, w, CV_8UC3);
  cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
  cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
  return out;
}

bool cmp(const decodeplugin_yolov5plate::Detection& a, const decodeplugin_yolov5plate::Detection& b) 
{
  return a.class_confidence > b.class_confidence;
}

float Detector_Yolov5plate::iou(float lbox[4], float rbox[4]) 
{
  float interBox[] = 
  {
    std::max(lbox[0]-lbox[2]/2, rbox[0]-rbox[2]/2), //left
    std::min(lbox[0]+lbox[2]/2, rbox[0]+rbox[2]/2), //right
    std::max(lbox[1]-lbox[3]/2, rbox[1]-rbox[3]/2), //top
    std::min(lbox[1]+lbox[3]/2, rbox[1]+rbox[3]/2), //bottom
  };
  if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
  {
    return 0.0f;
  }
  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  //return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
  return interBoxS / ((lbox[2]) * (lbox[3]) + (rbox[2]) * (rbox[3]) -interBoxS + 0.000001f);
}
void Detector_Yolov5plate::nms(std::vector<decodeplugin_yolov5plate::Detection>& res, float *output,float confidence,float nms_thresh) 
{
  std::vector<decodeplugin_yolov5plate::Detection>  Imgdet;
  for (int i = 0; i < this->OUTPUT_TENSOR; i++) 
  {
    
    int class_num=this->OUTPUT_CANDIDATES-13;
    float max_conf=0.0;
    int class_idx=-1;
    for (size_t j = 0; j < class_num; j++)
    {
      // conf = obj_conf * cls_conf
      float conf_temp=output[this->OUTPUT_CANDIDATES* i +4]*output[this->OUTPUT_CANDIDATES*i+j+13];
      if (max_conf<conf_temp)
      {
        max_conf=conf_temp;
        class_idx=j;
      }
    }
    if (max_conf<confidence)
    {
      continue;
    }
    decodeplugin_yolov5plate::Detection det;
    for (size_t j = 0; j < 4; j++)
    {
      det.bbox[j]=output[this->OUTPUT_CANDIDATES* i+j];
    }
    det.class_confidence=max_conf;//output[this->OUTPUT_TENSOR* i+4];
    for (size_t j = 0; j < 8; j++)
    {
      det.landmark[j]=output[this->OUTPUT_CANDIDATES* i+j+5];
    }
    det.label=class_idx;
    Imgdet.push_back(det);
  }
  std::sort( Imgdet.begin(),  Imgdet.end(), cmp);
  for (size_t m = 0; m <  Imgdet.size(); ++m) 
  {
    auto& item =  Imgdet[m];
    res.push_back(item);
    //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
    for (size_t n = m + 1; n <  Imgdet.size(); ++n) 
    {
      if (iou(item.bbox,  Imgdet[n].bbox) > nms_thresh) 
      {
        Imgdet.erase( Imgdet.begin()+n);
        --n;
      }
    }
  }
}
void Detector_Yolov5plate::get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10],Det&det) 
{
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) 
    {
        l = (bbox[0]-bbox[2]/2)/r_w;
        r = (bbox[0]+bbox[2]/2)/r_w;
        t = (bbox[1]-bbox[3]/2 - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[1]+bbox[3]/2 - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) 
        {
            det.key_points.push_back(lmk[i]/r_w);
            det.key_points.push_back((lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w);
        }
    } 
    else 
    {
        l = (bbox[0]-bbox[2]/2 - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[0]+bbox[2]/2 - (input_w - r_h * img.cols) / 2) / r_h;
        t = (bbox[1]-bbox[3]/2) / r_h;
        b = (bbox[1]+bbox[3]/2) / r_h;
        for (int i = 0; i < 10; i += 2) 
        {
            det.key_points.push_back((lmk[i] - (input_w - r_h * img.cols) / 2) / r_h);
            det.key_points.push_back(lmk[i + 1]/r_h);
        }
    }
    det.bbox.xmin=l>1?l:1;
    det.bbox.ymin=t>1?t:1;
    det.bbox.xmax=r>det.bbox.xmin?r:det.bbox.xmin+1;
    det.bbox.xmax=det.bbox.xmax<img.cols?det.bbox.xmax:img.cols-1;
    det.bbox.ymax=b>det.bbox.ymin?b:det.bbox.ymin+1;
    det.bbox.ymax=det.bbox.ymax<img.rows?det.bbox.ymax:img.rows-1;
    return;
}