#include<algorithm>
#include<cstring>
#include<exception>
#include<iostream>
#include<memory>
#include"PlateDetectionRecognition.h"
#include"platedetector_yolov7.h"
#include"platedetector_yolov5.h"
#include"PlateRecognitionOCRV5.h"
#include"PlateRecognitionOCRV6.h"
#include"PlateRecognitionLPRNet.h"


class PlateAlgorithm
{
public:
    ~PlateAlgorithm();
    /** 
     * @brief                   车牌初始化函数
     * @param config			模块配置参数结构体
     * @return                  HZFLAG
     */
    int Initialize(Config*config);
    /** 
     * @brief                   车牌检测识别(yolov5)
     * @param img			    Plate_ImageData
     * @param PlateDet		    车牌检测识别结果列表
     * @return                  HZFLAG
     */		
    int PlateRecognition_yolov5(Plate_ImageData*img,PlateDet*PlateDets);
    /** 
     * @brief                   车牌检测(yolov7_plate)
     * @param img			    Plate_ImageData
     * @param PlateDet		    车牌检测识别结果列表
     * @return                  HZFLAG
     */		
    int PlateRecognition_yolov7(Plate_ImageData*img,PlateDet*PlateDets);

    /**
     * @brief                   车牌检测识别(yolov5 + PP-OCRv5)
     */
    int PlateRecognition_yolov5_PPOCRV5(Plate_ImageData*img,PlateDet*PlateDets);

    /**
     * @brief                   车牌检测识别(yolov7 + PP-OCRv5)
     */
    int PlateRecognition_yolov7_PPOCRV5(Plate_ImageData*img,PlateDet*PlateDets);

    /**
     * @brief                   车牌检测识别(yolov5 + LPRNet)
     * @param img              Plate_ImageData
     * @param PlateDet         车牌检测识别结果列表
     * @return                 HZFLAG
     */
    int PlateRecognition_yolov5_LPRNet(Plate_ImageData*img,PlateDet*PlateDets);

    /**
     * @brief                   车牌检测识别(yolov7 + LPRNet)
     * @param img              Plate_ImageData
     * @param PlateDet         车牌检测识别结果列表
     * @return                 HZFLAG
     */
    int PlateRecognition_yolov7_LPRNet(Plate_ImageData*img,PlateDet*PlateDets);


    /** 
     * @brief                   车牌检测(yolov8_plate)
     * @param img			    Plate_ImageData
     * @param PlateDet		    车牌检测识别结果列表
     * @return                  HZFLAG
     */		
    int PlateRecognition_yolov8(Plate_ImageData*img,PlateDet*PlateDets);

    /** 
     * @brief               反初始化
     * @return              HZFLAG 
     */		
    int Release(Config*config);
private:
    Detector_Yolov5plate yolov5_plate;
    Detector_Yolov7Plate yolov7_plate;
    PlateRecognitionOCRV5 plate_recognition_ppocrv5;
    PlateRecognitionOCRV6 plate_recognition_ppocrv6;
    PlateRecognitionLPRNet plate_recognition_lprnet;
    bool yolov5_initialized=false;
    bool yolov7_initialized=false;
    bool ppocrv5_initialized=false;
    bool ppocrv6_initialized=false;
    bool lprnet_initialized=false;
    std::size_t plate_result_capacity=0;
    std::size_t plate_license_capacity=0;
    std::size_t plate_color_capacity=0;
private:
    enum class RecognitionBackend
    {
        PPOCRV5,
        PPOCRV6,
        LPRNet
    };
    void releaseInitializedComponents();
    int fillRecognitionResults(cv::Mat &image,
                               const std::vector<std::vector<Det>> &dets,
                               PlateDet *PlateDets,
                               RecognitionBackend backend);
    cv::Mat get_split_merge(cv::Mat &img);   //双层车牌 分割 拼接
    cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]); //透视变换
    float getNorm2(float x,float y);
};

namespace {

template <typename Function>
int GuardApiCall(Function function) noexcept
{
    try
    {
        return function();
    }
    catch (const std::exception &exception)
    {
        std::cerr<<"plate algorithm exception: "<<exception.what()<<std::endl;
    }
    catch (...)
    {
        std::cerr<<"unknown plate algorithm exception"<<std::endl;
    }
    return -1;
}

} // namespace

PlateAlgorithm::~PlateAlgorithm()
{
    releaseInitializedComponents();
}


/** 
 * @brief                   车牌初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
int PlateAlgorithm::Initialize(Config*config)
{
    if (config==NULL)
    {
        std::cerr<<"config is null"<<std::endl;
        return -1;
    }
    if (config->gpu_id<0)
    {
        std::cerr<<"gpu_id must be non-negative"<<std::endl;
        return -1;
    }
    if (config->plate_recognition_enable &&
        (config->plate_result_capacity==0 ||
         config->plate_license_capacity==0 ||
         config->plate_color_capacity==0))
    {
        std::cerr<<"plate result and string capacities must be greater than zero"
                 <<std::endl;
        return -1;
    }
    plate_result_capacity=config->plate_result_capacity;
    plate_license_capacity=config->plate_license_capacity;
    plate_color_capacity=config->plate_color_capacity;

    if (config->yolov5plate_detect_enable)
    {
        if (config->Yolov5PlateDetectModelPath==NULL ||
            config->Yolov5PlateDetectModelPath[0]=='\0' ||
            config->yolov5plate_detect_bs<=0 ||
            config->yolov5plate_confidence_thresh<0.0F ||
            config->yolov5plate_confidence_thresh>1.0F ||
            config->yolov5plate_nms_thresh<0.0F ||
            config->yolov5plate_nms_thresh>1.0F)
        {
            std::cerr<<"invalid YOLOv5 plate detector configuration"<<std::endl;
            return -1;
        }
        HZFLAG status=yolov5_plate.InitDetector_Yolov5plate(config);
        if (status!=HZ_SUCCESS)
        {
            releaseInitializedComponents();
            return -1;
        }
        yolov5_initialized=true;
    }
    if(config->yolov7plate_detect_enable)
    {
        if (config->Yolov7PlateDetectModelPath==NULL ||
            config->Yolov7PlateDetectModelPath[0]=='\0' ||
            config->yolov7plate_detect_bs<=0 ||
            config->yolov7plate_confidence_thresh<0.0F ||
            config->yolov7plate_confidence_thresh>1.0F ||
            config->yolov7plate_nms_thresh<0.0F ||
            config->yolov7plate_nms_thresh>1.0F)
        {
            std::cerr<<"invalid YOLOv7 plate detector configuration"<<std::endl;
            releaseInitializedComponents();
            return -1;
        }
        HZFLAG status=yolov7_plate.InitDetector_Yolov7Plate(config);
        if (status!=HZ_SUCCESS)
        {
            releaseInitializedComponents();
            return -1;
        }
        yolov7_initialized=true;
    }
    if (config->plate_recognition_enable)
    {
        bool model_configured=false;
        if (config->PlateRecognitionPPOCRV5ModelPath!=NULL &&
            config->PlateRecognitionPPOCRV5ModelPath[0]!='\0')
        {
            PlateRecognitionOCRV5Config ppocrv5_config;
            ppocrv5_config.onnx_model_path=config->PlateRecognitionPPOCRV5ModelPath;
            switch (config->plate_recognition_ppocrv5_variant)
            {
            case PlateRecognitionPPOCRV5Variant::kMobile:
                ppocrv5_config.model_variant=
                    PlateRecognitionOCRV5ModelVariant::kMobile;
                break;
            case PlateRecognitionPPOCRV5Variant::kServer:
                ppocrv5_config.model_variant=
                    PlateRecognitionOCRV5ModelVariant::kServer;
                break;
            default:
                std::cerr<<"invalid PP-OCRv5 model variant"<<std::endl;
                releaseInitializedComponents();
                return -1;
            }
            ppocrv5_config.gpu_id=config->gpu_id;
            HZFLAG status=plate_recognition_ppocrv5.PlateRecognitionOCRV5Init(&ppocrv5_config);
            if (status!=HZ_SUCCESS)
            {
                std::cerr<<"PP-OCRv5 init failed: "
                         <<plate_recognition_ppocrv5.LastError()<<std::endl;
                releaseInitializedComponents();
                return -1;
            }
            ppocrv5_initialized=true;
            model_configured=true;
        }

        if (config->PlateRecognitionPPOCRV6ModelPath!=NULL &&
            config->PlateRecognitionPPOCRV6ModelPath[0]!='\0')
        {
            PlateRecognitionOCRV6Config ppocrv6_config;
            ppocrv6_config.onnx_model_path=config->PlateRecognitionPPOCRV6ModelPath;
            ppocrv6_config.gpu_id=config->gpu_id;
            HZFLAG status=plate_recognition_ppocrv6.PlateRecognitionOCRV6Init(&ppocrv6_config);
            if (status!=HZ_SUCCESS)
            {
                std::cerr<<"PP-OCRv6 init failed: "
                         <<plate_recognition_ppocrv6.LastError()<<std::endl;
                releaseInitializedComponents();
                return -1;
            }
            ppocrv6_initialized=true;
            model_configured=true;
        }

        if (config->PlateRecognitionLPRNetModelPath!=NULL &&
            config->PlateRecognitionLPRNetModelPath[0]!='\0')
        {
            PlateRecognitionLPRNetConfig lprnet_config;
            lprnet_config.model_path=config->PlateRecognitionLPRNetModelPath;
            lprnet_config.gpu_id=config->gpu_id;
            HZFLAG status=plate_recognition_lprnet.PlateRecognitionLPRNetInit(lprnet_config);
            if (status!=HZ_SUCCESS)
            {
                std::cerr<<"LPRNet init failed: "
                         <<plate_recognition_lprnet.LastError()<<std::endl;
                releaseInitializedComponents();
                return -1;
            }
            lprnet_initialized=true;
            model_configured=true;
        }

        if (!model_configured)
        {
            std::cerr<<"plate recognition is enabled but no recognition model is configured"
                     <<std::endl;
            releaseInitializedComponents();
            return -1;
        }
    }
    std::cout<<"plate algorithm init finash"<<std::endl;
    return 0;
}
/** 
 * @brief                   车牌检测识别(yolov5)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateAlgorithm::PlateRecognition_yolov5(Plate_ImageData*img,PlateDet*PlateDets)
{
    if (!yolov5_initialized)
    {
        std::cerr<<"YOLOv5 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!ppocrv6_initialized)
    {
        std::cerr<<"PP-OCRv6 is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image); //转为图像数据
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat>ImgVec;
    std::vector<std::vector<Det>>dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov5_plate.Detect_Yolov5plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::PPOCRV6);
}
/** 
 * @brief                   车牌检测(yolov7_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateAlgorithm::PlateRecognition_yolov7(Plate_ImageData*img,PlateDet*PlateDets)
{
    if (!yolov7_initialized)
    {
        std::cerr<<"YOLOv7 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!ppocrv6_initialized)
    {
        std::cerr<<"PP-OCRv6 is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image); //转为图像数据
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat>ImgVec;
    std::vector<std::vector<Det>>dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov7_plate.Detect_Yolov7Plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::PPOCRV6);
}

int PlateAlgorithm::PlateRecognition_yolov5_PPOCRV5(Plate_ImageData*img,
                                                     PlateDet*PlateDets)
{
    if (!yolov5_initialized)
    {
        std::cerr<<"YOLOv5 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!ppocrv5_initialized)
    {
        std::cerr<<"PP-OCRv5 is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image);
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat> ImgVec;
    std::vector<std::vector<Det>> dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov5_plate.Detect_Yolov5plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::PPOCRV5);
}

int PlateAlgorithm::PlateRecognition_yolov7_PPOCRV5(Plate_ImageData*img,
                                                     PlateDet*PlateDets)
{
    if (!yolov7_initialized)
    {
        std::cerr<<"YOLOv7 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!ppocrv5_initialized)
    {
        std::cerr<<"PP-OCRv5 is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image);
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat> ImgVec;
    std::vector<std::vector<Det>> dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov7_plate.Detect_Yolov7Plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::PPOCRV5);
}

int PlateAlgorithm::PlateRecognition_yolov5_LPRNet(Plate_ImageData*img,PlateDet*PlateDets)
{
    if (!yolov5_initialized)
    {
        std::cerr<<"YOLOv5 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!lprnet_initialized)
    {
        std::cerr<<"LPRNet is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image);
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat>ImgVec;
    std::vector<std::vector<Det>>dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov5_plate.Detect_Yolov5plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::LPRNet);
}

int PlateAlgorithm::PlateRecognition_yolov7_LPRNet(Plate_ImageData*img,PlateDet*PlateDets)
{
    if (!yolov7_initialized)
    {
        std::cerr<<"YOLOv7 plate detector is not initialized"<<std::endl;
        return -1;
    }
    if (!lprnet_initialized)
    {
        std::cerr<<"LPRNet is not initialized"<<std::endl;
        return -1;
    }
    if (img==NULL || img->image==NULL || PlateDets==NULL ||
        img->width<=0 || img->height<=0 || img->channels!=3)
    {
        std::cerr<<"invalid image or output buffer"<<std::endl;
        return -1;
    }
    cv::Mat image_temp(img->height,img->width,CV_8UC3,img->image);
    if (image_temp.empty())
    {
        std::cout<<"img is empty"<<std::endl;
        return -1;
    }
    std::vector<cv::Mat>ImgVec;
    std::vector<std::vector<Det>>dets;
    ImgVec.push_back(image_temp);
    HZFLAG status=yolov7_plate.Detect_Yolov7Plate(ImgVec,dets);
    if (status!=HZ_SUCCESS)
    {
        return -1;
    }
    return fillRecognitionResults(image_temp,dets,PlateDets,
                                  RecognitionBackend::LPRNet);
}

int PlateAlgorithm::fillRecognitionResults(cv::Mat &image_temp,
                                           const std::vector<std::vector<Det>> &dets,
                                           PlateDet *PlateDets,
                                           RecognitionBackend backend)
{
    PlateDets[0].plate_index=0;
    if (dets.empty())
    {
        return 0;
    }
    if (dets.size()!=1)
    {
        std::cerr<<"unexpected detector batch size"<<std::endl;
        return -1;
    }

    const std::vector<Det> &results=dets.front();
    const std::size_t result_count=
        std::min(results.size(),plate_result_capacity);
    if (results.size()>plate_result_capacity)
    {
        std::cerr<<"plate detections truncated to caller capacity"
                 <<std::endl;
    }
    for (std::size_t j=0; j<result_count; ++j)
    {
        if (results[j].key_points.size()<8 ||
            PlateDets[j].plate_license==NULL)
        {
            std::cerr<<"invalid plate result buffer or detector key points"
                     <<std::endl;
            return -1;
        }

        cv::Point2f order_rect[4];
        for (int k=0; k<4; ++k)
        {
            order_rect[k]=cv::Point(results[j].key_points[2*k],
                                    results[j].key_points[2*k+1]);
            PlateDets[j].key_points[2*k]=results[j].key_points[2*k];
            PlateDets[j].key_points[2*k+1]=results[j].key_points[2*k+1];
        }
        cv::Mat roiImg=getTransForm(image_temp,order_rect);
        if (results[j].label)
        {
            roiImg=get_split_merge(roiImg);
        }

        std::string plate_str;
        HZFLAG status=HZ_ERROR;
        const char *backend_name="unknown";
        std::string backend_error;
        switch (backend)
        {
        case RecognitionBackend::PPOCRV5:
            backend_name="PP-OCRv5";
            status=plate_recognition_ppocrv5.PlateRecognitionOCRV5Run(
                roiImg,plate_str);
            backend_error=plate_recognition_ppocrv5.LastError();
            break;
        case RecognitionBackend::PPOCRV6:
            backend_name="PP-OCRv6";
            status=plate_recognition_ppocrv6.PlateRecognitionOCRV6Run(
                roiImg,plate_str);
            backend_error=plate_recognition_ppocrv6.LastError();
            break;
        case RecognitionBackend::LPRNet:
            backend_name="LPRNet";
            status=plate_recognition_lprnet.PlateRecognitionLPRNetRun(
                roiImg,plate_str);
            backend_error=plate_recognition_lprnet.LastError();
            break;
        }
        if (status!=HZ_SUCCESS)
        {
            std::cerr<<backend_name<<" recognition failed: "
                     <<backend_error
                     <<std::endl;
            return -1;
        }
        if (plate_str.size()>=plate_license_capacity)
        {
            std::cerr<<"recognized plate text exceeds caller buffer capacity"
                     <<std::endl;
            return -1;
        }

        PlateDets[j].bbox=results[j].bbox;
        PlateDets[j].confidence=results[j].confidence;
        PlateDets[j].label=results[j].label;
        std::memcpy(PlateDets[j].plate_license,plate_str.c_str(),
                    plate_str.size()+1);
        // The new recognition-only models do not produce the legacy
        // plate-color output.
        if (PlateDets[j].plate_color!=NULL && plate_color_capacity>0)
        {
            PlateDets[j].plate_color[0]='\0';
        }
    }
    for (std::size_t j=0; j<result_count; ++j)
    {
        PlateDets[j].plate_index=static_cast<int>(result_count);
    }
    return 0;
}




/** 
 * @brief                   车牌检测(yolov8_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateAlgorithm::PlateRecognition_yolov8(Plate_ImageData*img,PlateDet*PlateDets)
{
	(void)img;
	(void)PlateDets;
	return -1;
}

float PlateAlgorithm::getNorm2(float x,float y)
{
    return sqrt(x*x+y*y);
}
cv::Mat PlateAlgorithm::getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]) //透视变换
{
    cv::Point2f w1=order_rect[0]-order_rect[1];
    cv::Point2f w2=order_rect[2]-order_rect[3];
    auto width1 = getNorm2(w1.x,w1.y);
    auto width2 = getNorm2(w2.x,w2.y);
    auto maxWidth = std::max(width1,width2);

    cv::Point2f h1=order_rect[0]-order_rect[3];
    cv::Point2f h2=order_rect[1]-order_rect[2];
    auto height1 = getNorm2(h1.x,h1.y);
    auto height2 = getNorm2(h2.x,h2.y);
    auto maxHeight = std::max(height1,height2);
    //  透视变换
    std::vector<cv::Point2f> pts_ori(4);
    std::vector<cv::Point2f> pts_std(4);

    pts_ori[0]=order_rect[0];
    pts_ori[1]=order_rect[1];
    pts_ori[2]=order_rect[2];
    pts_ori[3]=order_rect[3];

    pts_std[0]=cv::Point2f(0,0);
    pts_std[1]=cv::Point2f(maxWidth,0);
    pts_std[2]=cv::Point2f(maxWidth,maxHeight);
    pts_std[3]=cv::Point2f(0,maxHeight);

    cv::Mat M = cv::getPerspectiveTransform(pts_ori,pts_std);
    cv:: Mat dstimg;
    cv::warpPerspective(src_img,dstimg,M,cv::Size(maxWidth,maxHeight));
    return dstimg;
}
 
cv::Mat PlateAlgorithm::get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));
    return out;
}


/** 
 * @brief               反初始化
 * @return              HZFLAG 
 */		
int PlateAlgorithm::Release(Config*config)
{
    (void)config;
    releaseInitializedComponents();
    std::cout<<"plate algorithm release finish"<<std::endl;
    return 0;
}

void PlateAlgorithm::releaseInitializedComponents()
{
    if (yolov5_initialized)
    {
        yolov5_plate.ReleaseDetector_Yolov5plate();
        yolov5_initialized=false;
    }
    if (yolov7_initialized)
    {
        yolov7_plate.ReleaseDetector_Yolov7Plate();
        yolov7_initialized=false;
    }
    if (ppocrv6_initialized)
    {
        plate_recognition_ppocrv6.PlateRecognitionOCRV6Release();
        ppocrv6_initialized=false;
    }
    if (ppocrv5_initialized)
    {
        plate_recognition_ppocrv5.PlateRecognitionOCRV5Release();
        ppocrv5_initialized=false;
    }
    if (lprnet_initialized)
    {
        plate_recognition_lprnet.PlateRecognitionLPRNetRelease();
        lprnet_initialized=false;
    }
}


/** 
 * @brief                   车牌初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
void*Initialize(Config*config)
{
    try
    {
        std::unique_ptr<PlateAlgorithm> plate_algorithm(new PlateAlgorithm());
        if (plate_algorithm->Initialize(config)!=0)
        {
            return NULL;
        }
        return plate_algorithm.release();
    }
    catch (const std::exception &exception)
    {
        std::cerr<<"plate algorithm initialization exception: "
                 <<exception.what()<<std::endl;
    }
    catch (...)
    {
        std::cerr<<"unknown plate algorithm initialization exception"
                 <<std::endl;
    }
    return NULL;
}

/** 
 * @brief                   车牌检测识别(yolov5)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov5(img,PlateDets);
    });
}

/** 
 * @brief                   车牌检测(yolov7_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov7(img,PlateDets);
    });
}

/**
 * @brief                   车牌检测识别(yolov5 + PP-OCRv5)
 */
int PlateRecognition_yolov5_PPOCRV5(void*p,Plate_ImageData*img,
                                    PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov5_PPOCRV5(img,PlateDets);
    });
}

/**
 * @brief                   车牌检测识别(yolov7 + PP-OCRv5)
 */
int PlateRecognition_yolov7_PPOCRV5(void*p,Plate_ImageData*img,
                                    PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov7_PPOCRV5(img,PlateDets);
    });
}

/**
 * @brief                   车牌检测识别(yolov5 + LPRNet)
 * @param img               Plate_ImageData
 * @param PlateDet          车牌检测识别结果列表
 * @return                  HZFLAG
 */
int PlateRecognition_yolov5_LPRNet(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov5_LPRNet(img,PlateDets);
    });
}

/**
 * @brief                   车牌检测识别(yolov7 + LPRNet)
 * @param img               Plate_ImageData
 * @param PlateDet          车牌检测识别结果列表
 * @return                  HZFLAG
 */
int PlateRecognition_yolov7_LPRNet(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov7_LPRNet(img,PlateDets);
    });
}


/** 
 * @brief                   车牌检测(yolov8_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    if (p==NULL)
    {
        return -1;
    }
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return GuardApiCall([&]() {
        return plate_algorithm->PlateRecognition_yolov8(img,PlateDets);
    });
}

/** 
 * @brief               反初始化
 * @return              HZFLAG 
 */		
int Release(void*p,Config*config)
{
    if (p==NULL)
    {
        return -1;
    }
    std::unique_ptr<PlateAlgorithm> plate_algorithm((PlateAlgorithm*)p);
    return GuardApiCall([&]() {
        return plate_algorithm->Release(config);
    });
}
