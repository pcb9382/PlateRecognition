#include<iostream>
#include"PlateDetectionRecognition.h"
#include"platedetector_yolov7.h"
#include"platedetector_yolov5.h"
#include"PlateRecognition.h"


class PlateAlgorithm
{
public:
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
    PlateRecognition plate_recognition;
private:
    cv::Mat get_split_merge(cv::Mat &img);   //双层车牌 分割 拼接
    cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]); //透视变换
    float getNorm2(float x,float y);
};


/** 
 * @brief                   车牌初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
int PlateAlgorithm::Initialize(Config*config)
{
    if (config->yolov5plate_detect_enable)
    {
        yolov5_plate.InitDetector_Yolov5plate(config);
    }
    if(config->yolov7plate_detect_enable)
    {
        yolov7_plate.InitDetector_Yolov7Plate(config);
    }
    if (config->plate_recognition_enable)
    {
        plate_recognition.PlateRecognitionInit(config);
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
    if (img->image==NULL)
    {
        std::cout<<"img is null"<<std::endl;
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
    yolov5_plate.Detect_Yolov5plate(ImgVec,dets);

    for (size_t i = 0; i < dets.size(); i++)
    {
        for (size_t j = 0; j < dets[i].size(); j++)
        {
            //cv::Mat plate_img=image_temp.clone();
            cv::Point2f  order_rect[4];
            for (int k= 0; k<4; k++)
            {
                order_rect[k]=cv::Point(dets[i][j].key_points[2*k],dets[i][j].key_points[2*k+1]);
                PlateDets[j].key_points[2*k]=dets[i][j].key_points[2*k];
                PlateDets[j].key_points[2*k+1]=dets[i][j].key_points[2*k+1];
            }
            cv::Mat roiImg = getTransForm(image_temp,order_rect);  //根据关键点进行透视变换
            int label = dets[i][j].label;
            if (label)             //判断是否双层车牌，是的话进行分割拼接
            {
                roiImg=get_split_merge(roiImg);
            }
            std::string plate_str;
            std::string plate_color;
            plate_recognition.PlateRecognitionRun(roiImg,plate_str,plate_color);
            PlateDets[j].bbox=dets[i][j].bbox;
            PlateDets[j].confidence =dets[i][j].confidence;
            PlateDets[j].label=dets[i][j].label;
            //PlateDets[j].plate_license=new char[strlen(plate_str.c_str())+1];
            //PlateDets[j].plate_color=new char[strlen(plate_color.c_str())+1];
            strcpy(PlateDets[j].plate_license, plate_str.c_str());
            strcpy(PlateDets[j].plate_color, plate_color.c_str());
            PlateDets[j].plate_index=dets[i].size();
        }
    }
    return 0;
}
/** 
 * @brief                   车牌检测(yolov7_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateAlgorithm::PlateRecognition_yolov7(Plate_ImageData*img,PlateDet*PlateDets)
{
    if (img->image==NULL)
    {
        std::cout<<"img is null"<<std::endl;
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
    yolov7_plate.Detect_Yolov7Plate(ImgVec,dets);

    for (size_t i = 0; i < dets.size(); i++)
    {
        for (size_t j = 0; j < dets[i].size(); j++)
        {
            //cv::Mat plate_img=image_temp.clone();
            cv::Point2f  order_rect[4];
            for (int k= 0; k<4; k++)
            {
                order_rect[k]=cv::Point(dets[i][j].key_points[2*k],dets[i][j].key_points[2*k+1]);
                PlateDets[j].key_points[2*k]=dets[i][j].key_points[2*k];
                PlateDets[j].key_points[2*k+1]=dets[i][j].key_points[2*k+1];
            }
            cv::Mat roiImg = getTransForm(image_temp,order_rect);  //根据关键点进行透视变换
            int label = dets[i][j].label;
            if (label)             //判断是否双层车牌，是的话进行分割拼接
            {
                roiImg=get_split_merge(roiImg);
            }
            std::string plate_str;
            std::string plate_color;
            plate_recognition.PlateRecognitionRun(roiImg,plate_str,plate_color);
            PlateDets[j].bbox=dets[i][j].bbox;
            PlateDets[j].confidence =dets[i][j].confidence;
            PlateDets[j].label=dets[i][j].label;
            //PlateDets[j].plate_license=new char[strlen(plate_str.c_str())+1];
            //PlateDets[j].plate_color=new char[strlen(plate_color.c_str())+1];
            strcpy(PlateDets[j].plate_license, plate_str.c_str());
            strcpy(PlateDets[j].plate_color, plate_color.c_str());
            PlateDets[j].plate_index=dets[i].size();
        }
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
    if (config->yolov5plate_detect_enable)
    {
        yolov5_plate.ReleaseDetector_Yolov5plate();
    }
    if(config->yolov7plate_detect_enable)
    {
        yolov7_plate.ReleaseDetector_Yolov7Plate();
    }
    if (config->plate_recognition_enable)
    {
        plate_recognition.PlateRecognitionRelease();
    }
    std::cout<<"plate algorithm init finash"<<std::endl;
    return 0;
    
}


/** 
 * @brief                   车牌初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
void*Initialize(Config*config)
{
    PlateAlgorithm *plate_algorithm=new PlateAlgorithm();
    plate_algorithm->Initialize(config);
    return plate_algorithm;
}

/** 
 * @brief                   车牌检测识别(yolov5)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return plate_algorithm->PlateRecognition_yolov5(img,PlateDets);
}

/** 
 * @brief                   车牌检测(yolov7_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return plate_algorithm->PlateRecognition_yolov7(img,PlateDets);
}


/** 
 * @brief                   车牌检测(yolov8_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets)
{
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    return plate_algorithm->PlateRecognition_yolov8(img,PlateDets);
}

/** 
 * @brief               反初始化
 * @return              HZFLAG 
 */		
int Release(void*p,Config*config)
{
    PlateAlgorithm *plate_algorithm=(PlateAlgorithm*)p;
    plate_algorithm->Release(config);
    delete plate_algorithm;
    plate_algorithm=NULL;
    return 0;
}