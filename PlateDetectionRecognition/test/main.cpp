#include<iostream>
#include"PlateDetectionRecognition.h"
#include <sys/types.h>
#include<dirent.h>
#include <sys/types.h>
#include <string.h>
#include <sys/stat.h>
#include <opencv2/freetype.hpp>
#include <chrono>

#define yolov7_plate    1       //yolov5车牌检测
#define yolov5_plate    0       //yolov7车牌检测

std::string getHouZhui(std::string fileName)
{
    int pos=fileName.find_last_of(std::string("."));
    std::string houZui=fileName.substr(pos+1);
    return houZui;
}
int readFileList(char *basePath,std::vector<std::string> &fileList,std::vector<std::string> fileType)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)
        {    ///file
            if (fileType.size())
            {
            std::string houZui=getHouZhui(std::string(ptr->d_name));
            for (auto &s:fileType)
            {
            if (houZui==s)
            {
            fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            break;
            }
            }
            }
            else
            {
                fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            }
        }
        else if(ptr->d_type == 10)    ///link file
            printf("d_name:%s/%s\n",basePath,ptr->d_name);
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            readFileList(base,fileList,fileType);
        }
    }
    closedir(dir);
    return 1;
}
void drawBboxes(cv::Mat &img ,PlateDet*PlateDets, cv::Ptr<cv::freetype::FreeType2>&ft2)//
{
    
    for (size_t f = 0; f <PlateDets[0].plate_index; f++)
    {
        std::string plate_str = PlateDets[f].plate_license;
        std::string plate_color = PlateDets[f].plate_color;
        std::string label3=plate_str+" "+plate_color;
        //标签
        int top = PlateDets[f].bbox.ymin;
        int left = PlateDets[f].bbox.xmin;
        int baseLine1;
        cv::Size labelSize1 = cv::getTextSize(label3, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine1);
        top = cv::max(top, labelSize1.height);
        cv::rectangle(img, cv::Point(left, top-round(2.0*labelSize1.height)), cv::Point(left + round(labelSize1.width), top ), 
                       cv::Scalar(255, 255, 255), cv::FILLED);//+ baseLine1
        ft2->putText(img, label3, cv::Point(left, top-5), 20, cv::Scalar(0, 0, 0), -1,4,true);
    
        cv::circle(img, cv::Point2f(PlateDets[f].key_points[0], PlateDets[f].key_points[1]), 4, cv::Scalar(255, 255, 0), -1);
        cv::circle(img, cv::Point2f(PlateDets[f].key_points[2], PlateDets[f].key_points[3]), 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv::Point2f(PlateDets[f].key_points[4], PlateDets[f].key_points[5]), 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(img, cv::Point2f(PlateDets[f].key_points[6], PlateDets[f].key_points[7]), 4, cv::Scalar(255, 0, 255), -1);
        //画框
        
        cv::rectangle(img, cv::Point(PlateDets[f].bbox.xmin, PlateDets[f].bbox.ymin),
                cv::Point(PlateDets[f].bbox.xmax,PlateDets[f].bbox.ymax), cv::Scalar(255,0, 0), 2, 8, 0);
    }
    cv::imshow("show", img);
    cv::waitKey(1);
}

int main()
{
    Config config;
    config.gpu_id=0;
    //recognition
    config.plate_recognition_bs=1;
    config.plate_recognition_enable=true;
    config.PlateReconitionModelPath="/home/pcb/Algorithm/Plate/PlateRecognition_data/crnn_s_0.989_0.987_best.onnx";  //识别模型路径

#if yolov7_plate
    //ylolv7 plate detect
    config.Yolov7PlateDetectModelPath="/home/pcb/Algorithm/Plate/PlateRecognition_data/yolov7plate_20230909.onnx";//检测模型路径
    config.yolov7plate_confidence_thresh=0.5;
    config.yolov7plate_detect_bs=1;
    config.yolov7plate_nms_thresh=0.3;
    config.yolov7plate_detect_enable=true;
#endif

#if yolov5_plate
    config.Yolov5PlateDetectModelPath="/home/pcb/Algorithm/Plate/PlateRecognition_data/yolov5plate_20230910.onnx";//检测模型路径
    config.yolov5plate_confidence_thresh=0.5;
    config.yolov5plate_detect_bs=1;
    config.yolov5plate_nms_thresh=0.3;
    config.yolov5plate_detect_enable=true;
#endif

    void *p=Initialize(&config);
    
    //字体
    std::string ttf_pathname = "NotoSansCJK-Regular.otf";                   //字体
    cv::Ptr<cv::freetype::FreeType2>ft2=cv::freetype::createFreeType2();
    ft2->loadFontData(ttf_pathname,0);
    
    //image path
    std::string imagepath="/media/pcb/Data2/车牌数据集/CRPD/CRPD_multi/train/images";   //待检测识别的图像文件夹
    //save path
    std::string imagepath1="./yolov7_result";                                         //检测识别结果保存的文件夹
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imagepath.c_str()),imagList,fileType);
    PlateDet PlateDets[10];
    for (size_t j = 0; j < 10; j++)
    {
        PlateDets[j].plate_license=new char[20];
        PlateDets[j].plate_color=new char[6];
    }
    Plate_ImageData*plate_imagedata=new Plate_ImageData();
    //while (1)
    //{
        for (size_t i = 0; i < imagList.size(); i++)
        {
            cv::Mat plateimg=cv::imread(imagList[i]);
            if(plateimg.empty())
            {
                continue;
            }
           
            plate_imagedata->channels=3;
            plate_imagedata->height=plateimg.rows;
            plate_imagedata->width=plateimg.cols;
            plate_imagedata->image=plateimg.data;
            auto start = std::chrono::system_clock::now();
#if yolov5_plate
            PlateRecognition_yolov5(p,plate_imagedata,PlateDets);
#endif

#if yolov7_plate
            PlateRecognition_yolov7(p,plate_imagedata,PlateDets);
#endif
            auto end = std::chrono::system_clock::now();
	        std::cout<<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<<"us"<<std::endl;
            drawBboxes(plateimg,PlateDets,ft2);//
            std::string::size_type lastPos=imagList[i].find_last_of("/");
            std::string image_name=imagList[i].substr(lastPos+1,imagList[i].size() - lastPos);
            cv::imwrite(imagepath1+"/"+image_name,plateimg);
        }
    //}
    for (size_t j = 0; j < 10; j++)
    {
        delete []PlateDets[j].plate_license;
        PlateDets[j].plate_license=NULL;
        delete []PlateDets[j].plate_color;
        PlateDets[j].plate_color=NULL;
    }
    Release(p,&config);
    return 0;
}   