#include<iostream>
#include"platedetector_yolov5.h"
#include <sys/types.h>
#include <iostream>
#include<dirent.h>
#include <sys/types.h>
#include <string.h>
#include <sys/stat.h>

std::string getHouZhui(std::string fileName)
{
    //    std::string fileName="/home/xiaolei/23.jpg";
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


int main()
{
    Config config;
    config.gpu_id=0;
    config.yolov5plate_detect_bs=1;
    config.yolov5plate_detect_enable=true;
    config.yolov5plate_nms_thresh=0.3;
    config.yolov5plate_confidence_thresh=0.5;
    config.Yolov5PlateDetectModelPath="./yolov5plate.onnx";

    Detector_Yolov5plate detector_yolov5plate;
    detector_yolov5plate.InitDetector_Yolov5plate(&config);
    std::string imagepath="./data";
    std::string imagepath1="./result";
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imagepath.c_str()),imagList,fileType);

    for (size_t i = 0; i < imagList.size(); i++)
    {
        std::vector<cv::Mat>ImgVec;
        cv::Mat image=cv::imread(imagList[i]);//
        if(image.empty())
        {
            continue;
        }
        ImgVec.push_back(image);
        std::vector<std::vector<Det>>dets;
        auto start = std::chrono::system_clock::now();
        detector_yolov5plate.Detect_Yolov5plate(ImgVec,dets);
        auto end = std::chrono::system_clock::now();
	    std::cout<<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<<"us"<<std::endl;;
        for (int k = 0; k < dets.size(); k++)
        {
            for (size_t f = 0; f < dets[k].size(); f++)
            {
                cv::rectangle(ImgVec[k], cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin),
                cv::Point(dets[k][f].bbox.xmax, dets[k][f].bbox.ymax), cv::Scalar(255, 0, 0), 2, 8, 0);
                cv::circle(ImgVec[k], cv::Point2f(dets[k][f].key_points[0], dets[k][f].key_points[1]), 4, cv::Scalar(255, 255, 0), -1);
                cv::circle(ImgVec[k], cv::Point2f(dets[k][f].key_points[2], dets[k][f].key_points[3]), 4, cv::Scalar(0, 0, 255), -1);
                cv::circle(ImgVec[k], cv::Point2f(dets[k][f].key_points[4], dets[k][f].key_points[5]), 4, cv::Scalar(0, 255, 0), -1);
                cv::circle(ImgVec[k], cv::Point2f(dets[k][f].key_points[6], dets[k][f].key_points[7]), 4, cv::Scalar(255, 0, 255), -1);
                std::string label3 = cv::format("%d:%.2f", dets[k][f].label,dets[k][f].confidence);
                cv::putText(ImgVec[k], label3, cv::Point(dets[k][f].bbox.xmin, dets[k][f].bbox.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0),1.5);
            }
            std::string::size_type lastPos=imagList[i].find_last_of("/");
            std::string image_name=imagList[i].substr(lastPos+1,imagList[i].size() - lastPos);
            cv::imwrite(imagepath1+"/"+image_name,ImgVec[k]);
            //cv::imshow("show", ImgVec[k]);
            //cv::waitKey(1);
        }
    }
    detector_yolov5plate.ReleaseDetector_Yolov5plate();
    return 0;
}