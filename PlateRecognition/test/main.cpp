#include<iostream>
#include"PlateRecognition.h"
#include<fstream>
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
    config.plate_recognition_bs=1;
    config.plate_recognition_enable=true;
    config.PlateReconitionModelPath="/home/pcb/Algorithm/Plate/PlateRecognition/PlateRecognition/test/crnn_m_0.9934_0.989_best.onnx";       //识别模型路径

    PlateRecognition plate_rcognition;
    plate_rcognition.PlateRecognitionInit(&config);

	
    int string_indx=0;
    int right=0;
    cv::Mat plate_img;
    double time_count=0.0;
    //image path      
    std::string imagepath="/home/pcb/Algorithm/Plate/data/123_plate_detect/1111";                                                           
    //save path
    std::string imagepath1="/home/pcb/Algorithm/Plate/PlateRecognition/PlateDetectionRecognition/test/123_result";
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imagepath.c_str()),imagList,fileType);

    for (size_t i = 0; i < imagList.size(); i++)
    {
        plate_img=cv::imread(imagList[i]);
        if (plate_img.empty())
        {
            continue;
        }
        string_indx++;
        std::string plate_num;
        std::string plate_color;
        auto start = std::chrono::system_clock::now();
        plate_rcognition.PlateRecognitionRun(plate_img,plate_num,plate_color);
        auto end = std::chrono::system_clock::now();
        time_count+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout<<"plate:"<<plate_num<<" plate_color:"<<plate_color<<std::endl;
    }
    float accury=(1.0*right)/(1.0*string_indx);
    std::cout<<"average time:"<<time_count/string_indx<<"us all num:"<<string_indx<<" plate accury:"<<accury<<std::endl;
    plate_rcognition.PlateRecognitionRelease();
    return 0;

}