#ifndef _DATATYPES_
#define _DATATYPES_

#include <vector>
#include <utility>
#include <time.h>
#include <string>

enum HZFLAG
{
	HZ_FILEOPENFAILED,            //文件打开失败
	HZ_IMGEMPTY,                  //图像为空
	HZ_SUCCESS,                   //成功
	HZ_ERROR,                     //失败
	HZ_WITHOUTMODEL,              //模型不存在                                                     
	HZ_IMGFORMATERROR,            //图像格式错误
	HZ_CLASSEMPTY,                //类别文件为空
	HZ_LOGINITFAILED,             //日志初始化失败           
	HZ_CONFIGLOADFAILED,          //configi加载失败            
	HZ_INITFAILED,                //初始化i失败                                             
};

//传入图像的数据结构
typedef struct
{
    unsigned char* image;  		
    int width;
    int height;
    int channels;          		 
}Plate_ImageData;

struct affineMatrix  //letter_box  仿射变换矩阵
{
    float i2d[6];       //仿射变换正变换
    float d2i[6];       //仿射变换逆变换
};
struct bbox 
{
    float x1,x2,y1,y2;
    float landmarks[8]; //5个关键点
    float score;
};
const float color_list[5][3] =
{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
    {255,255,0},
};

typedef struct 
{
	float xmin;
	float xmax;
	float ymin;
	float ymax;
}PlateRect;
typedef struct 
{
	PlateRect bbox;
	int label;
	int id;
	float confidence;
	std::vector<float> key_points;    //关键点
}Det;

typedef struct 
{
	PlateRect bbox;                   //bbox
	int label;						  //0->单层车牌 1->双层车牌
	float confidence;                 //检测置信度
	float key_points[8];              //关键点坐标
	char *plate_color;				  //车牌颜色 0->黑色,1->蓝色,2->绿色,3->白色,4->黄色
	char *plate_license;			  //车牌号
	int plate_index;				  //表示一张图像里面车牌数量		
}PlateDet;

//初始化的参数
typedef struct 
{
	int gpu_id;

	//yolov5plate detect params
	char* Yolov5PlateDetectModelPath;
	float yolov5plate_confidence_thresh;
	int yolov5plate_detect_bs;        
	float yolov5plate_nms_thresh;
	bool yolov5plate_detect_enable=false;

	//yolov7plate detect params
	char* Yolov7PlateDetectModelPath;
	float yolov7plate_confidence_thresh;
	int yolov7plate_detect_bs;        
	float yolov7plate_nms_thresh;
	bool yolov7plate_detect_enable=false;

	//yolov8plate detect params
	char* Yolov8PlateDetectModelPath;
	float yolov8plate_confidence_thresh;
	int yolov8plate_detect_bs;        
	float yolov8plate_nms_thresh;
	bool yolov8plate_detect_enable=false;

	//plate recogniton
	char* PlateReconitionModelPath;             
	int plate_recognition_bs;                       //人脸识别batchsize
	bool plate_recognition_enable=false;
}Config;
#endif
