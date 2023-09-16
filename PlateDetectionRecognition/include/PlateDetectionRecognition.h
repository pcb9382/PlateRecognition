#pragma once

#ifndef _PLATERECOGNITION_
#define _PLATERECOGNITION_

#include <iostream>
#include "DataTypes_Plate.h"
#include <opencv2/opencv.hpp>

#ifdef __cplusplus 
extern "C" { 
#endif 
/** 
 * @brief                   车牌初始化函数
 * @param config			模块配置参数结构体
 * @return                  HZFLAG
 */
void*Initialize(Config*config);

/** 
 * @brief                   车牌检测识别(yolov5)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/** 
 * @brief                   车牌检测(yolov7_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets);


/** 
 * @brief                   车牌检测(yolov8_plate)
 * @param img			    Plate_ImageData
 * @param PlateDet		    车牌检测识别结果列表
 * @return                  HZFLAG
 */		
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/** 
 * @brief               反初始化
 * @return              HZFLAG 
 */		
int Release(void*p,Config*config);

#ifdef __cplusplus 
} 
#endif

#endif



