 English|[简体中文](./README.md)

If you find it useful, you may wish to give a Star ⭐️🌟support~ Thank you!

# Acknowledgments & Contact 
### 1.WeChat ID: cbp931126
Add me WeChat(Note: FaceAlgorithm) to pull you into the group
### 2.QQ Group：517671804


# FaceAlgorithm
## Features
1. Face detection (retinaface, yolov5face, yolov7face), face rotation angle calculation (pitch angle, yaw angle), face correction, face recognition,mask recognition,age_gender recognition,silent living recognition;
2. All models use C++ and TensorRT to accelerate inference, and the preprocess and postprocess of yolov7face uses cuda acceleration,(other model acceleration optimizations can also be referred to);
3. All models use C++ and OnnxRuntime.OpenVINO, NCNN Accelerated Inference (TO DO);
4. Construct similar to NV Deepstream, support a variety of inference frameworks(TensorRT, OnnxRuntime, OpenVINO, NCNN), for multi-channel RTSP pull stream + hard decoding + Pipeline + push stream(TO DO);
5. Automatically generate the corresponding engine according to different graphics card models (if there are other graphics cards in the folder, delete the engine to regenerate the engine corresponding to the graphics card in use);
6. Provide C/C++ interface, which can be directly ported to the project;
7. General process of face recognition:

	1)Face detection (images, video streams)
			
	2)According to the angle by each face, the face at the appropriate angle is screened out for face correction and face recognition

	3)Face correction (based on 5 key points of the face)
			
	4)Face feature feature extraction (512-dimensional features)
					
	5)Face feature comparison (face similarity calculation)

8. Description of conditional compilation tests
	| Test category |  enable    |  description   |
	|:----------:|:----------:|:----------:|
   |face_detect                        |1|           Face detection                         |
   |yolov5face_detect				      |1|           yolov5face Face detection              |
   |yolov5face_detect				      |1|           yolov7face Face detection              |
   |yolov8face_detect				      |1|           yolov8face Face detection              |
   |face_recognition                   |1|           Face recognition (face feature extraction) + similarity calculation   |
   |face_detect_tracker                |1|           Face detection tracking                      |
   |face_detect_aligner_recognitiion   |0|           Face detection - correction - recognition (face feature extraction)   |
   |mask_recognition                   |1|           Mask identification                         |
   |gender_age_recognition             |1|           Gender age identification                      |
   |silnet_face_anti_spoofing          |1|           Silent living identification                      |

## Algorithm description
### 1.Face detection
#### 1)retinaface(mobilenet0.25,R50 Backbone requires your own code modifications）
   ![demoimg1](https://insightface.ai/assets/img/github/11513D05.jpg)
#### 2)yolov5face(yolov5sface(640*640)，The n,m,l,x model needs to convert the corresponding ONNX itself)
   <img src="./resources/yolov5face_test.jpg" alt="drawing" width="800"/> 

#### 3)yolov7face(yolov7sface(640*640),Models of different sizes need to be converted by themselves)
   <img src="./resources/yolov7face_test.jpg" alt="drawing" width="800"/>

#### 4)yolov8facee(yolov8sface(640*640),Models of different sizes need to be converted by themselves)
  <img src="./resources/yolov8nface_test.jpg" alt="drawing" width="800"/>

#### 5) ***Large family***
| Method              | Backbone       | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
|:-------------------:|:--------------:|:-----:|:------:|:-----:|:-----------:|:----------:|
| RetinaFace (CVPR20) | ResNet50       | 94.92 | 91.90  | 64.17 | 29.50       | 37.59      |
| ***YOLOv5s***       | CSPNet         | 94.67 | 92.75  | 83.03 | 7.075       | 5.751      |
| **YOLOv5s6**        | CSPNet         | 95.48 | 93.66  | 82.8  | 12.386      | 6.280      |
| ***YOLOv5m***       | CSPNet         | 95.30 | 93.76  | 85.28 | 21.063      | 18.146     |
| **YOLOv5m6**        | CSPNet         | 95.66 | 94.1   | 85.2  | 35.485      | 19.773     |
| ***YOLOv5l***       | CSPNet         | 95.78 | 94.30  | 86.13 | 46.627      | 41.607     |
| ***YOLOv5l6***      | CSPNet         | 96.38 | 94.90  | 85.88 | 76.674      | 45.279     |
| ***yolov7-tiny***   | 640            | 94.7  | 92.6   | 82.1  |  13.2       | -          | 
| ***yolov7s***       | 640            | 94.8  | 93.1   | 85.2  |  16.8       | -          |
| ***yolov7***        | 640            | 96.9  | 95.5   | 88.0  |  103.4      | -          |
| ***yolov7+TTA***    | 640            | 97.2  | 95.8   | 87.7  |  103.4      | -          |
| ***yolov7-w6***     | 960            | 96.4  | 95.0   | 88.3  |  89.0       | -          |
| ***yolov7-w6+TTA*** | 1280           | 96.9  | 95.8   | 90.4  |  89.0       | -          |
| ***yolov8s***       | 640            | 96.0  | 94.2   | 82.6  |  -          | -          | 
| ***yolov8m***       | 640            | 96.6  | 95.0   | 84.1  |  -          | -          |

#### 6) ***Small family***

| Method               | Backbone        | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
| -------------------- | --------------- | ----- | ------ | ----- | ----------- | ---------- |
| RetinaFace (CVPR20   | MobileNet0.25   | 87.78 | 81.16  | 47.32 | 0.44        | 0.802      |
| FaceBoxes (IJCB17)   |                 | 76.17 | 57.17  | 24.18 | 1.01        | 0.275      |
| ***YOLOv5n***        | ShuffleNetv2    | 93.74 | 91.54  | 80.32 | 1.726       | 2.111      |
| ***YOLOv5n-0.5***    | ShuffleNetv2    | 90.76 | 88.12  | 73.82 | 0.447       | 0.571      |
| ***yolov7-lite-t***  |                 | 88.7  | 85.2   | 71.5  |  0.8        |            |
| ***yolov7-lite-s***  |                 | 92.7  | 89.9   | 78.5  |  3.0        |  -         |
| ***yolov8-lite-t***  | 640             | 90.3  | 87.5   | 72.8  |  -          |  -         |
| ***yolov8-lite-s***  | 640             | 93.4  | 91.1   | 77.7  |  -          |  -         |
| ***yolov8n***        | 640             | 94.5  | 92.2   | 79.0  |  -          |  -         |
#### 7) TensorRT

|   Backbone   | Pytorch(1.10.0+cu102) | TRT_FP16(RTX2080Ti) |TRT_FP16(RTX3090)@640|
| :----------: | :---------: | :---------------: |:---------------: |
| yolov5n-0.5  |     7.7 ms     |        2.1 ms       |             |                 |
| yolov5n-face |     7.7 ms     |        2.4 ms       |             |                 |
| yolov5m-face |     9.9 ms     |        3.3 ms       |             |                 |
| yolov5l-face |    15.9 ms     |        4.5 ms       |             |                 |
| RetinaFace   |      -         |         -           |  1.38ms (Preprocess+postprocess+inference+angle cal)     |
| yolov5s-face |     5.6 ms     |        2.2 ms       |  1.47ms (Preprocess+postprocess+inference+angle cal)     |
| yolov7s-face |     -          |        -            |  1.69ms (Preprocess+postprocess+inference+angle cal)     |
| yolov8n-face |     -          |        -            |  1.03ms (Preprocess+postprocess+inference+angle cal)     |

### 2.Face recognition

#### 1) arcface(R50)

#### 2)arcface(R101,You need to download the model modification code yourself)
<div align="left">
  <img src="https://insightface.ai/assets/img/github/facerecognitionfromvideo.PNG" width="800"/>
</div>


### 3.Mask identification
#### 1)Detect->face crop->mask recognition(mask classification)
![demoimg1](https://insightface.ai/assets/img/github/cov_test.jpg)

### 4.gender_age recognition
#### 1)Face detection ->face cropping->age_gender recognition
<div align="left">
  <img src="https://insightface.ai/assets/img/github/t1_genderage.jpg" width="800"/>
</div>

### 5.Silent living identification
#### 1)Silent-Face-Anti-Spoofing
   
|name| sample| result |image| sample| result |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
0.jpg|<img src="./FaceAlgorithm_Test/antispoofing/0.jpg" width="300" height="300"/>|fake|1.jpg|<img src="./FaceAlgorithm_Test/antispoofing/1.jpg" width="300" height="300"/>|fake
2.jpg|<img src="./FaceAlgorithm_Test/antispoofing/2.jpg" width="300" height="300"/>|real|3.jpg|<img src="./FaceAlgorithm_Test/antispoofing/3.jpg" width="300" height="300"/>|real
4.jpg|<img src="./FaceAlgorithm_Test/antispoofing/4.jpg" width="300" height="300"/>|fake|5.jpg|<img src="./FaceAlgorithm_Test/antispoofing/5.jpg" width="300" height="300"/>|fake

### 6.FaceAlignment
<p align="center">Points mark-up(ordered by point names):</p>
<div align="center">
	<img src="./resources/2d106markup.jpg" alt="markup" width="320">
   <img src="./resources/C_jiaguo.gif" alt="videovis" width="272">
</div>
<p align="center">Image result:</p>
<div align="center">
	<img src="./resources/t1_out.jpg" alt="imagevis" width="800">
</div>

### 8.tracker
#### 1)ByteTracker(Add face bbox and face key as input for tracking to modify bugs)

# Algorithm interface
```
/** 
 * @brief               人脸初始化函数
 * @param config        模块配置参数结构体
 * @return              HZFLAG
 */
HZFLAG Initialize(Config& config);


/** 
 * @brief               人脸检测
 * @param img           opencv　Mat格式
 * @param FaceDets      人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return              HZFLAG
 */		
HZFLAG Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);


/** 
 * @brief               人脸检测(yolov5_face)
 * @param img           opencv　Mat格式
 * @param FaceDets      人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
 * @return              HZFLAG
 */		
HZFLAG Yolov5Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
   * @brief             人脸检测(yolov7_face)
   * @param img         opencv　Mat格式
   * @param FaceDets    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
   * @return            HZFLAG
   */		
HZFLAG Yolov7Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
   * @brief             人脸检测(yolov8_face)
   * @param img         opencv　Mat格式
   * @param FaceDets    人脸检测结果列表，包括人脸bbox，置信度，五个关键点坐标
   * @return            HZFLAG
   */		
HZFLAG Yolov8Face_Detect(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
 * @brief               人脸检测跟踪(视频流)
 * @param img           opencv　Mat格式
 * @param FaceDets      FaceDets	人脸检测结果列表，包括人脸bbox，id,置信度，偏航角度，俯仰角度，五个关键点坐标
 * @return              HZFLAG
 */	
HZFLAG Face_Detect_Tracker(std::vector<cv::Mat>&img, std::vector<std::vector<FaceDet>>&FaceDets);

/** 
 * @brief               人脸矫正
 * @param Faceimg       需要矫正的人脸图像(矩形框bbox外扩1.2倍得到的人脸图像然后进行矫正!!!!)
 * @param KeyPoints     人脸关键点
 * @param Face_Aligener 矫正之后的图像
 * @return              HZFLAG
 */	
HZFLAG Face_Aligner(cv::Mat&Face_image,cv::Point2f *KeyPoints,cv::Mat&Face_Aligener);

/** 
 * @brief               人脸特征提取
 * @param Face_Aligener 经过人脸矫正的人脸图像
 * @param Face_Feature  人脸特征(512维特征)
 * @return              HZFLAG
 */		
HZFLAG Face_Feature_Extraction(cv::Mat&Face_Aligener,Feature&Face_Feature);


/** 
 * @brief               计算人脸特征的相似度
 * @param Feature1      经过人脸矫正的人脸图像
 * @param Feature2      人脸特征(512维特征)
 * @return float        相似度得分               
 */	
float Cal_Score(Feature&Feature1,Feature&Feature2);

/** 
 * @brief               人脸戴口罩识别
 * @param img           需要识别的人脸戴口罩图像
 * @param Result        人脸戴口罩识别结果
 * @return              HZFLAG
 */
HZFLAG Mask_Recognition(cv::Mat &img,float&pred);

/** 
 * @brief               性别年龄识别
 * @param img           需要识别的人脸图像
 * @param Result        性别年龄识别别结果
 * @return              HZFLAG
 */
HZFLAG Gender_Age_Recognition(cv::Mat &img,attribute&gender_age);

/** 
 * @brief               静默活体检测
 * @param img           需要检测的人脸图像
 * @param Result        静默活体检测识别结果
 * @return              HZFLAG
 */
HZFLAG Silent_Face_Anti_Spoofing(cv::Mat&img, SilentFace&silentface);

/** 
 * @brief               反初始化
 * @return              HZFLAG
 */		
HZFLAG Release(Config& config);
```
# How to use
## 1.Model and test data download
model and test data ([Baidu Drive](https://pan.baidu.com/s/1vmrxV5oyzJKmJ_UABqfK3w)code: gbub)
| name |  function    |  description   |
|:----------:|:----------:|:----------:|
|FaceDetect.wts                        |Face detection|        
|FaceRecognition.wts				   |Face recognition|       
|GenderAge.onnx                        |Age gender identification|          
|MaskRecognition.onnx                  |Mask identification|          
|yolov5s-face_bs=1.onnx                |yolov5s Face detection|          
|yolov5s-face_bs=4.onnx                |yolov5s Face detection| bs=4
|yolov7s-face_bs=1.onnx                |yolov7s Face detection|          
|yolov7s-face_bs=4.onnx                |yolov7s Face detection| bs=4
|yolov8n-face_bs=1.onnx                |yolov8n Face detection|          
|yolov8n-face_bs=4.onnx                |yolov8n Face detection| bs=4      
|2.7_80x80_MiniFASNetV2.onnx           |Silent living identification|
|2d106det_bs=1.onnx                    |106keypoints|           bs=1 

## 2.environment
1. ubuntu20.04+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1(Test passed)
2. ubuntu18.04+cuda10.2+cudnn8.2.1+TrnsorRT8.2.5.1(Test passed)
3. Win10+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1      (Test passed)
4. For other environments, please try it yourself or join a group to understand


## 3.compilation

1. Change the CMakeLists .txt in the root directory and set the directory where tensorrt is installed
```
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. By default, OpenCV is installed, CUDA, CUDNN are installed
3. In order to debug the ```-g o0``` version is compiled by default, if you want to speed things up, compile the release version

4. Compile using Visual Studio Code shortcuts (choose one of 4,5):
```
   ctrl+shift+B
```
5. Compile using the command line (choose one of 4,5):
```
   mkdir build
   cd build
   cmake ..
   make -j6
```
 

# References
1. https://github.com/deepcam-cn/yolov5-face
2. https://github.com/wang-xinyu/tensorrtx
3. https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
4. https://github.com/linghu8812/tensorrt_inference
5. https://github.com/derronqi/yolov7-face/tree/main
6. https://github.com/we0091234/yolov7-face-tensorrt
7. https://github.com/deepinsight/insightface
8. https://github.com/derronqi/yolov8-face
