简体中文 | [繁體中文](./README_zh.md) | [English](./README_en.md) | [한국어](./README_ko.md)

如果觉得有用，不妨给个Star⭐️🌟支持一下吧~ 谢谢！

# Acknowledgments & Contact 
### 1.WeChat ID: cbp931126
1. 加入讨论群(备注：PlateAlgorithm),大佬多多多卷卷卷；
2. 加微信可以获得整理好的开源和部分私有车牌检测和识别数据(数据需要30元费用)；

### 2.邮箱:cbp931126@163.com


# Web Test
## 1. 大陆车牌识别、港澳车牌识别、台湾车牌识别在线体验：http://vision.zhoujiayao.com/

## 2. 车牌识别支持的国家和地区
### 亚洲
中国大陆 中国香港 中国澳门 中国台湾 韩国 日本 新加坡 泰国 马来西亚 印度尼西亚 越南 土耳其 以色列 卡塔尔 巴林 柬埔寨 格鲁吉亚 亚美尼亚 阿塞拜疆

### 欧洲
英国 法国 德国 意大利 西班牙 瑞士 荷兰 奥地利 比利时 葡萄牙 希腊 爱尔兰 瑞典 挪威 丹麦 芬兰 波兰 捷克 匈牙利 罗马尼亚 克罗地亚 冰岛 卢森堡 塞浦路斯 马耳他 保加利亚 斯洛伐克 斯洛文尼亚 爱沙尼亚 立陶宛 拉脱维亚 乌克兰 塞尔维亚 黑山 阿尔巴尼亚 波斯尼亚和黑塞哥维那 北马其顿 摩尔多瓦 白俄罗斯 安道尔 摩纳哥 列支敦士登 圣马力诺 直布罗陀 根西岛

### 北美洲
美国 加拿大 墨西哥

### 南美洲
巴西 阿根廷

### 大洋洲
澳大利亚 新西兰

## 特性
1. 支持车牌识别+车牌颜色+车辆距离+车道线识别
2. 支持车辆类型+车辆颜色+车头车尾+停车位识别
3. 支持Linux、Win、Centos、Ubuntu、统信UOS、麒麟系统(Kylin)下CPU、GPU部署，支持定制化开发
4. 支持C/C++，python，C#、Java等语言调用
5. 支持Android、ios、鸿蒙系统(HarmonyOS)部署，可获取商用测试demo(微信ID: cbp931126)
   |型号 |速度|精度 |接口|测试系统|图像大小|
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   |红米K30pro(骁龙865)        |~70ms(CPU/4Core) ~40ms(GPU)|       99%+    |kotlin、java|android|1920*1080|
   |红米K60(骁龙8+Gen1)        |~50ms(CPU/3Core) ~30ms(GPU) ~18ms(NPU)|       99%+    |kotlin、java|android|1920*1080|
   |红米K90 MAX(天玑9500)        |~40ms(CPU/4Core) ~28ms(GPU) ~15ms(NPU)|       99%+    |kotlin、java|android|1920*1080|
   |华为Mate80(麒麟9020)        |~40ms(CPU/4Core) ~20ms(NPU)|       99%+    |ArkTS|HarmonyOS6.0|1920*1080|
   |iphone13(A15)       |~50ms(CPU/2Core) ~13ms(NPU)|       99%+    |swift、Objective-C|ios|1920*1080|
   |iPad Air 3(A12)      |~100ms(CPU/2Core) ~35ms(NPU)|       99%+    |swift、Objective-C|ios|1920*1080|
   |UNISOC7885(展锐P7885))      |~60ms/100ms(GPU/CPU)|       99%+   |kotlin、java|android|1920*1080|
6. 支持瑞芯微rv1106/rv1106/rk3588/rk3568/rk3576等侧端部署，可获取商用测试demo(微信ID: cbp931126)
   | 型号 |速度|精度 |接口|测试系统|图像大小|
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   |rk3588           |~25ms|       99%+    |C/C++、python|ubuntu、buildroot、debian、android|1920*1080|
   |rk3576           |~25ms|       99%+    |C/C++、python|ubuntu、buildroot、debian、android|1920*1080|
   |rk3568 	         |~70ms|       99%+    |C/C++、python|ubuntu、buildroot、debian、android|1920*1080|
   |rv1126B 	      |~25ms|       99%+    |  C/C++      |ubuntu、buildroot、debian、android|1920*1080|
   |rv1126 	         |~62ms|       99%+    |  C/C++      |ubuntu、buildroot|1920*1080|
   |rv1106 	         |~150ms|      99%+    |  C/C++      |buildroot|1920*1080|
7. 支持海思系列侧端部署:SS928/Hi3403/Hi3519DV500/Hi3516DV500/Hi3516DV300/Hi3516CV610等，获取商用测试demo(微信ID: cbp931126)
   | 型号 |速度|精度 |接口|测试系统|图像大小|
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   |SS928/Hi3403 |~40ms|       99%+    |C/C++|ubuntu、BusyBox|1920*1080|
   |Hi3519DV500  |~50ms|       99%+    |C/C++|BusyBox|1920*1080|
   |Hi3516CV610  |~150ms|       99%+    |C/C++|BusyBox|1920*1080|
8. 支持算能BM1684系列侧端部署
9. 支持爱芯系列侧端部署:


# 识别效果

## 美国车牌识别
<img src="resources/106.jpg" alt="drawing" width="380"/> <img src="resources/109.jpg" alt="drawing" width="380"/>

## 欧洲车牌识别
<img src="resources/101.jpg" alt="drawing" width="380"/> <img src="resources/123.jpg" alt="drawing" width="380"/>

## 车道线+车距+车牌识别
<img src="resources/21.jpg" alt="drawing" width="380"/> <img src="resources/22.jpg" alt="drawing" width="380"/>
<img src="resources/23.jpg" alt="drawing" width="380"/> <img src="resources/24.jpg" alt="drawing" width="380"/>

## 韩国车牌识别
<img src="resources/1c30874f261703b7b8b20f93fb4ecfa1.jpg" alt="drawing" width="380"/> <img src="resources/438278612b5b53d29b563a0c92ba58c9.jpg" alt="drawing" width="380"/>
<img src="resources/13a9d96babfbad98417379ad59d8de53.jpg" alt="drawing" width="380"/> <img src="resources/5014889dcc17dab2bf77a14a589b3164.jpg" alt="drawing" width="380"/>

## 台湾车牌识别
<img src="resources/taiwan_081294.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_081657.jpg" alt="drawing" width="380"/>
<img src="resources/taiwan_083023.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_083192.jpg" alt="drawing" width="380"/>

## 港澳车牌识别
<img src="resources/2025-02-07_205049_5271.png" alt="drawing" width="380"/> <img src="resources/20250703_001005-1752890606.jpg" alt="drawing" width="380"/>
<img src="resources/20250703_001040-1752890765.jpg" alt="drawing" width="380"/> <img src="resources/20250703_001084-1752890731.jpg" alt="drawing" width="380"/>


## 中国大陆车牌识别
<img src="resources/1.jpg" alt="drawing" width="380"/> <img src="resources/3.jpg" alt="drawing" width="380"/>
<img src="resources/9.jpg" alt="drawing" width="380"/> <img src="resources/10.jpg" alt="drawing" width="380"/>
<img src="PlateDetectionRecognition/test/yolov7_result/37_0315.jpg" alt="drawing" width="380"/> <img src="resources/2.jpg" alt="drawing" width="380"/>
<img src="resources/7.jpg" alt="drawing" width="380"/> <img src="resources/8.jpg" alt="drawing" width="380"/>
<img src="resources/20250207205024_19-1738933131.jpg" alt="drawing" width="380"/> <img src="resources/20250207205058_20-1738933158.jpg" alt="drawing" width="380"/>
<img src="resources/1(313).jpg" alt="drawing" width="380"/> <img src="resources/1(323).jpg" alt="drawing" width="380"/>

# PlateAlgorithm
## **车牌识别算法，支持11种中文车牌类型**
**1.单行蓝牌**
**2.单行黄牌**
**3.新能源车牌**
**4.白色警用车牌**
**5 教练车牌**
**6 武警车牌**
**7 双层黄牌**
**8 双层武警**
**9 使馆车牌**
**10 港澳牌车**
**11 双层农用车牌**

## 说明
1. 车牌检测(yolov5plate,yolov7plate,yolov8playe),车牌校正，车牌识别，车牌检测识别;
   
   | 文件夹 | State    |  说明   |
   |:----------|:----------|:----------|
   |PLateDetection_yolov5                 |Done|           yolov5 车牌检测              |
   |PLateDetection_yolov7 				      |Done|           yolov7 车牌检测              |
   |PLateDetection_yolov8 				      |Doing|          yolov8 车牌检测              |
   |PlateRecognition 				         |Done|           车牌识别                     |
   |PlateDetectionRecognition 				|Done|           车牌检测->车牌校正->车牌识别    |
   
2. 所有模型均使用C++和TensorRT加速推理,yolov7plate的前后处理使用cuda加速,(其他模型加速优化也可参考);
3. 根据不同的显卡型号自动生成对应的engine(如果文件夹下有其他显卡适配engine，则删除engine才能重新生成使用中的显卡对应的engine);
4. PlateDetectionRecognition->test->main.cpp文件中的条件编译测试说明
	| 测试类别 |  enable    |  说明   |
	|:----------|:----------|:----------|
   |yolov5_plate                 |1|           yolov7车牌检测               |
   |yolov7_plate 				      |1|           yolov5 车牌检测              |

5. 车牌识别准确率(测试集数量:7.2w张)
   | 模型 |  size    |  准确率   |速度| 平台|
	|:----------:|:----------:|:----------:|:----------:|:----------:|
   |plate_recognition_color|s|   92.40%|452.480us|RTX3090|
   |plate_recognition_s    |s|   98.90%|452.597us|RTX3090|
   |plate_recognition_m    |m|   99.35%|463.316us|RTX3090|
   |plate_recognition_l    |l|   99.56%|507.082us|RTX3090|



## 算法说明

# 算法接口
```
/** 
 * @brief                  车牌初始化函数
 * @param config           模块配置参数结构体
 * @return                 HZFLAG
 */
void*Initialize(Config*config);

/** 
 * @brief                  车牌检测识别(yolov5)
 * @param img              Plate_ImageData
 * @param PlateDet         车牌检测识别结果列表
 * @return                 HZFLAG
 */		
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/** 
 * @brief                  车牌检测(yolov7_plate)
 * @param img              Plate_ImageData
 * @param PlateDet         车牌检测识别结果列表
 * @return                 HZFLAG
 */		
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets);


/** 
 * @brief                  车牌检测(yolov8_plate)
 * @param img              Plate_ImageData
 * @param PlateDet         车牌检测识别结果列表
 * @return                 HZFLAG
 */		
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/** 
 * @brief                  反初始化
 * @return                 HZFLAG 
 */		
int Release(void*p,Config*config);
```

## 2.环境
1. ubuntu20.04+cuda11.1+cudnn8.2.1+TensorRT8.2.5.1(测试通过)
2. ubuntu18.04+cuda10.2+cudnn8.2.1+TensorRT8.2.5.1(测试通过)
3. Win10+cuda11.1+cudnn8.2.1+TensorRT8.2.5.1      (测试通过)
4. 其他环境请自行尝试或者加群了解


## 3.编译
1. 更改根目录下的CMakeLists.txt,设置tensorrt的安装目录
```
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. 默认opencv已安装，cuda,cudnn已安装
3. 为了Debug默认编译 ```-g O0``` 版本,如果为了加快速度请编译Release版本

4. 使用Visual Studio Code快捷键编译(4,5二选其一):
```
   ctrl+shift+B
```
5. 使用命令行编译(4,5二选其一):
```
   mkdir build
   cd build
   cmake ..
   make -j6
```

# References
1. https://github.com/deepcam-cn/yolov5-face
2. https://github.com/derronqi/yolov7-face/tree/main
3. https://github.com/derronqi/yolov8-face

