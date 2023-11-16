简体中文 | [English](./README_en.md)

如果觉得有用，不妨给个Star⭐️🌟支持一下吧~ 谢谢！

# Acknowledgments & Contact 
### 1.WeChat ID: cbp931126
加我微信#(备注：FaceAlgorithm),拉进群
### 2.QQ Group：517671804


# PlateAlgorithm
## 特性
1. 车牌检测(yolov5plate,yolov7plate,yolov8playe),车牌校正，车牌识别;
2. 所有模型均使用C++和TensorRT加速推理,yolov7plate的前后处理使用cuda加速,(其他模型加速优化也可参考);
3. 所有模型使用C++和OnnxRuntime.OpenVINO,NCNN加速推理(TO DO);
4. 提供C接口，可以直接移植在项目里;
5. 提供python调用，C#调用的demo(TODO)
6. 根据不同的显卡型号自动生成对应的engine(如果文件夹下有其他显卡适配engine，则删除engine才能重新生成使用中的显卡对应的engien);
7. 条件编译测试说明
	| 测试类别 |  enable    |  说明   |
	|:----------|:----------|:----------|
   |yolov5_plate                 |1|           yolov7车牌检测               |
   |yolov7_plate 				      |1|           yolov5 车牌检测              |

## 算法说明

# 算法接口
```
```
# 使用方法
       
## 2.环境
1. ubuntu20.04+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
2. ubuntu18.04+cuda10.2+cudnn8.2.1+TrnsorRT8.2.5.1(测试通过)
3. Win10+cuda11.1+cudnn8.2.1+TrnsorRT8.2.5.1      (测试通过)
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
3. https://github.com/we0091234/yolov7-face-tensorrt
4. https://github.com/derronqi/yolov8-face   
