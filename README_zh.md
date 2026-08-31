[简体中文](./README.md) | 繁體中文 | [English](./README_en.md) | [한국어](./README_ko.md)

如果覺得有用，不妨給個 Star ⭐️🌟 支持一下吧～謝謝！

# 致謝與聯絡方式
### 1. WeChat ID：cbp931126
1. 加入討論群（備註：PlateAlgorithm），群內有許多高手一起交流；
2. 加 WeChat 可取得整理好的開源及部分私有車牌偵測與辨識資料（資料費用為人民幣 30 元）；

# 網頁測試
中國大陸車牌、香港與澳門車牌、台灣車牌辨識線上體驗：http://vision.zhoujiayao.com/

## 功能特色
1. 支援車牌辨識、車牌顏色、車輛距離及車道線辨識
2. 支援車輛類型、車輛顏色、車頭車尾及停車位辨識
3. 支援在 Linux、Windows、CentOS、Ubuntu、統信 UOS、麒麟系統（Kylin）上使用 CPU、GPU 部署，並支援客製化開發
4. 支援 C/C++、Python、C#、Java 等語言呼叫
5. 支援 Android、iOS、HarmonyOS 部署，可取得商用測試 Demo（WeChat ID：cbp931126）

   | 型號 | 速度 | 準確率 | 介面 | 測試系統 | 圖片大小 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | Redmi K30 Pro（Snapdragon 865） | ~70 ms（CPU/4 核心）、~40 ms（GPU） | 99%+ | Kotlin、Java | Android | 1920*1080 |
   | Redmi K60（Snapdragon 8+ Gen 1） | ~50 ms（CPU/3 核心）、~30 ms（GPU）、~18 ms（NPU） | 99%+ | Kotlin、Java | Android | 1920*1080 |
   | Redmi K90 MAX（Dimensity 9500） | ~40 ms（CPU/4 核心）、~28 ms（GPU）、~15 ms（NPU） | 99%+ | Kotlin、Java | Android | 1920*1080 |
   | HUAWEI Mate 80（Kirin 9020） | ~40 ms（CPU/4 核心）、~20 ms（NPU） | 99%+ | ArkTS | HarmonyOS 6.0 | 1920*1080 |
   | iPhone 13（A15） | ~50 ms（CPU/2 核心）、~13 ms（NPU） | 99%+ | Swift、Objective-C | iOS | 1920*1080 |
   | iPad Air 3（A12） | ~100 ms（CPU/2 核心）、~35 ms（NPU） | 99%+ | Swift、Objective-C | iOS | 1920*1080 |
   | UNISOC P7885 | ~60 ms/100 ms（GPU/CPU） | 99%+ | Kotlin、Java | Android | 1920*1080 |
6. 支援 Rockchip RV1106/RV1126/RK3588/RK3568/RK3576 等邊緣裝置部署，可取得商用測試 Demo（WeChat ID：cbp931126）

   | 型號 | 速度 | 準確率 | 介面 | 測試系統 | 圖片大小 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | RK3588 | ~25 ms | 99%+ | C/C++、Python | Ubuntu、Buildroot、Debian、Android | 1920*1080 |
   | RK3576 | ~25 ms | 99%+ | C/C++、Python | Ubuntu、Buildroot、Debian、Android | 1920*1080 |
   | RK3568 | ~70 ms | 99%+ | C/C++、Python | Ubuntu、Buildroot、Debian、Android | 1920*1080 |
   | RV1126B | ~25 ms | 99%+ | C/C++ | Ubuntu、Buildroot、Debian、Android | 1920*1080 |
   | RV1126 | ~62 ms | 99%+ | C/C++ | Ubuntu、Buildroot | 1920*1080 |
   | RV1106 | ~150 ms | 99%+ | C/C++ | Buildroot | 1920*1080 |
7. 支援 HiSilicon SS928/Hi3403/Hi3519DV500/Hi3516DV500/Hi3516DV300/Hi3516CV610 等邊緣裝置部署，可取得商用測試 Demo（WeChat ID：cbp931126）

   | 型號 | 速度 | 準確率 | 介面 | 測試系統 | 圖片大小 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | SS928/Hi3403 | ~40 ms | 99%+ | C/C++ | Ubuntu、BusyBox | 1920*1080 |
   | Hi3519DV500 | ~50 ms | 99%+ | C/C++ | BusyBox | 1920*1080 |
   | Hi3516CV610 | ~150 ms | 99%+ | C/C++ | BusyBox | 1920*1080 |
8. 支援 Sophgo BM1684 系列邊緣裝置部署
9. 支援 Axera 系列邊緣裝置部署

## 辨識效果

### 車道線 + 車距 + 車牌辨識效果
<img src="resources/21.jpg" alt="drawing" width="380"/> <img src="resources/22.jpg" alt="drawing" width="380"/>
<img src="resources/23.jpg" alt="drawing" width="380"/> <img src="resources/24.jpg" alt="drawing" width="380"/>

### 韓國車牌辨識效果
<img src="resources/1c30874f261703b7b8b20f93fb4ecfa1.jpg" alt="drawing" width="380"/> <img src="resources/438278612b5b53d29b563a0c92ba58c9.jpg" alt="drawing" width="380"/>
<img src="resources/13a9d96babfbad98417379ad59d8de53.jpg" alt="drawing" width="380"/> <img src="resources/5014889dcc17dab2bf77a14a589b3164.jpg" alt="drawing" width="380"/>

### 台灣車牌辨識效果
<img src="resources/taiwan_081294.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_081657.jpg" alt="drawing" width="380"/>
<img src="resources/taiwan_083023.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_083192.jpg" alt="drawing" width="380"/>

### 香港與澳門車牌辨識效果
<img src="resources/2025-02-07_205049_5271.png" alt="drawing" width="380"/> <img src="resources/20250703_001005-1752890606.jpg" alt="drawing" width="380"/>
<img src="resources/20250703_001040-1752890765.jpg" alt="drawing" width="380"/> <img src="resources/20250703_001084-1752890731.jpg" alt="drawing" width="380"/>

### 中國大陸車牌辨識效果
<img src="resources/1.jpg" alt="drawing" width="380"/> <img src="resources/3.jpg" alt="drawing" width="380"/>
<img src="resources/9.jpg" alt="drawing" width="380"/> <img src="resources/10.jpg" alt="drawing" width="380"/>
<img src="PlateDetectionRecognition/test/yolov7_result/37_0315.jpg" alt="drawing" width="380"/> <img src="resources/2.jpg" alt="drawing" width="380"/>
<img src="resources/7.jpg" alt="drawing" width="380"/> <img src="resources/8.jpg" alt="drawing" width="380"/>
<img src="resources/20250207205024_19-1738933131.jpg" alt="drawing" width="380"/> <img src="resources/20250207205058_20-1738933158.jpg" alt="drawing" width="380"/>
<img src="resources/1(313).jpg" alt="drawing" width="380"/> <img src="resources/1(323).jpg" alt="drawing" width="380"/>

# PlateAlgorithm
## **支援 12 種中文車牌類型的車牌辨識演算法**
**1. 單行藍牌**  
**2. 單行黃牌**  
**3. 新能源車牌**  
**4. 白色警用車牌**  
**5. 教練車牌**  
**6. 武警車牌**  
**7. 雙層黃牌**  
**8. 雙層武警車牌**  
**9. 使館車牌**  
**10. 香港與澳門車牌**  
**11. 雙層農用車牌**

## 說明
1. 車牌偵測（yolov5plate、yolov7plate、yolov8plate）、車牌校正、車牌辨識，以及整合式車牌偵測與辨識；

   | 資料夾 | 狀態 | 說明 |
   |:----------|:----------|:----------|
   | PLateDetection_yolov5 | 完成 | YOLOv5 車牌偵測 |
   | PLateDetection_yolov7 | 完成 | YOLOv7 車牌偵測 |
   | PLateDetection_yolov8 | 進行中 | YOLOv8 車牌偵測 |
   | PlateRecognition | 完成 | 車牌辨識 |
   | PlateDetectionRecognition | 完成 | 車牌偵測 -> 車牌校正 -> 車牌辨識 |

2. 所有模型均使用 C++ 和 TensorRT 加速推論。yolov7plate 的前處理與後處理使用 CUDA 加速（其他模型的加速最佳化也可參考）；
3. 依不同顯示卡型號自動產生對應的引擎。若資料夾內有適用於其他顯示卡的引擎，須先刪除，才能重新產生目前顯示卡對應的引擎；
4. `PlateDetectionRecognition/test/main.cpp` 中的條件編譯測試選項：

   | 測試類別 | enable | 說明 |
   |:----------|:----------|:----------|
   | yolov5_plate | 1 | YOLOv7 車牌偵測 |
   | yolov7_plate | 1 | YOLOv5 車牌偵測 |

5. 車牌辨識準確率（測試集數量：72,000 張）

   | 模型 | 大小 | 準確率 | 速度 | 平台 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|
   | plate_recognition_color | s | 92.40% | 452.480 us | RTX 3090 |
   | plate_recognition_s | s | 98.90% | 452.597 us | RTX 3090 |
   | plate_recognition_m | m | 99.35% | 463.316 us | RTX 3090 |
   | plate_recognition_l | l | 99.56% | 507.082 us | RTX 3090 |

## 演算法說明

# 演算法介面
```cpp
/**
 * @brief                  車牌模組初始化函式
 * @param config           模組設定參數結構
 * @return                 HZFLAG
 */
void*Initialize(Config*config);

/**
 * @brief                  車牌偵測與辨識（YOLOv5）
 * @param img              Plate_ImageData
 * @param PlateDet         車牌偵測與辨識結果清單
 * @return                 HZFLAG
 */
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/**
 * @brief                  車牌偵測（yolov7_plate）
 * @param img              Plate_ImageData
 * @param PlateDet         車牌偵測與辨識結果清單
 * @return                 HZFLAG
 */
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/**
 * @brief                  車牌偵測（yolov8_plate）
 * @param img              Plate_ImageData
 * @param PlateDet         車牌偵測與辨識結果清單
 * @return                 HZFLAG
 */
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/**
 * @brief                  解除初始化
 * @return                 HZFLAG
 */
int Release(void*p,Config*config);
```

## 2. 環境
1. Ubuntu 20.04 + CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.5.1（測試通過）
2. Ubuntu 18.04 + CUDA 10.2 + cuDNN 8.2.1 + TensorRT 8.2.5.1（測試通過）
3. Windows 10 + CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.5.1（測試通過）
4. 其他環境請自行嘗試，或加入討論群瞭解詳情

## 3. 編譯
1. 修改專案根目錄下的 `CMakeLists.txt`，設定 TensorRT 安裝目錄：
```cmake
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. 預設已安裝 OpenCV、CUDA 及 cuDNN
3. 為方便除錯，預設使用 `-g O0` 編譯。如需提升速度，請編譯 Release 版本
4. 使用 Visual Studio Code 快速鍵編譯（第 4、5 項擇一）：
```
ctrl+shift+B
```
5. 使用命令列編譯（第 4、5 項擇一）：
```bash
mkdir build
cd build
cmake ..
make -j6
```

# 參考資料
1. https://github.com/deepcam-cn/yolov5-face
2. https://github.com/derronqi/yolov7-face/tree/main
3. https://github.com/derronqi/yolov8-face

## 新增文字識別後端

已增加 `PlateRecognitionOCRV5`、`PlateRecognitionOCRV6`、
`PlateRecognition_LPRNet` 與 `PlateRecognitionFastOCR`，C/C++ 推論統一使用
專案的 TensorRT 8.2.5.1。

- V5 已包含 `models/mobile/inference.onnx` 與 `models/server/inference.onnx`；
  Mobile、Server 各有一份獨立固化的 77 字元映射，呼叫時透過
  `PlateRecognitionOCRV5Config::model_variant` 選擇。
- V6 已包含 `models/tiny/inference.onnx`、`models/small/inference.onnx` 與
  `models/medium/inference.onnx`；tiny 使用一套固化映射，small/medium 共用 full
  固化映射，兩套映射都只包含同一組 77 字元。
- V5/V6 部署不需要外部字元字典，函式庫及 C++ 範例成功時只輸出由
  `PlateRecognition` 允許字元組成的最終字串；LPRNet 維持官網 68 類字元定義。
- FastOCR 已下載官網 Hub 目前全部 10 個 ONNX 與 9 份設定，並提供 10 個經數值
  對齊的 TRT8 部署圖；使用官網 37 類拉丁字元表，預設 `cct-s-v2-global-model`。
- V5/V6 辨識效果迴歸使用 ONNX Runtime CPU 執行 ONNX；LPRNet 的 PT→ONNX
  一致性測試使用 PyTorch 與 ONNX Runtime；FastOCR 的 10 組官方/部署 ONNX 已逐組
  比較數值與字元 argmax。目前 Windows 主機沒有可執行的 Linux TensorRT 環境，
  TRT engine 與 GPU 辨識效果仍須在目標機完成最終驗收。

完整模型路徑、編譯、介面及驗證命令見 [OCR_BACKENDS.md](./OCR_BACKENDS.md)。
