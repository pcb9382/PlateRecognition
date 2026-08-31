[简体中文](./README.md) | [繁體中文](./README_zh.md) | [English](./README_en.md) | 한국어

이 프로젝트가 유용하다면 Star ⭐️🌟로 응원해 주세요. 감사합니다!

# 감사의 말 및 연락처
### 1. WeChat ID: cbp931126
1. 토론 그룹에 참여하려면 친구 요청에 `PlateAlgorithm`이라고 적어 주세요. 많은 숙련된 개발자가 활발하게 교류하고 있습니다.
2. WeChat 친구로 추가하면 정리된 오픈 소스 및 일부 비공개 번호판 검출·인식 데이터셋을 받을 수 있습니다(데이터 비용 30위안).

# 웹 테스트
중국 본토, 홍콩·마카오 및 대만 번호판 인식을 온라인으로 체험할 수 있습니다: http://vision.zhoujiayao.com/

## 주요 기능
1. 번호판 인식, 번호판 색상, 차량 거리 및 차선 인식 지원
2. 차량 유형, 차량 색상, 차량 전면·후면 및 주차 공간 인식 지원
3. Linux, Windows, CentOS, Ubuntu, UnionTech UOS, Kylin에서 CPU 및 GPU 배포를 지원하며 맞춤형 개발 가능
4. C/C++, Python, C#, Java 등의 언어용 API 지원
5. Android, iOS 및 HarmonyOS 배포 지원. 상용 테스트 데모 제공 가능(WeChat ID: cbp931126)
   | 모델 | 속도 | 정확도 | API | 테스트 OS | 이미지 크기 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | Redmi K30 Pro (Snapdragon 865) | ~70 ms (CPU/4코어), ~40 ms (GPU) | 99%+ | Kotlin, Java | Android | 1920*1080 |
   | Redmi K60 (Snapdragon 8+ Gen 1) | ~50 ms (CPU/3코어), ~30 ms (GPU), ~18 ms (NPU) | 99%+ | Kotlin, Java | Android | 1920*1080 |
   | Redmi K90 MAX (Dimensity 9500) | ~40 ms (CPU/4코어), ~28 ms (GPU), ~15 ms (NPU) | 99%+ | Kotlin, Java | Android | 1920*1080 |
   | HUAWEI Mate 80 (Kirin 9020) | ~40 ms (CPU/4코어), ~20 ms (NPU) | 99%+ | ArkTS | HarmonyOS 6.0 | 1920*1080 |
   | iPhone 13 (A15) | ~50 ms (CPU/2코어), ~13 ms (NPU) | 99%+ | Swift, Objective-C | iOS | 1920*1080 |
   | iPad Air 3 (A12) | ~100 ms (CPU/2코어), ~35 ms (NPU) | 99%+ | Swift, Objective-C | iOS | 1920*1080 |
   | UNISOC P7885 | ~60 ms/100 ms (GPU/CPU) | 99%+ | Kotlin, Java | Android | 1920*1080 |
6. Rockchip RV1106/RV1126/RK3588/RK3568/RK3576 등의 엣지 배포 지원. 상용 테스트 데모 제공 가능(WeChat ID: cbp931126)
   | 모델 | 속도 | 정확도 | API | 테스트 OS | 이미지 크기 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | RK3588 | ~25 ms | 99%+ | C/C++, Python | Ubuntu, Buildroot, Debian, Android | 1920*1080 |
   | RK3576 | ~25 ms | 99%+ | C/C++, Python | Ubuntu, Buildroot, Debian, Android | 1920*1080 |
   | RK3568 | ~70 ms | 99%+ | C/C++, Python | Ubuntu, Buildroot, Debian, Android | 1920*1080 |
   | RV1126B | ~25 ms | 99%+ | C/C++ | Ubuntu, Buildroot, Debian, Android | 1920*1080 |
   | RV1126 | ~62 ms | 99%+ | C/C++ | Ubuntu, Buildroot | 1920*1080 |
   | RV1106 | ~150 ms | 99%+ | C/C++ | Buildroot | 1920*1080 |
7. HiSilicon SS928/Hi3403/Hi3519DV500/Hi3516DV500/Hi3516DV300/Hi3516CV610 등의 엣지 배포 지원. 상용 테스트 데모 제공 가능(WeChat ID: cbp931126)
   | 모델 | 속도 | 정확도 | API | 테스트 OS | 이미지 크기 |
   |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
   | SS928/Hi3403 | ~40 ms | 99%+ | C/C++ | Ubuntu, BusyBox | 1920*1080 |
   | Hi3519DV500 | ~50 ms | 99%+ | C/C++ | BusyBox | 1920*1080 |
   | Hi3516CV610 | ~150 ms | 99%+ | C/C++ | BusyBox | 1920*1080 |
8. Sophgo BM1684 시리즈 엣지 배포 지원
9. Axera 시리즈 엣지 배포 지원


## 인식 결과

### 미국 번호판 인식
<img src="resources/106.jpg" alt="drawing" width="380"/> <img src="resources/109.jpg" alt="drawing" width="380"/>

### 유럽연합 번호판 인식
<img src="resources/101.jpg" alt="drawing" width="380"/> <img src="resources/123.jpg" alt="drawing" width="380"/>

### 차선 + 차량 거리 + 번호판 인식
<img src="resources/21.jpg" alt="drawing" width="380"/> <img src="resources/22.jpg" alt="drawing" width="380"/>
<img src="resources/23.jpg" alt="drawing" width="380"/> <img src="resources/24.jpg" alt="drawing" width="380"/>

### 한국 번호판 인식
<img src="resources/1c30874f261703b7b8b20f93fb4ecfa1.jpg" alt="drawing" width="380"/> <img src="resources/438278612b5b53d29b563a0c92ba58c9.jpg" alt="drawing" width="380"/>
<img src="resources/13a9d96babfbad98417379ad59d8de53.jpg" alt="drawing" width="380"/> <img src="resources/5014889dcc17dab2bf77a14a589b3164.jpg" alt="drawing" width="380"/>

### 대만 번호판 인식
<img src="resources/taiwan_081294.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_081657.jpg" alt="drawing" width="380"/>
<img src="resources/taiwan_083023.jpg" alt="drawing" width="380"/> <img src="resources/taiwan_083192.jpg" alt="drawing" width="380"/>

### 홍콩·마카오 번호판 인식
<img src="resources/2025-02-07_205049_5271.png" alt="drawing" width="380"/> <img src="resources/20250703_001005-1752890606.jpg" alt="drawing" width="380"/>
<img src="resources/20250703_001040-1752890765.jpg" alt="drawing" width="380"/> <img src="resources/20250703_001084-1752890731.jpg" alt="drawing" width="380"/>



### 중국 본토 번호판 인식
<img src="resources/1.jpg" alt="drawing" width="380"/> <img src="resources/3.jpg" alt="drawing" width="380"/>
<img src="resources/9.jpg" alt="drawing" width="380"/> <img src="resources/10.jpg" alt="drawing" width="380"/>
<img src="PlateDetectionRecognition/test/yolov7_result/37_0315.jpg" alt="drawing" width="380"/> <img src="resources/2.jpg" alt="drawing" width="380"/>
<img src="resources/7.jpg" alt="drawing" width="380"/> <img src="resources/8.jpg" alt="drawing" width="380"/>
<img src="resources/20250207205024_19-1738933131.jpg" alt="drawing" width="380"/> <img src="resources/20250207205058_20-1738933158.jpg" alt="drawing" width="380"/>
<img src="resources/1(313).jpg" alt="drawing" width="380"/> <img src="resources/1(323).jpg" alt="drawing" width="380"/>

# PlateAlgorithm
## **11가지 중국 번호판 유형을 지원하는 번호판 인식 알고리즘**
**1. 한 줄 파란색 번호판**
**2. 한 줄 노란색 번호판**
**3. 신에너지 차량 번호판**
**4. 흰색 경찰 차량 번호판**
**5 운전 교습 차량 번호판**
**6 무장경찰 차량 번호판**
**7 두 줄 노란색 번호판**
**8 두 줄 무장경찰 차량 번호판**
**9 대사관 차량 번호판**
**10 홍콩·마카오 차량 번호판**
**11 두 줄 농업용 차량 번호판**

## 설명
1. 번호판 검출(yolov5plate, yolov7plate, yolov8plate), 번호판 보정, 번호판 인식, 통합 번호판 검출 및 인식

   | 폴더 | 상태 | 설명 |
   |:----------|:----------|:----------|
   |PLateDetection_yolov5|완료|YOLOv5 번호판 검출|
   |PLateDetection_yolov7|완료|YOLOv7 번호판 검출|
   |PLateDetection_yolov8|진행 중|YOLOv8 번호판 검출|
   |PlateRecognition|완료|번호판 인식|
   |PlateDetectionRecognition|완료|번호판 검출 -> 번호판 보정 -> 번호판 인식|

2. 모든 모델은 C++와 TensorRT를 사용해 추론을 가속합니다. yolov7plate의 전처리와 후처리는 CUDA로 가속합니다(다른 모델의 가속 최적화에도 참고 가능).
3. GPU 모델에 맞는 엔진을 자동으로 생성합니다. 폴더에 다른 GPU용 엔진이 있으면 해당 엔진을 삭제해야 현재 GPU용 엔진이 새로 생성됩니다.
4. `PlateDetectionRecognition/test/main.cpp`의 조건부 컴파일 테스트 옵션
	| 테스트 유형 | enable | 설명 |
	|:----------|:----------|:----------|
   |yolov5_plate|1|YOLOv7 번호판 검출|
   |yolov7_plate|1|YOLOv5 번호판 검출|

5. 번호판 인식 정확도(테스트 세트: 72,000장)
   | 모델 | 크기 | 정확도 | 속도 | 플랫폼 |
	|:----------:|:----------:|:----------:|:----------:|:----------:|
   |plate_recognition_color|s|92.40%|452.480us|RTX3090|
   |plate_recognition_s|s|98.90%|452.597us|RTX3090|
   |plate_recognition_m|m|99.35%|463.316us|RTX3090|
   |plate_recognition_l|l|99.56%|507.082us|RTX3090|



## 알고리즘 설명

# 알고리즘 API
```
/**
 * @brief                  번호판 모듈 초기화 함수
 * @param config           모듈 설정 매개변수 구조체
 * @return                 HZFLAG
 */
void*Initialize(Config*config);

/**
 * @brief                  번호판 검출 및 인식(YOLOv5)
 * @param img              Plate_ImageData
 * @param PlateDet         번호판 검출 및 인식 결과 목록
 * @return                 HZFLAG
 */
int PlateRecognition_yolov5(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/**
 * @brief                  번호판 검출(yolov7_plate)
 * @param img              Plate_ImageData
 * @param PlateDet         번호판 검출 및 인식 결과 목록
 * @return                 HZFLAG
 */
int PlateRecognition_yolov7(void*p,Plate_ImageData*img,PlateDet*PlateDets);


/**
 * @brief                  번호판 검출(yolov8_plate)
 * @param img              Plate_ImageData
 * @param PlateDet         번호판 검출 및 인식 결과 목록
 * @return                 HZFLAG
 */
int PlateRecognition_yolov8(void*p,Plate_ImageData*img,PlateDet*PlateDets);

/**
 * @brief                  모듈 해제
 * @return                 HZFLAG
 */
int Release(void*p,Config*config);
```

## 2. 환경
1. Ubuntu 20.04 + CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.5.1(테스트 완료)
2. Ubuntu 18.04 + CUDA 10.2 + cuDNN 8.2.1 + TensorRT 8.2.5.1(테스트 완료)
3. Windows 10 + CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.5.1(테스트 완료)
4. 기타 환경은 직접 시도하거나 토론 그룹에 참여하여 문의해 주세요


## 3. 빌드
1. 프로젝트 루트의 `CMakeLists.txt`를 수정하여 TensorRT 설치 경로를 설정합니다.
```
set(TensorRT_INCLUDE "/xxx/xxx/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/xxx/xxx/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")
```
2. OpenCV, CUDA 및 cuDNN이 설치되어 있다고 가정합니다
3. 디버깅을 위해 기본적으로 ```-g O0``` 옵션으로 빌드합니다. 성능을 높이려면 Release 구성으로 빌드하세요

4. Visual Studio Code 단축키로 빌드합니다(4번과 5번 중 하나 선택).
```
   ctrl+shift+B
```
5. 명령줄에서 빌드합니다(4번과 5번 중 하나 선택).
```
   mkdir build
   cd build
   cmake ..
   make -j6
```

# 참고 자료
1. https://github.com/deepcam-cn/yolov5-face
2. https://github.com/derronqi/yolov7-face/tree/main
3. https://github.com/derronqi/yolov8-face

## 새 텍스트 인식 백엔드

`PlateRecognitionOCRV5`, `PlateRecognitionOCRV6`,
`PlateRecognition_LPRNet`, `PlateRecognitionFastOCR`가 추가되었습니다. C/C++ 추론은
모두 프로젝트의 TensorRT 8.2.5.1을 사용합니다.

- V5에는 `models/mobile/inference.onnx`와 `models/server/inference.onnx`가 포함되어 있습니다.
  Mobile과 Server에는 각각 고정된 77자 매핑이 있으며, 호출할 때
  `PlateRecognitionOCRV5Config::model_variant`로 선택합니다.
- V6에는 `models/tiny/inference.onnx`, `models/small/inference.onnx`,
  `models/medium/inference.onnx`가 포함되어 있습니다. tiny는 별도의 고정 매핑을 사용하고,
  small/medium은 full 고정 매핑을 공유하며, 두 매핑은 모두 동일한 77자 집합만 포함합니다.
- V5/V6 배포에는 외부 문자 사전이 필요하지 않습니다. 라이브러리와 C++ 데모는 성공 시
  `PlateRecognition`에서 허용하는 문자로만 구성된 최종 문자열을 출력합니다. LPRNet은
  공식 68개 클래스 문자 정의를 그대로 유지합니다.
- FastOCR에는 공식 Hub에서 제공하는 현재 전체 모델인 ONNX 10개와 설정 파일 9개가
  다운로드되어 있으며, 수치 정합성을 맞춘 TensorRT 8 배포 그래프 10개도 제공합니다.
  공식 37개 라틴 문자 집합을 사용하며 기본 모델은 `cct-s-v2-global-model`입니다.
- V5/V6 인식 결과 회귀 테스트는 ONNX Runtime CPU에서 ONNX를 실행합니다. LPRNet의
  PyTorch→ONNX 일관성 테스트는 PyTorch와 ONNX Runtime을 사용하며, FastOCR의 공식/배포
  ONNX 10개 쌍은 수치와 문자 argmax를 각각 비교했습니다. 현재 Windows 호스트에는 실행
  가능한 Linux TensorRT 환경이 없으므로 TensorRT 엔진과 GPU 인식 결과는 대상 장비에서
  최종 검증해야 합니다.

전체 모델 경로, 빌드 방법, API 및 검증 명령은 [OCR_BACKENDS.md](./OCR_BACKENDS.md)를 참조하세요.
