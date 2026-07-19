# Image Classification – Detect Phôi Trứng

Hệ thống phân loại hình ảnh trứng thành hai lớp **Có phôi** / **Không có phôi** bằng kỹ thuật Transfer Learning với backbone **MobileNetV2**, đi kèm ứng dụng Android chạy suy luận trực tiếp trên thiết bị nhờ **TensorFlow Lite**.

## Mục lục
- [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Pipeline huấn luyện](#pipeline-huấn-luyện)
- [Pipeline Android (TensorFlow Lite)](#pipeline-android-tensorflow-lite)
- [Build & chạy](#build--chạy)
- [Ghi chú](#ghi-chú)

## Tổng quan kiến trúc

```
  Dataset ảnh trứng (.jpg / .png)
            │
            ▼
  ┌────────────────────────────┐
  │  Jupyter Notebook (Python) │  ← Tiền xử lý + Augmentation
  │  TensorFlow / Keras        │
  │  MobileNetV2 (pretrained)  │  ← Transfer Learning
  │  GlobalAvgPool → Dropout   │
  │  Dense(softmax, num_class) │  ← Xuất .tflite
  └──────────────┬─────────────┘
                 │  mobinet.tflite
                 ▼
  ┌────────────────────────────┐
  │  Android App (Java)        │
  │  Camera2 API               │  ← Camera realtime
  │  TensorFlow Lite Runtime   │  ← Inference trên CPU/GPU/NNAPI
  │  lib_task_api / lib_support│  ← Hai pipeline inference
  └────────────────────────────┘
```

## Công nghệ sử dụng

### 1. Huấn luyện mô hình (Python)

| Thành phần | Công nghệ / Phiên bản |
|---|---|
| Ngôn ngữ | Python 3.x |
| Framework Deep Learning | **TensorFlow 2.x** + **Keras** |
| Backbone | **MobileNetV2** pretrained trên ImageNet (`tf.keras.applications.MobileNetV2`) |
| Preprocess input | `tf.keras.applications.mobilenet_v2.preprocess_input` (scale `1./127.5, offset=-1`) |
| Input size | `160 × 160 × 3` |
| Kiến trúc head | `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(num_class, activation='softmax')` |
| Loss | `SparseCategoricalCrossentropy` |
| Optimizer | `Adam` |
| Chia dữ liệu | `sklearn.model_selection.train_test_split` |
| Data pipeline | `tf.keras.utils.image_dataset_from_directory`, `tf.data.AUTOTUNE` |
| Data augmentation | `tf.keras.layers.RandomFlip('horizontal')`, `RandomRotation(0.2)` |
| Môi trường chạy | Jupyter Notebook / Google Colab |
| Trực quan hoá | `matplotlib.pyplot`, `tf.keras.utils.plot_model` |
| Đầu ra | File `.tflite` (`mobinet.tflite`, `mobinetfull.tflite`) |

### 2. Ứng dụng Android

| Thành phần | Công nghệ / Phiên bản |
|---|---|
| Ngôn ngữ | Java |
| Android Gradle Plugin | 8.3.1 |
| `compileSdkVersion` | 28 |
| `minSdkVersion` | 21 (Android 5.0) |
| `targetSdkVersion` | 28 |
| Java compatibility | source/target 1.8 |
| UI | AndroidX AppCompat, Material Components, CoordinatorLayout |
| Camera | Camera2 API (custom `CameraConnectionFragment`, `LegacyCameraConnectionFragment`) |
| TensorFlow Lite | Bundled `.tflite` model trong `assets/`, không nén (`aaptOptions { noCompress "tflite" }`) |
| Pipeline A (mặc định) | **TensorFlow Lite Task Library** – `lib_task_api` (`ClassifierFloatMobileNet`, `ClassifierFloatEfficientNet`, `ClassifierQuantizedMobileNet`, `ClassifierQuantizedEfficientNet`) |
| Pipeline B | **TensorFlow Lite Support Library** – `lib_support` (`ClassifierFloatMobileNet`, `ClassifierQuantizedMobileNet`) |
| Build flavor | `flavorDimensions "tfliteInference"` với 2 flavor: `taskApi` (mặc định) và `support` |
| Test | AndroidX Test (JUnit, Truth), `androidTest` với asset test `fox.jpg` |
| Gradle wrapper | `gradlew` / `gradlew.bat` |
| Tool download phụ thuộc | `de.undercouch:gradle-download-task:4.0.2` |

### 3. Công cụ phát triển khác
- **Git** – quản lý mã nguồn
- **Android Studio** – IDE chính cho phần Android
- **Jupyter / Colab** – môi trường train model
- **PowerShell** – thao tác dòng lệnh trên Windows

## Cấu trúc dự án

```
.
├── android/                          # Ứng dụng Android
│   ├── app/                          # Module chính
│   │   ├── src/main/java/.../        # CameraActivity, ClassifierActivity, customview, env utils
│   │   ├── src/main/res/             # layouts, drawable, mipmap
│   │   ├── src/androidTest/          # Instrumented test (fox.jpg + ClassifierTest)
│   │   └── build.gradle              # productFlavors: taskApi | support
│   ├── lib_support/                  # Inference pipeline dùng TFLite Support Library
│   ├── lib_task_api/                 # Inference pipeline dùng TFLite Task Library
│   ├── models/
│   │   └── src/main/assets/          # labels.txt, mobinet.tflite, mobinetfull.tflite
│   ├── gradle/wrapper/               # gradle-wrapper.jar + .properties
│   ├── build.gradle                  # AGP 8.3.1, jcenter + google
│   ├── settings.gradle
│   └── gradlew / gradlew.bat
│
├── file code model/
│   └── egg_tranferlearning_mobinet_v2.ipynb   # Notebook train MobileNetV2
│
└── DATASET-TRUNG/
    ├── CO PHOI/                      # Ảnh trứng có phôi
    └── KHONG CO PHOI/                # Ảnh trứng không có phôi
```

## Pipeline huấn luyện

1. Mở `file code model/egg_tranferlearning_mobinet_v2.ipynb` (Jupyter / Colab).
2. `image_dataset_from_directory` đọc `DATASET-TRUNG/CO PHOI/` và `DATASET-TRUNG/KHONG CO PHOI/`.
3. `train_test_split` chia tập train/validation.
4. Augmentation: `RandomFlip('horizontal')` + `RandomRotation(0.2)`.
5. Load backbone `MobileNetV2(input_shape=(160,160,3))` pretrained ImageNet, đóng băng (`base_model.trainable = False`).
6. Thêm head: `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(num_class, softmax)`.
7. Compile với `Adam` + `SparseCategoricalCrossentropy`, train và đánh giá.
8. Convert sang `.tflite`, copy vào `android/models/src/main/assets/`.

## Pipeline Android (TensorFlow Lite)

- App mở camera sau qua **Camera2 API**, đọc frame liên tục.
- Frame được resize về đúng kích thước input của model (`160×160`).
- Model `.tflite` chạy qua hai lựa chọn:
  - **`taskApi`** (mặc định) – dùng `org.tensorflow.lite.task.vision` API cao cấp, code ngắn gọn.
  - **`support`** – dùng `org.tensorflow.lite.support` để tuỳ biến pipeline (preprocess, postprocess, custom op).
- Đổi pipeline bằng cách chọn Build Variant trong Android Studio, hoặc flag `flavorDimensions "tfliteInference"` trong `app/build.gradle`.
- Kết quả top-K classification hiển thị realtime trên overlay.

## Build & chạy

### Huấn luyện

```bash
pip install tensorflow numpy scikit-learn matplotlib
jupyter notebook "file code model/egg_tranferlearning_mobinet_v2.ipynb"
```

### Android app

Yêu cầu:
- Android Studio
- Thiết bị Android thật (đã bật Developer options + USB debugging) hoặc máy ảo
- File `mobinet.tflite` đặt trong `android/models/src/main/assets/` (đã có sẵn trong repo)

```bash
cd android
./gradlew assembleTaskApiDebug     # Linux/macOS – dùng TFLite Task Library
.\gradlew.bat assembleTaskApiDebug # Windows
# hoặc
./gradlew assembleSupportDebug     # dùng TFLite Support Library
```

Sau đó mở Android Studio → `Run > Run app`.

## Ghi chú

- `.gitignore` loại trừ `.gradle/`, `build/`, `.idea/`, dataset ảnh (`*.jpg`, `*.png`, ...) và model lớn (`*.h5`, `*.pt`, `*.onnx`) để giữ repo gọn nhẹ. File `.tflite` trong `android/models/src/main/assets/` được commit vì đã là đầu ra cuối của model.
- Repo dựa trên [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android), được điều chỉnh để chạy model MobileNetV2 nhận diện phôi trứng.
