# 🥚 Egg Embryo Stage Detection

> Phân loại ảnh trứng thành **4 giai đoạn phát triển phôi** bằng Transfer Learning (MobileNetV2) và triển khai suy luận thời gian thực trên Android với TensorFlow Lite.

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Các giai đoạn được phân loại](#các-giai-đoạn-được-phân-loại)
- [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Pipeline huấn luyện](#pipeline-huấn-luyện)
- [Pipeline Android (TensorFlow Lite)](#pipeline-android-tensorflow-lite)
- [Build & chạy](#build--chạy)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Giới thiệu

Dự án xây dựng hệ thống **phát hiện giai đoạn phát triển phôi trong trứng** dựa trên ảnh chụp. Mô hình deep learning phân loại mỗi ảnh đầu vào vào **4 lớp** tương ứng với 4 giai đoạn ấp hoặc tình trạng hỏng của trứng. Kết quả được đưa vào ứng dụng Android chạy inference thời gian thực qua camera.

**Đặc điểm nổi bật:**
- Sử dụng **Transfer Learning** từ `MobileNetV2` (pretrained ImageNet) — gọn nhẹ, phù hợp thiết bị di động.
- Huấn luyện hai giai đoạn: **feature extraction** (frozen backbone) → **fine-tuning** (unfreeze các layer cuối).
- Model được chuyển sang định dạng **TensorFlow Lite (`.tflite`)** để suy luận trực tiếp trên Android, không cần kết nối mạng.

## Các giai đoạn được phân loại

| ID | Tên lớp | Giai đoạn | Mô tả |
|---|---|---|---|
| 0 | `0-7`   | **Đầu** (Early)  | Phôi trứng ngày 0–7  | Giai đoạn hình thành phôi sớm |
| 1 | `8-14`  | **Giữa** (Mid)   | Phôi trứng ngày 8–14 | Giai đoạn phát triển mạch máu, cơ quan |
| 2 | `15-21` | **Cuối** (Late)  | Phôi trứng ngày 15–21 | Giai đoạn hoàn thiện, gần nở |
| 3 | `trungchet` | **Trứng chết** (Dead) | Trứng không phát triển, hỏng | Cần loại bỏ khỏi lô ấp |

**Phân bố dữ liệu** (tổng 3955 ảnh gốc trong `dataset_egg_full/`):

| Lớp | Số ảnh gốc | Train (80%) | Validation (20%) |
|---|---|---|---|
| `dau` (0-7)   | 1025 | ~820 | ~205 |
| `giua` (8-14) | 970  | ~776 | ~194 |
| `cuoi` (15-21)| 1035 | ~828 | ~207 |
| `trungchet`   | 925  | ~740 | ~185 |
| **Tổng**      | **3955** | **3164** | **791** |

## Tổng quan kiến trúc

```
            Dataset ảnh trứng (4 lớp)
                       │
                       ▼
  ┌──────────────────────────────────────────┐
  │   Bước 1: Tiền xử lý & Augmentation     │
  │   • Resize 160×160                       │
  │   • RandomFlip('horizontal')             │
  │   • RandomRotation(0.2)                  │
  │   • train_test_split(80/20)              │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │   Bước 2: Transfer Learning             │
  │   Backbone: MobileNetV2 (ImageNet)       │
  │   • Phase 1: Frozen backbone, Adam       │
  │     lr=1e-4, 100 epochs                  │
  │   • Phase 2: Unfreeze từ layer 100,      │
  │     RMSprop lr=1e-5, +100 epochs         │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │   Bước 3: Xuất model                    │
  │   • mobinet.h5     (Keras, 16.9 MB)      │
  │   • mobinet.tflite (TFLite, 2.7 MB)      │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │   Bước 4: Suy luận trên Android          │
  │   Camera2 → Image 160×160                │
  │   → TFLite Interpreter                   │
  │   → Top-1 class + confidence             │
  └──────────────────────────────────────────┘
```

## Công nghệ sử dụng

### 1. Huấn luyện mô hình (Python — Jupyter Notebook)

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.x |
| Framework Deep Learning | **TensorFlow 2.x** + **Keras** |
| Backbone | **MobileNetV2** pretrained ImageNet (`tf.keras.applications.MobileNetV2`) |
| Input shape | `IMG_SHAPE = (160, 160, 3)` |
| Preprocess | `tf.keras.applications.mobilenet_v2.preprocess_input` |
| Rescaling layer | `Rescaling(1./127.5, offset=-1)` |
| Data Augmentation | `RandomFlip('horizontal')` + `RandomRotation(0.2)` |
| Custom Head | `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(4, activation='softmax')` |
| Loss function | `SparseCategoricalCrossentropy` |
| Phase 1 optimizer | `Adam(learning_rate=1e-4)` (frozen base) |
| Phase 2 optimizer | `RMSprop(learning_rate=1e-5)` (unfreeze từ layer 100) |
| Epochs | 100 (phase 1) + 100 (phase 2) = **200 tổng** |
| Batch size | 32 |
| Dataset API | `tf.keras.utils.image_dataset_from_directory` + `tf.data.AUTOTUNE` |
| Chia dữ liệu | `sklearn.model_selection.train_test_split(test_size=0.2, random_state=42)` |
| Tiện ích file | `os`, `shutil` |
| Trực quan hoá | `matplotlib.pyplot`, `tf.keras.utils.plot_model` |
| Đầu ra | `.h5` (Keras) + `.tflite` (TFLite) |
| Môi trường | Google Colab / Jupyter Notebook |

### 2. Ứng dụng Android (Java)

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Java |
| Android Gradle Plugin | `com.android.tools.build:gradle:8.3.1` |
| `compileSdkVersion` | 28 |
| `minSdkVersion` | 21 (Android 5.0) |
| `targetSdkVersion` | 28 |
| Java compatibility | source/target 1.8 |
| UI Framework | AndroidX AppCompat, Material Components, CoordinatorLayout |
| Camera | **Camera2 API** (`CameraActivity`, `CameraConnectionFragment`, `LegacyCameraConnectionFragment`) |
| TensorFlow Lite | Bundled `.tflite` trong `assets/`, không nén (`aaptOptions { noCompress "tflite" }`) |
| Pipeline A (mặc định) | **TFLite Task Library** — `lib_task_api` |
| Pipeline B | **TFLite Support Library** — `lib_support` |
| Build flavors | `flavorDimensions "tfliteInference"` → 2 flavor: `taskApi` (mặc định), `support` |
| Custom views | `AutoFitTextureView`, `OverlayView`, `RecognitionScoreView`, `ResultsView` |
| Image utils | `ImageUtils`, `BorderedText`, `Logger` |
| Testing | AndroidX Test (JUnit, Truth), instrumented test với `fox.jpg` |
| Gradle plugin phụ | `de.undercouch:gradle-download-task:4.0.2` |

### 3. Công cụ phát triển
- **Git** — quản lý mã nguồn, GitHub Actions-friendly
- **Android Studio** — IDE chính cho phần Android
- **Jupyter / Google Colab** — môi trường train model
- **PowerShell** — CLI trên Windows

## Cấu trúc dự án

```
.
├── android/                              # Ứng dụng Android (TensorFlow Lite)
│   ├── app/                              # Module chính
│   │   ├── src/main/
│   │   │   ├── java/.../classification/  # CameraActivity, ClassifierActivity, customview, env
│   │   │   ├── res/                      # layouts, drawable, mipmap, values
│   │   │   └── AndroidManifest.xml
│   │   ├── src/androidTest/              # Instrumented test (fox.jpg + ClassifierTest.java)
│   │   ├── build.gradle                  # productFlavors: taskApi | support
│   │   └── proguard-rules.pro
│   ├── lib_support/                      # Inference pipeline dùng TFLite Support Library
│   ├── lib_task_api/                     # Inference pipeline dùng TFLite Task Library
│   ├── models/
│   │   ├── src/main/assets/
│   │   │   ├── labels.txt
│   │   │   ├── mobinet.tflite            # Model đã convert, dùng cho inference
│   │   │   └── mobinetfull.tflite
│   │   └── build.gradle
│   ├── gradle/wrapper/                   # gradle-wrapper.jar + .properties
│   ├── build.gradle                      # Top-level: AGP 8.3.1
│   ├── settings.gradle
│   ├── gradlew / gradlew.bat
│   └── README.md
│
├── file code model/
│   └── egg_tranferlearning_mobinet_v2.ipynb   # Notebook train MobileNetV2 (đầy đủ pipeline)
│
├── dataset_egg_full/                     # Dataset ảnh gốc (3955 ảnh)
│   ├── dau/                              # 1025 ảnh — phôi giai đoạn đầu (0-7 ngày)
│   ├── giua/                             # 970 ảnh — phôi giai đoạn giữa (8-14 ngày)
│   ├── cuoi/                             # 1035 ảnh — phôi giai đoạn cuối (15-21 ngày)
│   ├── trungchet/                        # 925 ảnh — trứng hỏng / chết phôi
│   ├── train/                            # Tập train (80%) — tự sinh từ notebook
│   │   ├── dau/
│   │   ├── giua/
│   │   ├── cuoi/
│   │   └── trungchet/
│   └── validation/                       # Tập validation (20%) — tự sinh từ notebook
│       ├── dau/
│       ├── giua/
│       ├── cuoi/
│       └── trungchet/
│
├── mobinet.h5                            # Model Keras (16.9 MB)
├── mobinet.tflite                        # Model TFLite (2.7 MB) — dùng trong Android app
├── mobinet_with_detection.h5             # Model Keras biến thể (9.4 MB)
├── .gitignore
└── README.md
```

## Pipeline huấn luyện

Mở `file code model/egg_tranferlearning_mobinet_v2.ipynb` trên Jupyter Notebook hoặc Google Colab, rồi chạy tuần tự:

1. **Load & chia dữ liệu** — Duyệt `dataset_egg_full/{dau,giua,cuoi,trungchet}/`, lấy tất cả file `.jpg`, chia 80/20 bằng `train_test_split(random_state=42)` và copy vào `dataset_egg_full/train/` & `dataset_egg_full/validation/`.

2. **Tạo `tf.data` pipeline** — `image_dataset_from_directory` với `IMG_SIZE=(160,160)`, `BATCH_SIZE=32`, shuffle + prefetch (`AUTOTUNE`).

3. **Augmentation** — `Sequential([RandomFlip('horizontal'), RandomRotation(0.2)])`.

4. **Khởi tạo backbone** — `MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')`, đặt `base_model.trainable = False` cho phase 1.

5. **Xây head** — `GlobalAveragePooling2D → Dropout(0.2) → Dense(4, softmax)`.

6. **Compile & train phase 1** — `Adam(lr=1e-4)` + `SparseCategoricalCrossentropy`, `initial_epochs=100`.

7. **Fine-tune phase 2** — `base_model.trainable = True`, đóng băng lại các layer đầu (`for layer in base_model.layers[:100]: layer.trainable = False`), compile lại với `RMSprop(lr=1e-5)`, train thêm 100 epochs.

8. **Đánh giá trên test set** — `model.evaluate(test_dataset)`.

9. **Lưu model** — `model.save('mobinet.h5')`, sau đó convert sang `.tflite` qua TFLite Converter và copy vào `android/models/src/main/assets/mobinet.tflite`.

**Class labels cuối cùng (predict):** `['15-21', '0-7', '8-14', 'trungchet']` (thứ tự argmax).

## Pipeline Android (TensorFlow Lite)

- App mở **camera sau** bằng `Camera2 API`, đọc frame liên tục.
- Frame được resize về `(160, 160, 3)` và feed vào TFLite Interpreter.
- Model suy luận ra vector xác suất 4 lớp, hiển thị top-1 lên overlay.

**Chọn pipeline inference** trong `android/app/build.gradle`:

```gradle
flavorDimensions "tfliteInference"
productFlavors {
    support  { dimension "tfliteInference" }                 // TFLite Support Library
    taskApi  { dimension "tfliteInference"; getIsDefault().set(true) } // TFLite Task Library (mặc định)
}
```

## Build & chạy

### 1. Huấn luyện (Python)

```bash
pip install tensorflow numpy scikit-learn matplotlib
jupyter notebook "file code model/egg_tranferlearning_mobinet_v2.ipynb"
```

Trong notebook, chỉnh đường dẫn `data_dir` đến thư mục chứa `dataset_egg_full/`, chạy tuần tự các cell.

### 2. Convert sang TFLite

```python
import tensorflow as tf
model = tf.keras.models.load_model("mobinet.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mobinet.tflite", "wb").write(tflite_model)
```

Copy `mobinet.tflite` vào `android/models/src/main/assets/`.

### 3. Build Android

Yêu cầu: Android Studio, thiết bị Android thật (Developer options + USB debugging) hoặc emulator.

```bash
cd android
./gradlew assembleTaskApiDebug     # dùng TFLite Task Library (mặc định)
./gradlew assembleSupportDebug     # dùng TFLite Support Library
.\gradlew.bat assembleTaskApiDebug # trên Windows
```

Mở Android Studio → `File > Open > android/` → chọn Build Variant → `Run > Run app`.

## Troubleshooting

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| App crash khi load model | File `.tflite` thiếu / sai đường dẫn | Kiểm tra `android/models/src/main/assets/mobinet.tflite` |
| `Method X is undefined for type Interpreter` | API TFLite đổi | Đồng bộ version TFLite trong `lib_support` / `lib_task_api` |
| Không tìm thấy class khi predict | Sai thứ tự `class_names` | Đảm bảo `class_names = ['15-21', '0-7', '8-14', 'trungchet']` khớp `argmax` |
| Overfitting | Augmentation chưa đủ | Tăng rotation/zoom, hoặc thêm `RandomZoom`, `RandomContrast` |
| Model quá lớn | `.h5` chứa cả optimizer state | Chỉ ship file `.tflite` đã quantize lên app |

## License

Dự án dựa trên [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) (Apache 2.0). Phần huấn luyện và dữ liệu phôi trứng là phần mở rộng độc lập.
