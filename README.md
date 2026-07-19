# Image Classification - Detect Phôi Trứng

Đồ án tốt nghiệp: **Phân loại hình ảnh phát hiện trứng có phôi / không có phôi** sử dụng Transfer Learning (MobileNetV2) và triển khai trên thiết bị Android với TensorFlow Lite.

## Cấu trúc dự án

```
.
├── android/                     # Ứng dụng Android (TensorFlow Lite Image Classification)
│   ├── app/                     # Module chính của app
│   ├── lib_support/            # Pipeline inference dùng TFLite Support Library
│   ├── lib_task_api/           # Pipeline inference dùng TFLite Task Library
│   ├── models/                  # Module build model TFLite
│   ├── gradle/                  # Gradle wrapper
│   ├── build.gradle             # Top-level Gradle config
│   └── settings.gradle
│
├── file code model/             # Jupyter Notebook huấn luyện model
│   └── egg_tranferlearning_mobinet_v2.ipynb
│
└── DATASET-TRUNG/               # Dataset ảnh trứng
    ├── CO PHOI/                 # Ảnh trứng có phôi
    └── KHONG CO PHOI/           # Ảnh trứng không có phôi
```

## 1. Huấn luyện mô hình

Mở file `file code model/egg_tranferlearning_mobinet_v2.ipynb` bằng Jupyter Notebook / Google Colab để:

- Load dataset từ `DATASET-TRUNG/`
- Áp dụng **Transfer Learning** với backbone **MobileNetV2**
- Train, đánh giá và xuất model `.tflite` để chạy trên Android

## 2. Ứng dụng Android

App được phát triển dựa trên [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

### Yêu cầu
- Android Studio 3.2 trở lên
- Thiết bị Android đã bật chế độ nhà phát triển & USB debugging
- Cáp USB

### Build & chạy

```bash
cd android
./gradlew build           # Linux/macOS
.\gradlew.bat build       # Windows
```

Sau đó vào Android Studio, mở thư mục `android/` và chọn `Run -> Run app`.

### Hai kiểu pipeline inference
- **lib_task_api**: dùng [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier) (khuyến nghị, dễ dùng)
- **lib_support**: dùng [TensorFlow Lite Support Library](https://www.tensorflow.org/lite/inference_with_metadata/lite_support) (tuỳ biến sâu hơn)

Có thể đổi giữa hai kiểu bằng cách chỉnh `flavorDimensions "tfliteInference"` trong `app/build.gradle`.

## Ghi chú

- Dataset ảnh gốc và file model lớn đã được loại trừ khỏi git (xem `.gitignore`). Bạn cần tự thêm dataset và copy file `.tflite` vào `android/app/src/main/assets/` trước khi build app thật.
- File `.gitignore` đã cấu hình để bỏ qua: `.gradle/`, `build/`, `.idea/`, file dataset (`*.jpg`, `*.png`, ...), file model (`*.tflite`, `*.h5`, ...).
