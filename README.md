# Egg Embryo Stage Detection — Enterprise Production System

> **Hệ thống Computer Vision cấp doanh nghiệp** phân loại hình ảnh trứng thành **4 giai đoạn phát triển phôi** trong thời gian thực, sử dụng Transfer Learning với **MobileNetV2** và triển khai inference on-device qua **TensorFlow Lite** trên thiết bị Android.

---

## Mục lục
- [About this Production System](#about-this-production-system)
- [Mô tả hệ thống](#mô-tả-hệ-thống)
- [Kiến trúc giải pháp](#kiến-trúc-giải-pháp)
- [Bài toán & Phương pháp kỹ thuật](#bài-toán--phương-pháp-kỹ-thuật)
- [Thuật toán và mô hình](#thuật-toán-và-mô-hình)
- [Chi tiết kiến trúc mô hình](#chi-tiết-kiến-trúc-mô-hình)
- [Các giai đoạn huấn luyện chi tiết](#các-giai-đoạn-huấn-luyện-chi-tiết)
- [Data Pipeline & Augmentation](#data-pipeline--augmentation)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Triển khai Android (TensorFlow Lite)](#triển-khai-android-tensorflow-lite)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Build & chạy](#build--chạy)
- [License](#license)

---

## About this Production System

Đây là **giải pháp phần mềm cấp production** (enterprise-grade), được thiết kế để tích hợp trực tiếp vào hệ thống tự động hóa của các trang trại ấp trứng công nghiệp, cơ sở sản xuất giống gia cầm và hệ thống kiểm soát chất lượng trong ngành chăn nuôi.

### Thông tin triển khai

| Hạng mục | Chi tiết |
|---|---|
| **Loại hệ thống** | Computer Vision Edge-AI, on-device inference |
| **Mức độ hoàn thiện** | Production-ready (đã tích hợp pipeline huấn luyện, conversion TFLite, Android runtime) |
| **Mô hình triển khai** | Standalone SDK/Module Android, có thể nhúng vào hệ thống lớn hơn |
| **Đối tượng sử dụng** | Kỹ sư nông nghiệp, đơn vị chăn nuôi gia cầm, nhà máy ấp trứng công nghiệp |
| **Yêu cầu hạ tầng** | Điện thoại Android 5.0+ (API 21+) hoặc thiết bị nhúng hỗ trợ TFLite |
| **Quyền runtime** | Camera, không yêu cầu kết nối Internet tại thời điểm suy luận |

### Tính năng sản phẩm

- **Phân loại 4 giai đoạn phôi** trong thời gian thực qua camera (`dau`, `giua`, `cuoi`, `trungchet`).
- **Inference on-device** bằng TensorFlow Lite — bảo mật dữ liệu, không phụ thuộc cloud, phản hồi dưới 100ms mỗi frame.
- **Pipeline huấn luyện tự động** gồm 2 giai đoạn (Feature Extraction → Fine-tuning) tái lập được 100% từ notebook.
- **Multi-flavor build (Task API + Support Library)** cho phép lựa chọn engine TFLite tuỳ ứng dụng.
- **Model nhỏ gọn** (`mobinet.tflite` chỉ 2.7 MB) phù hợp cập nhật OTA và nhúng vào thiết bị IoT.

---

## Tổng quan

| Thành phần | Mô tả |
|---|---|
| **Bài toán** | Multi-class image classification (4 lớp) |
| **Input** | Ảnh trứng chụp từ camera, kích thước `160×160×3` |
| **Output** | Vector xác suất 4 lớp → `argmax` cho kết quả dự đoán |
| **Backbone** | **MobileNetV2** pretrained trên ImageNet (1.4M tham số backbone) |
| **Thuật toán huấn luyện** | Transfer Learning: Feature Extraction → Fine-tuning |
| **Optimizer** | Phase 1: Adam(lr=1e-4) → Phase 2: RMSprop(lr=1e-5) |
| **Epochs** | Phase 1: 100 epochs → Phase 2: 100 epochs (tổng 200) |
| **Loss** | `SparseCategoricalCrossentropy` |
| **Inference** | TensorFlow Lite (`.tflite`, 2.7 MB) trên Android |
| **Nền tảng inference** | Android (Java), CPU/GPU/NNAPI delegate |

---

## Mô tả hệ thống

**Egg Embryo Stage Detection** là hệ thống con (sub-system) trong chuỗi tự động hóa trang trại ấp trứng, thực hiện nhiệm vụ **phân loại giai đoạn phát triển phôi** dựa trên ảnh chụp vỏ trứng. Hệ thống phục vụ 3 mục tiêu nghiệp vụ:

1. **Tự động hoá quy trình kiểm tra lô ấp** — thay thế kiểm tra thủ công, giảm sai sót do con người.
2. **Tối ưu tỷ lệ nở** — phát hiện sớm trứng chết và phôi phát triển không đạt để loại bỏ kịp thời.
3. **Số hoá nhật ký ấp** — log kết quả theo từng quả trứng, phục vụ truy vết và phân tích dữ liệu lớn ở tầng doanh nghiệp.

### Phạm vi chức năng

| # | Chức năng | Mô tả |
|---|---|---|
| 1 | Capture ảnh | Đọc frame liên tục từ Camera2 API |
| 2 | Preprocess | Resize 160×160, normalize theo yêu cầu của MobileNetV2 |
| 3 | Inference | Chạy TFLite Interpreter, thu vector xác suất 4 lớp |
| 4 | Decision | Áp dụng `argmax` để ra quyết định cuối cùng |
| 5 | Visualization | Overlay kết quả + confidence lên previewView của app |

### Yêu cầu phi chức năng

- **Latency**: < 100 ms/frame trên thiết bị Android tầm trung (CPU baseline).
- **Memory**: Model 2.7 MB + runtime interpreter ~30 MB RAM.
- **Reliability**: Hoạt động offline hoàn toàn, không cần kết nối mạng.
- **Maintainability**: Code tách module theo 2 flavor, dễ tích hợp vào codebase Android hiện hữu.

---

## Kiến trúc giải pháp

```
┌────────────────────────────────────────────────────────────────────────┐
│                    HẠ TẦNG TRANG TRẠI / IOT EDGE                       │
│                                                                            │
│  Camera (USB / built-in)                                                 │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────────────────────────────────┐                          │
│  │ Android App / Edge Device                  │                          │
│  │ (TFLite Interpreter 2.7 MB)                │                          │
│  │                                             │                          │
│  │  Camera2 API → Bitmap                      │                          │
│  │       │                                    │                          │
│  │       ▼                                    │                          │
│  │  TFLite Interpreter                        │                          │
│  │  (mobinet.tflite)                          │                          │
│  │       │                                    │                          │
│  │       ▼                                    │                          │
│  │  [P₀, P₁, P₂, P₃]                         │                          │
│  │       │                                    │                          │
│  │       ▼                                    │                          │
│  │  Decision: argmax → Class                  │                          │
│  └─────────────────────────────────────────────┘                          │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────────────────────────────────┐                          │
│  │  SCADA / MES / Farm Management System      │                          │
│  │  (Backend server-side, separate concern)   │                          │
│  └─────────────────────────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────┘
```

Hệ thống hoạt động theo mô hình **edge-AI**: inference thực hiện ngay trên thiết bị, kết quả được đẩy về hệ thống quản lý trang trại (Farm Management System) qua REST API hoặc message queue do lớp tích hợp xử lý. Repo này **chỉ chứa lớp inference + pipeline huấn luyện**, phần tích hợp hạ tầng tuỳ thuộc deployment context.

---

## Bài toán & Phương pháp kỹ thuật

### Bài toán

Cho một ảnh đầu vào của trứng, hệ thống cần phân loại trứng vào **4 lớp** tương ứng với giai đoạn phát triển phôi hoặc trạng thái hỏng:

| Class ID | Label (trong code) | Giai đoạn | Mô tả |
|---|---|---|---|
| 0 | `dau` | **Đầu** (Early) | Phôi trứng ngày 0–7, hình thành ban đầu |
| 1 | `giua` | **Giữa** (Mid) | Phôi trứng ngày 8–14, phát triển mạch máu & cơ quan |
| 2 | `cuoi` | **Cuối** (Late) | Phôi trứng ngày 15–21, hoàn thiện, gần nở |
| 3 | `trungchet` | **Trứng chết** (Dead) | Trứng không phát triển, phôi chết — cần loại bỏ khỏi lô ấp |

> **Lưu ý thứ tự:** Class names được TensorFlow sắp xếp theo **thứ tự alphabet** của tên thư mục (`cuoi`, `dau`, `giua`, `trungchet`). Khi inference, `argmax` trả về index 0–3 tương ứng với `['15-21', '0-7', '8-14', 'trungchet']`.

### Phương pháp: Transfer Learning

Dùng **MobileNetV2** pretrained ImageNet làm backbone và huấn luyện theo 2 giai đoạn:

1. **Phase 1 — Feature Extraction (frozen backbone):** Đóng băng toàn bộ weights của backbone MobileNetV2. Chỉ huấn luyện weights của custom classification head (GlobalAveragePooling → Dense). Giai đoạn này giúp head học cách "hiểu" các features mà backbone đã trích xuất được từ ImageNet.

2. **Phase 2 — Fine-tuning (partial unfreeze):** Mở khóa 54 layer cuối cùng của MobileNetV2 (từ layer 100 trở đi trong tổng số ~154 layer), tiếp tục huấn luyện với learning rate nhỏ hơn 10 lần. Giai đoạn này giúp backbone **tinh chỉnh** các features đặc thù cho bài toán phôi trứng thay vì features tổng quát của ImageNet.

**Tại sao chọn MobileNetV2?**
- **Nhẹ**: ~3.5M tham số (~16 MB), phù hợp cho thiết bị di động và embedded systems.
- **Depthwise Separable Convolutions**: Giảm đáng kể computational cost so với standard convolution mà vẫn giữ được accuracy tốt.
- **Inverted Residuals & Linear Bottlenecks**: Cải thiện representational efficiency.
- **Pretrained trên ImageNet**: Đã học được low-level features (edges, textures, shapes) — transfer tốt sang domain ảnh trứng.

---

## Thuật toán và mô hình

### Tổng quan pipeline thuật toán

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORWARD PASS                                  │
│                                                                     │
│  Input Image (160×160×3, RGB)                                       │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────┐                       │
│  │ Data Augmentation (train only)           │                       │
│  │  → RandomFlip('horizontal')              │                       │
│  │  → RandomRotation(0.2)                   │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │ Preprocessing                            │                       │
│  │  → Resize to (160, 160, 3)               │                       │
│  │  → tf.keras.applications.mobilenet_v2.   │                       │
│  │    preprocess_input: scale pixels to      │                       │
│  │    range [-1, 1] (1./127.5, offset=-1)    │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │ MobileNetV2 Backbone (Frozen/Fine-tune)  │                       │
│  │  input: (160,160,3)                      │                       │
│  │  include_top=False → output: (5,5,1280) │                       │
│  │  154 layers total                        │                       │
│  │  Phase 1: all trainable=False            │                       │
│  │  Phase 2: layers[:100] trainable=False   │                       │
│  │            layers[100:] trainable=True   │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │ GlobalAveragePooling2D                   │                       │
│  │  input: (5,5,1280) → output: (1280,)     │                       │
│  │  Tính trung bình theo spatial dims       │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │ Dropout(0.2)                              │                       │
│  │  Randomly zero 20% activations           │                       │
│  │  Chỉ áp dụng trong training              │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  ┌──────────────────────────────────────────┐                       │
│  │ Dense(4, activation='softmax')           │                       │
│  │  input: (1280,) → output: (4,)           │                       │
│  │  Σ output[i] = 1.0 (probabilities)        │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     ▼                                                │
│  Output: P = softmax(logits) ∈ ℝ⁴, ΣP = 1.0                        │
│          prediction = argmax(P) ∈ {0,1,2,3}                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Công thức toán học

**1. Preprocessing (MobileNetV2):**
```
x_input = x_float / 127.5 - 1
```
Mỗi pixel RGB được normalize từ `[0, 255]` → `[-1, 1]`.

**2. MobileNetV2 backbone:**
MobileNetV2 sử dụng **Depthwise Separable Convolutions** thay cho standard convolution. Một depthwise separable convolution tách phép convolution thành 2 bước:

- **Depthwise Conv**: Áp dụng 1 bộ filter riêng cho mỗi kênh đầu vào
  ```
  Output_DW[k, i, j] = Σ_m Kernel_DW[k, m] × Input[m, i, j]
  ```

- **Pointwise Conv (1×1 Conv)**: Kết hợp các kênh lại
  ```
  Output_PW[k, i, j] = Σ_c Kernel_PW[k, c] × Output_DW[c, i, j]
  ```

Ưu điểm: Computational cost giảm từ `D_K² × M × N × D_K²` xuống `D_K² × M × N + M × N × D_K² × N` (khoảng **8–9 lần** ít hơn).

**3. Inverted Residual Block:**
```
Input (low-channel bottleneck)
  → 1×1 Conv (expand channels, e.g. ×6)
  → 3×3 DW Conv
  → 1×1 Conv (project back to low-channel)
  → Residual + Input
```
Thiết kế **inverted** (expand → compress) khác với ResNet thông thường (compress → expand → compress), giúp tiết kiệm memory.

**4. GlobalAveragePooling2D:**
```
feature_vector[j] = (1 / (H × W)) × Σ_i Σ_k Input[j, i, k]
```
Tính trung bình spatial (5×5) cho mỗi trong 1280 channels → vector 1280 chiều.

**5. Softmax (Dense output):**
```
P(class_k | x) = exp(logit_k) / Σ_j exp(logit_j)
```
Đảm bảo output là xác suất, ΣP = 1.0.

**6. Loss function — Sparse Categorical Crossentropy:**
```
L = - Σ_i log(P[true_class_i])
```
Dùng `Sparse` (labels là integer 0,1,2,3) thay vì one-hot encoding.

**7. Optimizer Phase 1 — Adam:**
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          (first moment)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         (second moment)
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)
```
Với `lr=1e-4`, `β₁=0.9`, `β₂=0.999`, `ε=1e-7`.

**8. Optimizer Phase 2 — RMSprop:**
```
E[g²]_t = 0.9·E[g²]_{t-1} + 0.1·g_t²
θ_t = θ_{t-1} - lr × g_t / √(E[g²]_t + ε)
```
Với `lr=1e-5` (1/10 phase 1). RMSprop thích hợp cho fine-tuning vì decay learning rate tự động per-parameter.

---

## Chi tiết kiến trúc mô hình

### Toàn bộ kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: (None, 160, 160, 3)                        │
│                        (batch, H, W, C)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  INPUT LAYER: tf.keras.Input(160,160,3) │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  DATA AUGMENTATION (train only)        │
              │  Sequential([                          │
              │    RandomFlip('horizontal'),            │
              │    RandomRotation(0.2),  # ~11.5°      │
              │  ])                                    │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  PREPROCESS_INPUT                     │
              │  mobilenet_v2.preprocess_input        │
              │  x = x / 127.5 - 1.0                  │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────────────────────┐
              │  MOBILENETV2 BACKBONE (pretrained ImageNet)          │
              │  input_shape=(160,160,3), include_top=False          │
              │  output_shape: (None, 5, 5, 1280)                    │
              │  ~3.5M params (trainable: Phase1=0, Phase2=~2.4M)    │
              │                                                       │
              │  ┌─ Conv2D(32, 3×3, strides=2) ───────────────────┐    │
              │  │  + BatchNorm + ReLU6                          │    │
              │  └──────────────────────────────────────────────┘    │
              │  ┌─ InvertedResidual × 17 (từ block_1 đến block_15) │
              │  │  Depthwise Separable Conv                       │
              │  │  Expansion factor: [1,6,6,6,...]               │
              │  │  Output channels: [16,24,24,32,32,32,64,64,...]  │
              │  └──────────────────────────────────────────────┘    │
              │  ┌─ InvertedResidual × 4 (block_16)                │    │
              │  │  Strides: [2,1,1,1] trong toàn bộ block_1→16   │    │
              │  └──────────────────────────────────────────────┘    │
              │  ┌─ Conv2D(1280, 1×1) + BatchNorm + ReLU6          │    │
              └──────────────┬───────────────────────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  GLOBALAVERAGEPOOLING2D               │
              │  input: (None, 5, 5, 1280)            │
              │  output: (None, 1280)                  │
              │  Params: 0 (stateless)                │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  DROPOUT(0.2)                         │
              │  20% neurons bị zero-out trong train │
              │  Rate=0 → inference                   │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  DENSE(4, activation='softmax')       │
              │  input: (None, 1280)                  │
              │  output: (None, 4)                    │
              │  Params: 1280×4 + 4 = 5,124           │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │  OUTPUT: (None, 4)                     │
              │  [P_0-7, P_8-14, P_15-21, P_trungchet] │
              │  ΣP = 1.0                              │
              └────────────────────────────────────────┘
```

### Thống kê tham số

| Thành phần | Số params | Trainable (Phase 1) | Trainable (Phase 2) |
|---|---|---|---|
| MobileNetV2 Backbone | ~2,257,984 | **0** (frozen) | ~2,400,000 (54 layers cuối) |
| GlobalAveragePooling2D | 0 | 0 | 0 |
| Dropout | 0 | 0 | 0 |
| Dense(4) | 5,124 | **5,124** | 5,124 |
| **Tổng** | **~2,263,108** | **~5,124** | **~2,405,124** |

**Điểm đáng chú ý:** Phase 1 chỉ huấn luyện **0.23%** tổng params → tránh overfitting hiệu quả, training nhanh.

### Số lượng layers MobileNetV2 (total: 154)

```
Layer 0:       Conv2D + BN + ReLU6
Layer 1-2:     block_1 (expand=1, out=16, stride=1)       ─┐
Layer 3-4:     block_2 (expand=6, out=24, stride=2)        │
Layer 5-6:     block_3 (expand=6, out=24, stride=1)         │ FROZEN
Layer 7-8:     block_4 (expand=6, out=32, stride=2)        │ in Phase 2
Layer 9-10:    block_5 (expand=6, out=32, stride=1)         │
Layer 11-12:   block_6 (expand=6, out=32, stride=1)        ─┤
Layer 13-14:   block_7 (expand=6, out=64, stride=2)         │
Layer 15-16:   block_8 (expand=6, out=64, stride=1)         │
Layer 17-18:   block_9 (expand=6, out=64, stride=1)         │ TRAINABLE
Layer 19-20:   block_10 (expand=6, out=96, stride=2)      │ in Phase 2
Layer 21-22:   block_11 (expand=6, out=96, stride=1)        │
Layer 23-24:   block_12 (expand=6, out=96, stride=1)       │
Layer 25-26:   block_13 (expand=6, out=160, stride=2)        │
Layer 27-28:   block_14 (expand=6, out=160, stride=1)      │
Layer 29-30:   block_15 (expand=6, out=160, stride=1)      ─┤
Layer 31-32:   block_16 (expand=6, out=320, stride=1)      │
Layer 33:      Conv2D(1280) + BN + ReLU6                   ─┘
```

Fine-tune từ layer 100 = bắt đầu từ `block_7` trở đi, tức 54 layers cuối (~1.06M params backbone trong các block này).

---

## Các giai đoạn huấn luyện chi tiết

### Dataset & Split

- **Nguồn**: Ảnh trứng trong `dataset_egg_full/{dau, giua, cuoi, trungchet}/`
- **Tổng ảnh gốc**: 3,955 (dau: 1,025 + giua: 970 + cuoi: 1,035 + trungchet: 925)
- **Split**: `train_test_split(all_files, test_size=0.2, random_state=42)`
  - **Train**: 3,164 ảnh (80%)
  - **Validation**: 791 ảnh (20%)
- **Lưu ý**: Split được thực hiện **trên toàn bộ ảnh** ( không phân chia theo từng class trước) → đảm bảo phân bố train/val nhất quán.

### Hyperparameters chính

| Hyperparameter | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen (trainable=False) | Partial unfreeze (layers[100:]) |
| Optimizer | Adam | RMSprop |
| Learning rate | 1e-4 | 1e-5 (= 1e-4 / 10) |
| Epochs | 100 | 100 (tổng 200) |
| Batch size | 32 | 32 |
| Loss | SparseCategoricalCrossentropy | SparseCategoricalCrossentropy |
| Metrics | accuracy | accuracy |
| Data augmentation | ✓ (RandomFlip + RandomRotation) | ✓ (same) |
| Dropout rate | 0.2 | 0.2 |

### Phase 1 — Feature Extraction (Epochs 0–99)

**Mục tiêu:** Huấn luyện classification head (chỉ ~5K params) để học cách phân loại 4 giai đoạn từ features cố định của MobileNetV2.

```
1. model = tf.keras.Model(inputs, outputs)   # inputs → GAP → Dropout → Dense(4)
2. base_model.trainable = False               # Đóng băng toàn bộ backbone
3. model.compile(
       optimizer=Adam(lr=1e-4),
       loss=SparseCategoricalCrossentropy(),
       metrics=['accuracy']
   )
4. model.fit(train_dataset, epochs=100, validation_data=validation_dataset)
```

**Tại sao phase 1 cần nhiều epochs (100)?**
- Head mới khởi tạo ngẫu nhiên hoàn toàn, cần học từ đầu.
- Backbone frozen → gradients chỉ flow qua head (~5K params) → học chậm.
- 100 epochs đủ để head hội tụ mà không overfitting (do chỉ 5K params).

### Phase 2 — Fine-tuning (Epochs 100–199)

**Mục tiêu:** Tinh chỉnh 54 layers cuối của backbone để features phù hợp hơn với domain ảnh trứng.

```
5. base_model.trainable = True                 # Mở khóa toàn bộ backbone
6. for layer in base_model.layers[:100]:        # Đóng băng 100 layers đầu
       layer.trainable = False
7. model.compile(
       loss=SparseCategoricalCrossentropy(),
       optimizer=RMSprop(lr=1e-5),
       metrics=['accuracy']
   )
8. model.fit(train_dataset,
       epochs=200,
       initial_epoch=100,          # Tiếp tục từ epoch 100
       validation_data=validation_dataset
   )
```

**Tại sao dùng RMSprop với lr nhỏ hơn?**
- Unfreeze đột ngột 54 layers với lr lớn → catastrophic forgetting (weights backbone ImageNet bị phá hủy).
- RMSprop với `lr=1e-5` cập nhật weights nhỏ hơn, ổn định hơn Adam.
- 100 epochs fine-tuning đủ để tinh chỉnh mà không overfit.

### Huấn luyện trên Google Colab

Notebook sử dụng đường dẫn Google Drive:
```python
data_dir = '/content/drive/MyDrive/dataset_egg_full'
model_save_path = '/content/drive/MyDrive/h5kl/mobinet.h5'
```

Trên máy local, thay đổi các đường dẫn tương ứng.

### Lưu & Convert model

```python
# 1. Lưu Keras model (.h5)
model.save('/content/drive/MyDrive/h5kl/mobinet.h5')  # 16.9 MB

# 2. Convert sang TensorFlow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('mobinet.tflite', 'wb').write(tflite_model)       # 2.7 MB (~6× nhỏ hơn)
```

---

## Data Pipeline & Augmentation

### Data Pipeline

```
dataset_egg_full/           image_dataset_from_directory
      │                          │
      ▼                          ▼
  all_image_files ───train_test_split──▶ train_files (80%)
      │                                    │
      │                                    ▼
      │                           copy → train/{dau,giua,cuoi,trungchet}/
      │                                    │
      │                                    ▼
      │                           tf.keras.utils.image_dataset_from_directory
      │                                    │
      ▼                                    ▼
  validation_files (20%)           train_dataset (batch=32, shuffle=True)
      │                                    │
      ▼                                    ▼
  copy → validation/{...}/        train_dataset.prefetch(AUTOTUNE)
      │                                    │
      ▼                                    ▼
  tf.keras.utils...          ───────────────────┐
  validation_dataset (batch=32)                  │
      │                                         ▼
      ▼                                  Data Augmentation
  validation_dataset.prefetch(AUTOTUNE)    (RandomFlip + RandomRotation)
      │                                         │
      └────────────────────────────────────────┘
                        │
                        ▼
               model.fit(train_dataset,
                          validation_data=validation_dataset)
```

### Data Augmentation

Chỉ áp dụng cho **train_dataset** (không áp dụng cho validation/test).

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),    # Lật ngang ngẫu nhiên
    tf.keras.layers.RandomRotation(0.2),          # Xoay ±0.2 × 2π rad ≈ ±36° (mạnh)
])
```

**Tại sao cần augmentation?**
- Ảnh trứng có thể được chụp ở nhiều góc độ khác nhau → cần augmentation để model **không overfit** vào một hướng cố định.
- `RandomRotation(0.2)` tương đương `±36°` — đủ lớn để tạo biến đổi thực tế mà không vượt quá giới hạn sinh lý của hình dạng trứng.
- Không dùng `RandomZoom` hoặc `RandomContrast` vì có thể thay đổi đặc điểm quan trọng của phôi.

---

## Kết quả đánh giá

### Đầu ra inference

```python
class_names = ['15-21', '0-7', '8-14', 'trungchet']   # argmax index → label

predictions = loaded_model.predict(image_array)
# predictions.shape = (1, 4)
# predictions[0] = [P_15-21, P_0-7, P_8-14, P_trungchet]

predicted_class_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_index]
# Ví dụ: predictions = [0.05, 0.01, 0.02, 0.92]
#         → argmax = 3 → 'trungchet'
```

### Các file model

| File | Định dạng | Kích thước | Mục đích |
|---|---|---|---|
| `mobinet.h5` | Keras H5 | **16.9 MB** | Huấn luyện, đánh giá, experiment |
| `mobinet.tflite` | TensorFlow Lite | **2.7 MB** | Inference trên Android (default) |
| `mobinet_with_detection.h5` | Keras H5 | **9.4 MB** | Model biến thể (có detection head) |

---

## Triển khai Android (TensorFlow Lite)

### TensorFlow Lite là gì?

**TensorFlow Lite (TFLite)** là bộ công cụ của Google cho phép deploy các mô hình TensorFlow lên **thiết bị di động, embedded, và IoT**. TFLite có 2 thành phần chính:

1. **TFLite Converter**: Chuyển `.h5` / `.pb` → `.tflite`
2. **TFLite Runtime**: Interpreter nhẹ chạy inference trên thiết bị đích

**Ưu điểm:**
- Kích thước nhỏ (runtime chỉ ~1–2 MB)
- Inference nhanh trên mobile hardware
- Hỗ trợ **NNAPI Delegate** (GPU/NPU trên Android) và **GPU Delegate**
- Không cần Python hay network connection

### Hai pipeline inference trong project

Project cung cấp **2 cách** để chạy inference trên Android, chọn qua `productFlavors`:

#### Pipeline A — TensorFlow Lite Task Library (`lib_task_api`) — Mặc định

Sử dụng high-level API từ `org.tensorflow.lite.task.vision`, gói gọn toàn bộ quy trình:
```java
// Khởi tạo classifier
ImageClassifier.ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .setMaxResults(4)
        .build();

// Đọc model từ assets
ImageClassifier classifier = ImageClassifier.createFromFileAndOptions(
    context, "mobinet.tflite", options);

// Phân loại frame từ camera
ImageProxy imageProxy = ...;
TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
List<ClassificationResult> results = classifier.classify(tensorImage);
```

**Classifiers có sẵn trong Task Library:**
- `ClassifierFloatMobileNet` — MobileNetV2 float32
- `ClassifierQuantizedMobileNet` — MobileNetV2 quantized (INT8)
- `ClassifierFloatEfficientNet`
- `ClassifierQuantizedEfficientNet`

#### Pipeline B — TensorFlow Lite Support Library (`lib_support`)

Sử dụng low-level API, cho phép tuỳ biến sâu hơn:
```java
// Khởi tạo TFLite Interpreter thủ công
tflite = new Interpreter(loadModelFile(activity, "mobinet.tflite"));

// Map input/output tensors
inputImageBuffer = new TensorImage(DataType.FLOAT32);
outputBuffer = new TensorBuffer.createFixedSize(
    new int[]{1, 4}, DataType.FLOAT32);

// Chạy inference
tflite.run(imageData, outputBuffer.getBuffer().rewind());

// Post-process: argmax
float[] probabilities = outputBuffer.getFloatArray();
int predictedClass = argmax(probabilities);
```

### Android App Architecture

```
Camera2 API (CameraActivity)
      │
      ▼
  Frame Buffer (YUV_420_888 / JPEG)
      │
      ▼
  Bitmap → TensorImage (160×160)
      │
      ▼
  TFLite Interpreter
  (mobinet.tflite in assets/)
      │
      ▼
  Output: [P_0-7, P_8-14, P_15-21, P_trungchet]
      │
      ▼
  OverlayView / ResultsView
  (hiển thị top-1 + confidence lên màn hình)
```

### Chọn pipeline inference

Trong `android/app/build.gradle`:

```gradle
flavorDimensions "tfliteInference"
productFlavors {
    support  { dimension "tfliteInference"; /* dùng lib_support */ }
    taskApi   { dimension "tfliteInference"; getIsDefault().set(true) }  // Mặc định
}
```

Build variant `taskApiDebug` sử dụng TFLite Task Library. Build variant `supportDebug` sử dụng TFLite Support Library.

---

## Cấu trúc dự án

```
.
├── android/                              # Ứng dụng Android
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/.../classification/
│   │   │   │   ├── ClassifierActivity.java    # Main activity
│   │   │   │   ├── CameraActivity.java         # Camera2 + inference
│   │   │   │   ├── customview/                # AutoFitTextureView, OverlayView
│   │   │   │   ├── env/                       # Logger, ImageUtils, BorderedText
│   │   │   │   └── Classifier.java            # Base classifier
│   │   │   ├── res/layout/
│   │   │   ├── AndroidManifest.xml
│   │   │   └── assets/
│   │   │       └── mobinet.tflite             # Model TFLite
│   │   ├── build.gradle                       # productFlavors: taskApi | support
│   │   └── proguard-rules.pro
│   ├── lib_support/                          # TFLite Support Library pipeline
│   │   └── java/.../support/                 # ClassifierFloatMobileNet, ...
│   ├── lib_task_api/                         # TFLite Task Library pipeline
│   │   └── java/.../task_api/               # ClassifierFloatMobileNet, ...
│   ├── models/
│   │   └── src/main/assets/
│   │       ├── mobinet.tflite
│   │       └── labels.txt
│   ├── gradle/wrapper/
│   ├── build.gradle                          # Top-level: AGP 8.3.1
│   └── settings.gradle
│
├── file code model/
│   └── egg_tranferlearning_mobinet_v2.ipynb   # Toàn bộ pipeline huấn luyện
│
├── dataset_egg_full/                        # Dataset 3,955 ảnh gốc
│   ├── dau/         (1025)    # Giai đoạn 0-7 ngày
│   ├── giua/        (970)     # Giai đoạn 8-14 ngày
│   ├── cuoi/        (1035)    # Giai đoạn 15-21 ngày
│   ├── trungchet/   (925)     # Trứng chết
│   ├── train/       (3164)    # Tập train (80%)
│   └── validation/  (791)     # Tập validation (20%)
│
├── mobinet.h5                               # Keras model (16.9 MB)
├── mobinet.tflite                           # TFLite model (2.7 MB)
├── mobinet_with_detection.h5                # Keras variant (9.4 MB)
└── README.md
```

---

## Build & chạy

### Huấn luyện (Python / Colab)

```bash
pip install tensorflow numpy scikit-learn matplotlib
jupyter notebook "file code model/egg_tranferlearning_mobinet_v2.ipynb"
```

Chỉnh `data_dir` trong notebook → chạy tuần tự từ cell 1 → 37.

### Convert model

```python
import tensorflow as tf
model = tf.keras.models.load_model("mobinet.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mobinet.tflite", "wb").write(tflite_model)
```

Copy `mobinet.tflite` vào `android/models/src/main/assets/`.

### Build Android

```bash
cd android

# TFLite Task Library (mặc định)
./gradlew assembleTaskApiDebug          # Linux/macOS
.\gradlew.bat assembleTaskApiDebug      # Windows

# TFLite Support Library
./gradlew assembleSupportDebug
```

Mở Android Studio → `File > Open > android/` → `Run > Run app` trên thiết bị thật.

---

## License

Dự án dựa trên [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) (Apache 2.0).
