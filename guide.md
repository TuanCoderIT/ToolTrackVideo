# HƯỚNG DẪN SỬ DỤNG PIPELINE & KIẾN TRÚC MÃ NGUỒN MODULAR
**GreenSM Vehicle Detection Project (Active HITL Lab)**

Tài liệu này chuẩn hóa quy tắc đặt tên file, mô tả cấu trúc module hóa mã nguồn (`src/`) và hướng dẫn vận hành pipeline lọc trùng frame (**Intra-Video FastDedup**), trích xuất metadata (**Automated Data Profiling**) và đóng gói nhãn chuẩn CVAT.

---

## 🏛️ 1. KIẾN TRÚC MÃ NGUỒN MODULAR (`src/`)

Toàn bộ hệ thống được chia thành các module độc lập theo nguyên lý **Single Responsibility Principle (SRP)**:

```text
ToolTrackVideo/
├── src/
│   ├── __init__.py
│   ├── config.py           # 1. Quản lý Taxonomy 7 class CVAT, các ngưỡng pHash, HSV, IoA
│   ├── utils.py            # 2. Tiện ích đặt tên chuẩn (vid_/img_), tính IoA
│   ├── streamers.py        # 3. Đọc dữ liệu dạng Generator (1 FPS Video, EXIF ảnh)
│   ├── dedup.py            # 4. GIAI ĐOẠN 1: pHash + Keep-Most-Vehicles (Object-Density Selection)
│   ├── profiling.py        # 5. GIAI ĐOẠN 2: Trích xuất Metadata & Pre-label (trừ greensm để gán tay)
│   └── cvat_packager.py    # 6. Đóng gói CVAT (Native XML 7-class + Split ZIP batches)
├── run_pipeline.py         # File CLI điều phối chính (Orchestrator)
├── fast_dedup_profiling.py # Wrapper tương thích ngược (trỏ vào run_pipeline.py)
├── package_for_cvat.py     # Wrapper tương thích ngược (trỏ vào src.cvat_packager)
├── data/
│   ├── inputs/             # Thư mục nạp data: videos/ và images/
│   └── outputs/            # Thư mục kết quả: deduplicated_images/, pre_labels/, cvat_uploads/, data_profile.json
└── guide.md
```

---

## 🏷️ 2. QUY CHUẨN ĐẶT TÊN FILE (PROJECT NAMING STANDARD)

### 🔹 Dạng 1: Ảnh trích xuất từ Video (`videos/`)
```text
vid_<random_id_video>_<frame_stt>.jpg
```
- **Ví dụ**: `vid_v8a1_000001.jpg`, `vid_v8a1_000002.jpg`...
- **Ý nghĩa**: Toàn bộ frame trích xuất từ cùng một video sẽ mang cùng một mã `vid_<id>`. Khi chia tập dữ liệu ở Bước 3, chỉ cần lọc theo mã `vid_<id>` là có thể **bốc nguyên cả cụm video vào Fixed Test Set hoặc Train Set**, chống tuyệt đối hiện tượng rò rỉ không gian - thời gian (Anti-Leakage Group Split).

### 🔹 Dạng 2: Ảnh từ thư mục ảnh rời (`images/`)
```text
img_<random_8_ky_tu>.jpg
```
- **Ví dụ**: `img_a4f89c2e.jpg`, `img_7b21d90f.jpg`
- **Ý nghĩa**: Ngắn gọn, không dấu cách, duy nhất và bảo mật.

---

## 🚀 3. THUẬT TOÁN ĐANG CHẠY TRONG PIPELINE

Hệ thống đang chạy **chính xác 100%** giải pháp được đề xuất:
**`Proposed: pHash + Keep-Most-Vehicles (HD < 5) — Perceptual Hashing with Object-Density Selection`**

1. **Giai đoạn 1 — pHash Clustering (Lọc tinh)** (`src/dedup.py`):
   - Tính Perceptual Hash (pHash) 64-bit cho từng frame.
   - So sánh trượt với ngưỡng Hamming Distance $HD < 5$ (`--hd_thresh 5`) để gom nhóm các frame trùng bối cảnh / góc quay tĩnh.
2. **Giai đoạn 2 — Object-Density Selection (Keep-Most-Vehicles)** (`src/dedup.py`):
   - Trong mỗi cụm trùng nhau ($HD < 5$), chạy mô hình **YOLO11n COCO** đếm tổng số đối tượng giao thông (`person`, `bike`, `car`, `motorcycle`, `bus`, `truck`).
   - Ưu tiên chọn giữ lại **01 frame duy nhất có mật độ đối tượng cao nhất** (`max vehicle count`), loại bỏ các frame trùng còn lại.
   - Nếu hòa số xe, ưu tiên frame có độ sắc nét (Laplacian variance) cao hơn.
3. **Giai đoạn 3 — Automated Profiling & Pre-labeling** (`src/profiling.py`):
   - Tái sử dụng luôn kết quả inference của YOLO11n từ bước chọn lọc trên để làm Data Profiling và sinh file nhãn mồi `.txt` cho CVAT (không chạy lại YOLO lần 2).
   - **Cơ chế Pre-labeling**: Tự động gán nhãn cho 6 lớp (`car`, `motorcycle`, `bike`, `person`, `bus`, `truck`). Riêng nhãn `greensm` được giữ nguyên để annotator gán nhãn thủ công trên CVAT.

---

## 💻 4. HƯỚNG DẪN CHẠY DÒNG LỆNH (CLI)

### Chạy toàn bộ Pipeline:
```bash
python run_pipeline.py
```
*(Hoặc `python fast_dedup_profiling.py` vẫn hoạt động bình thường)*.

> Nếu máy có GPU NVIDIA, thêm `--device 0` để tăng tốc độ xử lý:
> ```bash
> python run_pipeline.py --device 0
> ```

### Chạy riêng công cụ nén CVAT khi cần:
```bash
python package_for_cvat.py --batch_size 50
```

---

## 📤 5. UPLOAD LÊN CVAT (NHẬN 100% NHÃN MỒI)

1. Mở Task trên CVAT.
2. Vào **Actions** ➜ **Upload annotations** ➜ Chọn định dạng: **`CVAT 1.1`** (hoặc **`CVAT for images 1.1`**).
3. Tải file ZIP Native XML từ thư mục `data/outputs/cvat_uploads/`:
   - Nếu Task chứa toàn bộ ảnh: tải file `cvat_NATIVE_XML_ALL_...zip`.
   - Nếu Task chia theo từng Part: tải file `cvat_NATIVE_XML_part01_...zip`.
