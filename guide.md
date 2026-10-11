# HƯỚNG DẪN SỬ DỤNG PIPELINE & QUY CHUẨN ĐẶT TÊN DỮ LIỆU
**GreenSM Vehicle Detection Project (Active HITL Lab)**

Tài liệu này chuẩn hóa quy tắc đặt tên file ngắn gọn, phục vụ trực tiếp cho việc **chia tập Test Set chống rò rỉ dữ liệu (Anti-Leakage Group Split)** và vận hành pipeline CVAT.

---

## 🏷️ 1. QUY CHUẨN ĐẶT TÊN FILE CHUẨN DỰ ÁN (NEW NAMING CONVENTION)

Hệ thống tự động đặt tên theo 2 định dạng ngắn gọn, chuẩn hóa, **hoàn toàn không có dấu cách**:

### 🔹 Dạng 1: Ảnh trích xuất từ Video (`videos/`)
```text
vid_<random_id_video>_<frame_stt>.jpg
```
- **`<random_id_video>`**: Mã định danh 4 ký tự duy nhất đại diện cho từng video (ví dụ `v8a1`, `v509`).
- **`<frame_stt>`**: Số thứ tự frame theo thời gian 6 chữ số (`000001`, `000002`...).
- **Ý nghĩa**: Toàn bộ frame trích xuất từ cùng một video sẽ mang cùng một mã `vid_<id>`. Khi chia tập dữ liệu ở Bước 3, chỉ cần lọc theo mã `vid_<id>` là có thể **bốc nguyên cả cụm video vào Fixed Test Set hoặc Train Set**, chống tuyệt đối hiện tượng rò rỉ không gian - thời gian (Data Leakage).
- **Ví dụ**:
  - Video A: `vid_v8a1_000001.jpg`, `vid_v8a1_000002.jpg`...
  - Video B: `vid_v509_000001.jpg`, `vid_v509_000002.jpg`...

---

### 🔹 Dạng 2: Ảnh từ thư mục ảnh rời (`images/`)
```text
img_<random_8_ky_tu>.jpg
```
- **`<random_8_ky_tu>`**: Mã băm ngẫu nhiên 8 ký tự hex (ví dụ `a4f89c2e`, `7b21d90f`).
- **Ý nghĩa**: Tên file cực kỳ ngắn gọn, không dấu cách, không ký tự đặc biệt, đảm bảo tính duy nhất và bảo mật.
- **Ví dụ**:
  - `img_a4f89c2e.jpg`
  - `img_7b21d90f.jpg`

---

## 📁 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ToolTrackVideo/
├── data/
│   ├── inputs/
│   │   ├── videos/   <-- Thả file video (.mp4, .avi,...) vào đây
│   │   └── images/   <-- Thả file ảnh thô vào đây
│   └── outputs/
│       ├── deduplicated_images/  (Ảnh đã lọc trùng, đặt tên theo chuẩn img_... / vid_...)
│       ├── pre_labels/           (Nhãn mồi YOLO .txt khớp 100% tên ảnh)
│       ├── cvat_uploads/         (File nén CVAT chia theo Part và Native XML)
│       └── data_profile.json     (Metadata profiling chi tiết: night, blur, small, occlusion, density)
├── fast_dedup_profiling.py       (Script chạy chính)
├── package_for_cvat.py           (Công cụ đóng gói ZIP CVAT)
└── guide.md
```

---

## 🚀 3. THUẬT TOÁN ĐANG CHẠY TRONG PIPELINE

Hệ thống đang chạy **chính xác 100%** giải pháp được đề xuất:
**`Proposed: pHash + Keep-Most-Vehicles (HD < 5) — Perceptual Hashing with Object-Density Selection`**

1. **Giai đoạn 1 — pHash Clustering (Lọc tinh)**:
   - Dùng Perceptual Hashing (pHash) tính mã băm 64-bit cho từng frame.
   - So sánh trượt với ngưỡng Hamming Distance $HD < 5$ (`--hd_thresh 5`) để gom nhóm các frame trùng bối cảnh / góc quay tĩnh.
2. **Giai đoạn 2 — Object-Density Selection (Keep-Most-Vehicles)**:
   - Trong mỗi cụm trùng nhau ($HD < 5$), chạy mô hình **YOLO11n COCO** đếm tổng số phương tiện (`car`, `motorcycle`, `bus`, `truck`).
   - Ưu tiên chọn giữ lại **01 frame duy nhất có mật độ phương tiện cao nhất** (`max vehicle count`), loại bỏ các frame trùng còn lại.
   - Nếu hòa số xe, ưu tiên frame có độ sắc nét (Laplacian variance) cao hơn.
3. **Giai đoạn 3 — Zero-Cost Re-use**:
   - Tái sử dụng luôn kết quả inference của YOLO11n từ bước chọn lọc trên để làm Data Profiling và sinh file nhãn mồi `.txt` cho CVAT (không chạy lại YOLO lần 2).

Mở Terminal tại thư mục `ToolTrackVideo` và chạy:

```bash
python fast_dedup_profiling.py
```

> Nếu máy có GPU NVIDIA, thêm `--device 0` để tăng tốc độ xử lý:
> ```bash
> python fast_dedup_profiling.py --device 0
> ```

---

## 📤 4. UPLOAD LÊN CVAT (NHẬN 100% NHÃN MỒI)

1. Mở Task trên CVAT.
2. Vào **Actions** ➜ **Upload annotations** ➜ Chọn định dạng: **`CVAT 1.1`** (hoặc **`CVAT for images 1.1`**).
3. Tải file ZIP Native XML từ thư mục `data/outputs/cvat_uploads/`:
   - Nếu Task chứa toàn bộ ảnh: tải file `cvat_NATIVE_XML_ALL_...zip`.
   - Nếu Task chia theo từng Part: tải file `cvat_NATIVE_XML_part01_...zip`.
