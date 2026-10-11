# HƯỚNG DẪN SỬ DỤNG PIPELINE (STAGE 1 & STAGE 2)
**GreenSM Vehicle Detection Project**

Tài liệu này hướng dẫn chi tiết cách đưa dữ liệu đầu vào, chạy tự động pipeline lọc trùng frame (**Intra-Video FastDedup**) & tạo metadata + nhãn mồi (**Automated Data Profiling**), và nén file ZIP tự động upload lên CVAT.

---

## 📁 1. CẤU TRÚC THƯ MỤC DỮ LIỆU

Hệ thống đã được thiết lập sẵn cấu trúc thư mục mặc định bên dưới:

```text
ToolTrackVideo/
├── data/
│   ├── inputs/
│   │   ├── videos/   <-- THẢ CÁC FILE VIDEO (.mp4, .avi, .mov,...) VÀO ĐÂY (CÓ THỂ CHỨA NHIỀU VIDEO)
│   │   └── images/   <-- HOẶC THẢ THƯ MỤC / CÁC FILE ẢNH THÔ VÀO ĐÂY
│   └── outputs/      <-- KẾT QUẢ SẼ TỰ ĐỘNG XUẤT RA ĐÂY
│       ├── deduplicated_images/  (Ảnh đã lọc trùng)
│       ├── pre_labels/           (Nhãn mồi chuẩn YOLO .txt để upload CVAT)
│       ├── cvat_uploads/         (File ZIP đã đóng gói sẵn sàng upload lên CVAT)
│       └── data_profile.json     (File metadata profiling chi tiết)
├── fast_dedup_profiling.py       (Script chạy chính)
├── package_for_cvat.py           (Công cụ đóng gói ZIP CVAT chống trùng)
└── guide.md
```

---

## 🚀 2. CHẠY PIPELINE (ĐƠN GIẢN NHẤT)

### Bước 1: Thả dữ liệu vào thư mục
- **Nếu chạy cho Video**: Đặt file video của bạn (có thể thả nhiều video cùng lúc như `video1.mp4`, `video2.mp4`) vào thư mục `data/inputs/videos/`.
- **Nếu chạy cho Thư mục ảnh**: Thả các file ảnh thô (`.jpg`, `.png`,...) vào thư mục `data/inputs/images/`.

### Bước 2: Chạy duy nhất 1 lệnh Terminal
Mở terminal tại thư mục dự án `ToolTrackVideo` và chạy:

```bash
python fast_dedup_profiling.py
```

> **Hệ thống tự động xử lý và bảo vệ dữ liệu:**
> 1. **Tự động gắn Tiền tố tên video (Prefix)**: Ví dụ file `video1.mp4` sẽ sinh ảnh `video1_frame_000001.jpg`, `video2.mp4` sẽ sinh `video2_frame_000001.jpg`. Do đó khi tải video mới vào chạy tiếp **KHÔNG BAO GIỜ BỊ GHI ĐÈ** ảnh cũ.
> 2. **Tự động cộng dồn Profile (JSON Merge)**: Cập nhật thông tin các ảnh mới vào file `data_profile.json` mà không xóa metadata của các lượt chạy trước.
> 3. **Tự động quét nhiều Video**: Nếu trong `data/inputs/videos/` có 5 video, script sẽ tự động duyệt chạy lần lượt cả 5 video.
> 4. **Tự động đóng gói ZIP CVAT**: Tạo file ZIP tại `data/outputs/cvat_uploads/` theo định dạng YOLO 1.1 sẵn sàng import vào CVAT, tự động lọc bỏ các ảnh đã đóng gói trước đó để tránh trùng lặp.

---

## 📦 3. ĐÓNG GÓI THỦ CÔNG LÊN CVAT (NẾU CẦN)

Nếu bạn chỉ muốn đóng gói các ảnh mới vừa lọc trùng chưa được nén lên CVAT:

```bash
python package_for_cvat.py
```

---

## ⚙️ 4. TÙY CHỌN NÂNG CAO (DÀNH CHO ADVANCED USER)

Nếu bạn muốn chỉ định đường dẫn dữ liệu cụ thể hoặc điều chỉnh thông số lọc, bạn có thể truyền các đối số CLI:

```bash
python fast_dedup_profiling.py --input /duong/dan/tuy/y/video.mp4 --output_dir ./my_output --hd_thresh 5
```

### Bảng tham số:
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--input` | *Tự động quét* | Đường dẫn đến file Video HOẶC Thư mục chứa ảnh. |
| `--output_dir` | `data/outputs` | Thư mục lưu kết quả. |
| `--hd_thresh` | `5` | Ngưỡng Hamming Distance lọc trùng pHash (giá trị nhỏ hơn = lọc khắt khe hơn). |
| `--model_path` | `yolo11n.pt` | Mô hình YOLO đếm phương tiện. |
| `--conf` | `0.25` | Ngưỡng confidence YOLO. |
| `--device` | *Tự chọn* | `cuda` (nếu có GPU) hoặc `cpu`. |
| `--no_zip` | `False` | Tắt tự động nén ZIP CVAT ở cuối pipeline. |

---

## 📊 5. KẾT QUẢ ĐẦU RA (OUTPUT)

Sau khi chạy xong, trong thư mục `data/outputs/` sẽ xuất hiện:

1. **`deduplicated_images/`**: Chứa danh sách ảnh "sống sót" (đã loại bỏ ảnh trùng lặp và giữ lại frame có nhiều xe nhất).
2. **`pre_labels/`**: Các file `.txt` tương ứng với mỗi ảnh sống sót theo chuẩn 5 classes GreenSM (`0: greensm, 1: car, 2: motorcycle, 3: bus, 4: truck`).
3. **`cvat_uploads/`**: Chứa các file ZIP đóng gói theo định dạng YOLO 1.1 (`cvat_batch_YYYYMMDD_HHMMSS.zip`) sẵn sàng upload CVAT.
4. **`data_profile.json`**: File chứa thông tin phân tích chất lượng ảnh chuyên sâu (được cộng dồn tự động):
   ```json
   {
     "route01_frame_000001.jpg": {
       "source": "route01.mp4",
       "vehicle_count": 15,
       "brightness_v": 52.4,
       "laplacian_var": 142.1,
       "tags": {
         "is_night": true,
         "is_blur": false,
         "has_small_object": true,
         "has_high_occlusion": true,
         "is_dense_scene": true
       }
     }
   }
   ```
