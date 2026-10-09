Hãy viết cho tôi một script CLI bằng Python để thực hiện gộp 2 bước đầu tiên trong pipeline xử lý dữ liệu Computer Vision (Dự án GreenSM Vehicle Detection):
1. INTRA-VIDEO FASTDEDUP (Lọc trùng lặp frame)
2. AUTOMATED DATA PROFILING (Tự động trích xuất metadata & chuẩn bị nhãn mồi)

Yêu cầu tiêu chuẩn: Code gọn nhẹ, tối ưu hiệu năng (CPU/GPU), dùng Streamer đọc frame để tránh tràn RAM, chạy bằng dòng lệnh terminal (CLI), không cần GUI.

---

### 1. ĐẦU VÀO & ĐẦU RA (INPUT/OUTPUT)
- Input CLI Arguments:
  + `--input` (str): Đường dẫn tới file Video (.mp4, .avi,...) HOẶC Thư mục chứa ảnh thô.
  + `--output_dir` (str): Thư mục lưu kết quả.
  + `--hd_thresh` (int, default=5): Ngưỡng Hamming Distance để gom cụm pHash.
- Output Structure:
  + `<output_dir>/deduplicated_images/`: Chứa các bức ảnh "sống sót" sau khi lọc trùng.
  + `<output_dir>/data_profile.json`: File tổng hợp metadata profiling của toàn bộ ảnh sống sót.
  + `<output_dir>/pre_labels/`: Chứa các file `.txt` nhãn mồi định dạng YOLO normalized (0-1) để sẵn sàng upload lên CVAT.

---

### 2. LOGIC XỬ LÝ CHI TIẾT (2-STAGE PIPELINE)

#### GIAI ĐOẠN 1: DUAL-STREAM FAST DEDUPLICATION (Lọc trùng)
1. Pre-sampling:
   - Nếu input là Video: Dùng OpenCV đọc dạng Generator, chỉ trích xuất đúng 1 frame/giây (1 FPS).
   - Nếu input là Thư mục ảnh: Đọc EXIF Timestamp (nếu có), gom nhóm các ảnh chụp cách nhau < 1 giây.
2. Lọc tinh bằng pHash (CPU-based Hashing):
   - Tính Perceptual Hash (pHash) cho các frame bằng `imagehash` hoặc OpenCV.
   - So sánh trượt (Sliding Comparison): Nếu Hamming Distance (HD) giữa frame hiện tại và frame gốc của cụm < `hd_thresh`, gom chúng vào cùng 1 cụm trùng bối cảnh.
3. Quy tắc giải quyết xung đột (Keep-Most-Vehicles):
   - Khi phát hiện 1 cụm các frame trùng nhau (HD < threshold), KHÔNG xóa ngẫu nhiên.
   - Chạy mô hình YOLO11n COCO (`yolo11n.pt`) quét các frame trong cụm trùng đó.
   - Đếm tổng số phương tiện thuộc 4 classes: `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
   - Ưu tiên GIỮ LẠI 01 frame duy nhất có TỔNG SỐ PHƯƠNG TIỆN LỚN NHẤT. Loại bỏ các frame trùng còn lại.

#### GIAI ĐOẠN 2: AUTOMATED DATA PROFILING (Tính toán trên các ảnh sống sót)
*LƯU Ý CỰC KỲ QUAN TRỌNG: Tận dụng luôn kết quả Inference từ YOLO11n COCO ở Giai đoạn 1 (Bounding Boxes, Conf, Class) của các ảnh sống sót. KHÔNG được load/run lại YOLO lần thứ 2.*

Trích xuất các chỉ số metadata sau cho mỗi ảnh sống sót và lưu vào `data_profile.json`:
1. Day / Night Tag:
   - Chuyển ảnh sang không gian màu HSV, tính trung bình kênh V (Brightness).
   - Nếu V_mean < 65 -> Tag: `is_night: true`, ngược lại `false`.
2. Blur Tag:
   - Tính phương sai toán tử Laplacian (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
   - Nếu Variance < 100 -> Tag: `is_blur: true`, ngược lại `false`.
3. Scale Variance Tag (Xe nhỏ):
   - Dựa vào BBox từ YOLO-COCO: Tính `Area_box / Area_img`.
   - Nếu có ít nhất 1 BBox phương tiện có tỷ lệ < 0.01 (1%) -> Tag: `has_small_object: true`.
4. High Occlusion Tag (Che khuất):
   - Tự viết hàm tính IoA (Intersection over Area) giữa từng cặp BBox: `IoA = Area_Intersection / Area_Box_B`.
   - Nếu có cặp BBox nào có `IoA > 0.30` (30%) -> Tag: `has_high_occlusion: true`.
5. Density Tag (Mật độ đông):
   - Nếu tổng số phương tiện YOLO-COCO đếm được > 12 xe -> Tag: `is_dense_scene: true`.
6. Xuất nhãn mồi Pre-label:
   - Xuất file `.txt` tương ứng với tên ảnh vào thư mục `pre_labels/` theo chuẩn YOLO format (`class x_center y_center width height`).

---

### 3. CẤU TRÚC JSON OUTPUT (`data_profile.json`)
```json
{
  "frame_0012.jpg": {
    "source": "video_route_01.mp4",
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

### 4. YÊU CẦU KỸ THUẬT
Ngôn ngữ: Python 3.10+

Thư viện: opencv-python, ultralytics, Pillow, imagehash, numpy, tqdm, argparse.

Code sạch, modular (fast_dedup(), profile_image()), có thanh tiến trình tqdm và log chi tiết từng giai đoạn.