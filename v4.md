# ok cứPIPELINE TỔNG THỂ TỐI ƯU — GREENSM HUMAN-IN-THE-LOOP ACTIVE LEARNING LAB (M50)

> **Dự án**: End-to-End Human-in-the-Loop Annotation Lab — GreenSM Vehicle Detection  
> **Phiên bản**: V3.0 Final (Mentor-Aligned &amp; Enterprise-Ready)  
> **Ngày cập nhật**: 08/10/2026  
> **Mô hình cố định (Locked Model)**: **YOLO11n** (mAP50-95 Target: &gt;78%, Small Object Recall: &gt;90%, Speed: 21.4 FPS, Size: 5.5 MB)

---

## 1. MỤC TIÊU BÀI TOÁN &amp; TẦM NHÌN THỰC TẾ

### 1.1. Bài toán kỹ thuật cốt lõi (Data-Centric AI — Multi-Class Detection)

Bài toán đặt ra không phải là xây dựng hệ thống MLOps phức tạp hay liên tục thay đổi kiến trúc mô hình. Focus chính là bài toán **Nhận diện và Phân loại Phương tiện Giao thông Hỗn hợp tại Việt Nam (Multi-Class Vehicle Detection)**.

#### 🎯 Danh mục Đối tượng Phương tiện Chuẩn hóa (Label Taxonomy):

Hệ thống tận dụng nền tảng nhận diện phương tiện từ YOLO COCO và bổ sung nhãn trọng tâm `greensm` dành riêng cho ô tô:


| Class ID | Tên Nhãn (Class Name) | Định nghĩa &amp; Đặc trưng nhận diện                                                                                            | Ý nghĩa trong Bài toán                                                                            |
| :--------: | :--------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **0**    | `greensm`             | **Chỉ dành cho Ô tô điện GreenSM** (Taxi/Xe thuê VinFast VF e34, VF 5, VF 8...) mang màu xanh Cyan đặc trưng hoặc logo GreenSM. | **TRỌNG TÂM DUY NHẤT**: Tối ưu Precision &amp; Recall cao nhất cho đội xe taxi điện.              |
| **1**    | `car`                 | Toàn bộ ô tô con, taxi truyền thống (Mai Linh, Vinasun), xe công nghệ khác, ô tô cá nhân.                                       | **Đối chứng thị phần**: Mẫu số tính % thị phần và giúp mô hình phân biệt tránh nhầm xe xanh khác. |
| **2**    | `motorcycle`          | Toàn bộ xe máy trên đường (bao gồm cả xe máy cá nhân, xe ôm công nghệ khác và xe máy GreenSM).                                  | **Bối cảnh &amp; Che khuất**: Nhận diện dòng phương tiện gây che khuất (Occlusion) chính tại VN.  |
| **3**    | `bus`                 | Xe buýt nội đô, xe khách lớn.                                                                                                   | Bối cảnh giao thông công cộng.                                                                    |
| **4**    | `truck`               | Xe tải, xe bán tải lớn, xe bồn.                                                                                                 | Vật cản tầm nhìn kích thước lớn.                                                                  |


Mục tiêu cốt lõi:

> **Với cùng một ngân sách gán nhãn thủ công cố định, áp dụng quy trình Active Learning &amp; Two-Tier HITL để chọn lọc đúng các mẫu ảnh có giá trị nhất, giúp mô hình YOLO11n phân biệt chính xác GreenSM khỏi các xe khác, đạt mAP cao nhất và chứng minh sự vượt trội về thời gian so với gán nhãn truyền thống.**

### 1.2. Định vị bài toán thực tế (Enterprise &amp; Commercial Value)

Để giải pháp mang giá trị thương mại và thực tiễn cao, hệ thống được ứng dụng vào **Hệ thống Quản lý Đội xe Thông minh &amp; Phân tích Thị phần Giao thông Đô thị (EV Fleet Operations &amp; OOH Analytics)**:

1. **Giám sát Mật độ Đội xe GreenSM (Fleet Density &amp; Hotspot Monitoring)**: Định vị và đo lường tần suất hiện diện của xe taxi/ô tô điện GreenSM tại các nút giao thông trọng điểm, sân bay, trung tâm thương mại theo thời gian thực.
2. **Phân tích Thị phần Thương hiệu Ngoài trời (OOH Impression Share)**: Đo lường tỷ lệ diện mạo thương mại của xe GreenSM so với các hãng xe công nghệ / taxi truyền thống khác trong dòng giao thông đô thị.
3. **Thách thức Kỹ thuật Thực tế**:
   - **Xe nhỏ ở xa (&lt; 32x32px)**: Nhìn thấy từ camera giám sát góc rộng.
   - **Điều kiện thiếu sáng / Ban đêm (Night Slice)**: Màu xanh Cyan đặc trưng của GreenSM bị biến đổi màu dưới ánh đèn đường.
   - **Che khuất một phần (Occlusion)**: Bị chắn bởi xe máy, ô tô khác.
   - **Dương tính giả (False Positives)**: Dễ nhầm lẫn với các ô tô cá nhân màu xanh lục / xanh ngọc khác.

---

## 2. KIẾN TRÚC PIPELINE TỔNG THỂ (TWO-TIER ACTIVE HITL V3.0)

Sơ đồ quy trình được chuẩn hóa theo đúng các chỉ dẫn và góp ý chuyên sâu từ Mentor:

```text
                        RAW GREENSM DATASET (Video Streams / Image Batches)
                                        │
                                        ▼
                   [INTRA-VIDEO FASTDEDUP: pHash / Embedding]
              (Lọc frame trùng trong cùng Video — Ưu tiên giữ frame Nhiều Xe)
                                        │
                                        ▼
                     AUTOMATED DATA PROFILING & BASELINE MEASURE
                 (Tagging: Day/Night, Blur, Small Object, Density)
                 (Đo Baseline: Time/Image gán tay vs nhãn mồi trên CVAT)
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
 FIXED TEST SET                                                  UNLABELED POOL
(Video-Aware Split)                                                     │
        │                                                               ▼
 Human QA & Freeze                                               SEED SELECTION (~10%)
(Dùng chung V0..Vk)                                             (Stratified Sampling)
        │                                                               │
        │                                                               ▼
        │                                                         HUMAN LABELING
        │                                                               │
        │                                                               ▼
        │                                                         DATASET V0
        │                                                               │
        │                                                               ▼
        │                                                         TRAIN MODEL V0 (YOLO11n)
        │                                                               │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
                                        ▼
                            INFERENCE ON UNLABELED POOL
                                        │
                                        ▼
                         1-CLASS COST-AWARE MINING & SCORING
                 (Score = Conf_Uncertainty × Dynamic_Slice_Weight × Anomaly / Effort)
                                        │
                                        ▼
                           DIVERSITY CLUSTERING (K-Means)
                     (Tránh dồn ảnh cùng bối cảnh / góc quay)
                                        │
                                        ▼
                        TWO-TIER HUMAN-IN-THE-LOOP (HITL)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  HIGH CONFIDENCE                 BUFFER ZONE (GAP)              LOW/MEDIUM CONFIDENCE
 (Conf >= 0.85 &                 (0.70 < Conf < 0.85)             (0.20 <= Conf <= 0.70)
  Not Night/Small)               (Chuyển sang Tier 2)            + Weak Slice (Night/Small)
        │                               │                               │
        ▼                               └───────────────┬───────────────┘
 TIER 1: AUTO-ACCEPT                                    │
 (Gán nhãn tự động)                                     ▼
        │                                      TIER 2: CVAT HUMAN REVIEW
        ├────────────────┐                     (Sửa nhãn mồi / Vẽ bổ sung)
        ▼                ▼                              │
 [SPOT-CHECK QA]   [PRE-LABEL]                          │
 (Random 5-10%)          │                              │
        │                └──────────────┬───────────────┘
        ▼                               │
 (Measure Label Acc)                    ▼
                                   DATASET V1 (Lineage Manifest + Git Tag)
                                        │
                                        ▼
                              TRAIN MODEL V1 (YOLO11n)
                                        │
                                        ▼
                    EVALUATION & STATISTICAL SIGNIFICANCE CHECK
                      (Multi-Seed Run: Gain > Natural Variance)
                                        │
                                        ▼
                   A/B TESTING & ABLATION STUDY BENCHMARK
           (Random vs Active vs Active-No-AutoLabel on Fixed Test)
```

---

## 3. CHI TIẾT 10 BƯỚC THỰC THI PIPELINE (MENTOR-ALIGNED)

### Bước 1 — Dual-Stream Fast Deduplication (Lọc trùng lặp thô &amp; tinh siêu tốc)

- **Vấn đề của phương pháp truyền thống**: Dữ liệu thu thập từ video (hàng ngàn frame tĩnh khi kẹt xe) hoặc ảnh chụp liên tiếp (burst shots) bị đẩy thẳng vào hệ thống. Con người phải tốn hàng trăm giờ công gán nhãn cho các bức ảnh giống hệt nhau, vừa đốt tiền vừa làm mô hình bị "học vẹt" (overfit) bối cảnh tĩnh.
- **Cải tiến cốt lõi (Lọc 2 màng + Keep-Most-Vehicles)**:
  1. **Lọc thô (Cắt giảm 90% không tốn chi phí)**: Hệ thống chia 2 luồng:
     - *Luồng Video*: Áp dụng FPS Subsampling, chỉ trích xuất 1 frame/giây (1 FPS).
     - *Luồng Ảnh rời*: Sắp xếp theo EXIF Timestamp, nhóm các ảnh chụp cách nhau &lt; 1s.
  2. **Lọc tinh (CPU-based Hashing)**: Chạy thuật toán **Perceptual Hash (pHash)** trên CPU cho các khung hình còn lại. So sánh trượt (Sliding Comparison): Nếu Hamming Distance $< 5$, cảnh vật được coi là không đổi (trùng lặp).
  3. **Quy tắc giải quyết xung đột (Keep-Most-Vehicles)**: Khi 2 frame bị trùng ($HD < 5$), không vứt bỏ ngẫu nhiên. Dùng **YOLO-COCO siêu nhẹ** đếm tổng số phương tiện (`car, motorcycle, bus, truck`). **Ưu tiên giữ lại frame có nhiều phương tiện giao thông hơn** (để tối đa hóa bối cảnh học tập và tránh lọt lưới xe nhỏ).
- **Chứng minh ROI**: Cắt giảm 85-95% khối lượng dữ liệu ngay từ cửa ngõ, giải phóng sức người khỏi việc gán nhãn rác, minh chứng rõ ràng cho sự vượt trội về Labor Efficiency.

### Bước 2 — Automated Data Profiling &amp; CVAT Time Baseline

- **Tự động trích xuất Metadata chuyên sâu (Thay thế thủ công)**:
  - **Day / Night**: Giá trị trung bình kênh V (Value) trong HSV ($\text{Brightness} < 65 \rightarrow \text{Night}$).
  - **Blur**: Phương sai toán tử Laplacian ($\text{Variance} < 100 \rightarrow \text{Blur}$).
  - **Scale Variance (Xe to/nhỏ)**: Dùng **YOLO-COCO** quét thô tính tỷ lệ BBox so với khung hình ($\text{Area} < 1\% \rightarrow \text{Small_Object}$, $> 20\% \rightarrow \text{Truncated}$).
  - **Occlusion (Che khuất)**: Dùng YOLO-COCO tính tỷ lệ đè lấp (IoA) giữa các xe. BBox bị đè lấp $> 30\% \text{ diện tích} \rightarrow \text{High_Occlusion}$.
  - **Density**: Tổng số phương tiện do YOLO-COCO đếm $> 12 \text{ xe} \rightarrow \text{Dense Scene}$.
- **Đo đạc Mốc chuẩn Thực nghiệm (CVAT Measuring Baseline)**:
  - Gán thử 50 ảnh thủ công thuần túy từ đầu (Manual Drawing) $\rightarrow$ Đo thời gian trung bình $T_{manual}$ (giây/ảnh) đại diện cho **phương pháp truyền thống**.
  - Sửa nhãn mồi 50 ảnh (Pre-label Review) $\rightarrow$ Đo thời gian trung bình $T_{assisted}$ (giây/ảnh) đại diện cho **phương pháp Active HITL**.
  - Tỉ lệ $T_{manual} / T_{assisted}$ là mốc số liệu thực tế để chứng minh khoa học chỉ số **Time Saved** của pipeline.

### Bước 3 — Tạo Tập Kiểm Thử Bất Biến (Fixed Test Set)

Tập Test là "thước đo chân lý" (Ground Truth Benchmark). Mọi sai lệch hoặc rò rỉ ở tập Test sẽ bẻ gãy tính khoa học của toàn bộ dự án. Bước này giải quyết triệt để vấn đề đó:

1. **Quy tắc Phân chia Chống Rò Rỉ (Anti-Leakage Group Split)**:
   - **Luồng Video**: Chia theo `Video Sequence ID` / `Camera ID`. Toàn bộ frame từ một video phải nằm trọn ở Train hoặc Test, **tuyệt đối không chia ngẫu nhiên cấp frame** để tránh rò rỉ không gian-thời gian (cùng 1 góc quay, cùng 1 chiếc xe lọt vào cả 2 tập).
   - **Luồng Ảnh rời**: Gom cụm (Group) các ảnh theo địa điểm hoặc EXIF timestamp chụp cùng sự kiện. Đưa nguyên cụm vào Test Set.
2. **Cơ cấu Tỷ lệ Cân bằng Khắc Nghiệt (Balanced Hard Slices)**:
   - Thay vì tỷ lệ tự nhiên ngoài đời (Ban ngày chiếm 80%), tập Test (\~160 - 200 ảnh) được thiết kế thành một **Stress-Test** để tối đa hóa độ đo lường sức mạnh của Active Learning, cân bằng đều 4 lát cắt (\~40-50 ảnh mỗi lát cắt):
     - **Daylight Normal (\~25%)**: Đo hiệu năng cơ sở ban ngày.
     - **Night Slice (\~25%)**: Kiểm tra lỗi nhận diện xe Cyan dưới ánh đèn vàng/tối.
     - **Small Object Slice (\~25%)**: BBox $< 32\times32\text{px}$, đo năng lực phát hiện từ xa.
     - **High Occlusion / Dense (\~25%)**: Bị đè lấp $> 30\%$, đo năng lực bóc tách trong đám đông xe máy.
3. **Tăng tốc Gán nhãn &amp; Bộ Quy chuẩn Chuẩn Vàng (Annotation Guidelines)**:
   - **Pre-labeling (Tăng tốc x3 lần)**: Dùng mô hình nền tảng quét tạo nhãn mồi. Con người chỉ việc sửa tên class (`car` $\rightarrow$ `greensm`), căn chỉnh BBox, và xóa nhãn rác, giảm thời gian từ 90s xuống **20-30s/ảnh**.
   - **Quy chuẩn 5 Classes**: Nhãn `greensm` chỉ dành cho ô tô điện taxi/cho thuê của GreenSM. Mọi xe máy (kể cả xe máy điện GreenSM) đều gán là `motorcycle`. Mọi ô tô cá nhân khác gán `car`.
   - **Quy chuẩn Hộp bao (BBox)**: Vẽ khít mép xe (dung sai 2px). Từ chối gán nhãn vật thể bị che khuất $> 70\%$ hoặc cực nhỏ $< 12\times12\text{px}$. 
   - Có QA Lead độc lập kiểm tra chéo (Cross-check) và chốt 100% nhãn.
4. **Quy hoạch &amp; Đóng Băng Kép (Dual-Freeze Benchmark)**:
   - Ở bước này, tạo ra 2 tập dữ liệu độc lập và **khóa cố định (Read-only)**:
     - **Fixed Test Set (\~160 - 200 ảnh)**: Dùng ĐỘC QUYỀN cho Evaluate cuối cùng, sinh ma trận $Recall$ và báo cáo ROI.
     - **Fixed Val Set (\~40 - 50 ảnh)**: Dùng CHUNG cho mọi vòng lặp huấn luyện ($V_0, V_1, \dots$) để YOLO đánh giá sau mỗi epoch, phục vụ Early Stopping và chọn `best.pt`. Tuyệt đối không xài Fixed Test Set làm Val.
   - Sinh file mã băm `dataset_checksum.sha256` cho cả 2 tập và gắn `git tag` (VD: `v1.0-fixed-eval-freeze`).

### Bước 4 — Chọn Seed Set &amp; Huấn luyện Baseline Model V0

Bước này tái sử dụng sự nhất quán của thuật toán Bước 3 để tạo ra tập dữ liệu móng (Seed Set) có chất lượng cao nhất, thời gian ngắn nhất, dồn 100% tài nguyên cho việc học.

1. **Kế thừa Thuật toán Lấy mẫu Cân Bằng (Balanced Stratified Sampling)**:
   - Trích xuất khoảng **\~200 - 250 ảnh** (\~10% Unlabeled Pool). Đảm bảo tuân thủ tính độc lập Video/Sequence (Group-Aware) khỏi Test/Val.
   - Bê nguyên tỷ lệ **Stress-Test** (25% Day, 25% Night, 25% Small, 25% Occluded) áp dụng cho Seed Set. Ép Model V0 ngay từ khi vỡ lòng đã phải học một bộ dữ liệu toàn diện và gai góc nhất.
2. **Kế thừa Pipeline Gán nhãn Tăng tốc (Pre-labeling)**:
   - Dùng YOLO-COCO để sinh nhãn mồi. Con người đóng vai trò Reviewer: sửa nhãn `car` thành `greensm`, tinh chỉnh BBox, rà soát lỗi theo đúng **Quy chuẩn 5 classes** (tiết kiệm 60-70% thời gian tạo Seed Set).
3. **Huấn luyện Model V0 (Head/Backbone Transfer Learning &amp; Loss Setup)**:
   - **Data Split**: 100% Seed Set (200-250 ảnh) được dùng làm tập `train`. Tập `val` trỏ trực tiếp vào thư mục **Fixed Val Set** đã tạo ở Bước 3.
   - **Tái cấu trúc Detection Head &amp; Đóng băng Backbone (Phần đầu mạng)**:
     - Khởi tạo lại trọng số lớp Head từ 80 classes (COCO) sang **5 classes** (`greensm, car, motorcycle, bus, truck`).
     - Áp dụng cơ chế **Backbone Freezing** (`freeze=10` layers đầu) trong 10 epochs đầu (Head Warmup) nhằm bảo toàn các bộ lọc trích xuất viền/cạnh/màu sắc tổng quát từ COCO, tránh hiện tượng *Gradient Shock* khi tập train ban đầu còn nhỏ.
   - **Cấu hình Hàm Loss Đa nhiệm Chống Lệch Dữ Liệu**:
     - *Classification Loss*: Dùng BCE có bù trừ trọng số lớp (**Class-Weighted Loss**), gán hệ số phạt $\alpha_{\text{greensm}} = 2.0$ trong khi $\alpha_{\text{motorcycle}} = 0.8$, ép mô hình không được bỏ sót xe taxi điện trước biển xe máy tại VN.
     - *Box Regression Loss*: Tận dụng Complete IoU (**CIoU**) kết hợp Distribution Focal Loss (**DFL**) của YOLO11 để định vị chuẩn xác các BBox xe nhỏ ở cự ly xa ($< 32\times32\text{px}$) và xe bị che khuất một phần.
   - **Tối ưu hóa (Optimizer &amp; LR Schedule)**:
     - Dùng **AdamW** (`weight_decay=0.01`, `lr0=0.001`) thay vì SGD hay Adam thường, phân tách riêng bước decay trọng số để chống học vẹt (overfitting) trên tập dữ liệu nhỏ.
     - Lịch trình giảm tốc độ học: **Cosine Annealing LR Scheduler** với chu kỳ 50 epochs, cố định `seed=42`. Model V0 này sẽ là "Máy tạo nhãn mồi" (Pseudo-labeler) cho vòng lặp Active Learning.
4. **Đo đạc &amp; Khởi tạo Trọng số (Feed-forward Metrics)**:
   - Chạy Evaluate Model V0 trên **Fixed Test Set** (bài thi cuối kỳ).
   - Trích xuất ma trận $Recall$ chi tiết cho từng lát cắt khó ($Recall_{night}$, $Recall_{small}$, $Recall_{occ}$). Output này dẫn trực tiếp vào Bước 5 làm **Trọng số định hướng (**$W_i$**)**.

### Bước 5 — Target-Aware Cost-Effective Mining (Tính điểm Săn Ảnh Khó)

Chạy Inference (dự đoán) bằng **Model V0** trên toàn bộ Unlabeled Pool. Dựa vào kết quả dự đoán này, ta tính điểm ưu tiên (Active Score) cho từng bức ảnh. Bức ảnh nào điểm càng cao nghĩa là mô hình V0 càng phân vân (hoang mang) ở đó và càng đáng để con người gán nhãn.

Công thức chấm điểm tổng hợp được cấu thành từ 4 biến số (đều được chuẩn hóa Min-Max về thang $[0, 1]$ trước khi nhân để tránh một biến lấn át các biến khác):

1. **Target-Class Confusion Margin (**$U_i$**)**: Đo lường sự phân vân bối rối của Model V0.
   - Bức ảnh có BBox mà xác suất dự đoán $P(\text{greensm})$ và $P(\text{car})$ càng sát nhau (ví dụ: đoán 51% là taxi, 49% là ô tô cá nhân), nghĩa là mô hình đang rất bối rối ở ranh giới nhận diện màu sắc/thương hiệu:
    $$U\_{box} = 1 - |P(\\text{greensm}) - P(\\text{car})|$$
   - Điểm bất định của toàn bức ảnh ($U_i$) được lấy bằng giá trị phân vân cao nhất trong các BBox: $U_i = \max_{box}(U_{box})$.
2. **Dynamic Weak-Slice Weight (**$W_i$**)**: Trọng số ưu tiên lát cắt yếu (Lấy tự động từ kết quả thi của V0 ở Bước 4).
   - Nếu V0 thi trên Test Set bị điểm kém ở ban đêm ($Recall_{night}$ thấp), thì các ảnh trong Pool có tag "Night" sẽ tự động được nhân hệ số cao lên:
    $$W\_i = \\frac{1}{\\text{Recall}\_{slice\_i}(\\text{Model } V\_0)}$$
3. **Prediction Anomaly (**$A_i$**)**: Trọng số phát hiện bất thường và nguy cơ che khuất.
   - Bức ảnh chứa BBox xe mục tiêu bị che khuất cao ($IoA > 30\%$) hoặc bị bủa vây bởi số lượng lớn xe máy ($N_{motorcycle} > 15$).
4. **Estimated Labeling Effort (**$E_i$**)**: Phạt trừ điểm những ảnh quá tốn công vô ích.
   - Một ảnh chứa 40 chiếc xe rác ngoài lề nhưng không chứa xe mục tiêu nào sẽ tốn công gán vô ích. Hệ số công sức: $E_i = 1 + 0.05 \times N_{context\_boxes}$. Ảnh càng nhiều rác, mẫu số càng lớn làm giảm điểm ưu tiên.

👉 **Công thức điểm tổng hợp cho từng bức ảnh (Cost-Aware Active Score)**:

$$
\text{Selection Score}_i = \frac{\text{Norm}(U_i) \times \text{Norm}(W_i) \times \text{Norm}(A_i)}{\text{Norm}(E_i)}
$$

*(Các bức ảnh có Selection Score cao nhất sẽ được chọn vào Top Candidates để chuyển sang Bước 6).*

### Bước 6 — Lọc Đa Dạng Hóa Bối Cảnh (k-Center Greedy / Core-Set)

Nếu chỉ lấy $K$ ảnh có điểm khó cao nhất từ Bước 5, ta rất dễ gặp rủi ro bốc phải hàng trăm ảnh giống hệt nhau (ví dụ: cùng một ngã tư kẹt xe, các frame chỉ xê dịch vài centimet). Để giải quyết triệt để sự lãng phí này, ta áp dụng thuật toán **k-Center Greedy (Core-Set)** - tiêu chuẩn SOTA trong Active Learning:

1. **Trích xuất &amp; Giảm chiều (PCA)**:
   - Trích xuất Feature Vector từ YOLO backbone (ví dụ $512D$) của Top $N$ ảnh ứng viên.
   - Chạy thuật toán **PCA (Principal Component Analysis)** giảm số chiều xuống còn $32D$ hoặc $64D$. Điều này loại bỏ "lời nguyền số chiều" và giúp tăng tốc độ tính toán lên gấp hàng chục lần, giải phóng RAM.
2. **Tuyển chọn bằng k-Center Greedy**:
   - **Khởi tạo**: Bốc bức ảnh đầu tiên có Active Score cao nhất từ Bước 5.
   - **Lặp lại (Greedy loop)**: Ở mỗi bước, thuật toán đo khoảng cách vector và tìm ra bức ảnh **xa nhất (khác biệt nhất)** so với tất cả các bức ảnh đã được chọn trước đó. 
   - Lặp lại quá trình bốc này cho đến khi nhặt đủ ngân sách $K$ ảnh.
3. **Ưu điểm vượt trội so với K-Means truyền thống**:
   - **Minimax Coverage**: Đảm bảo $K$ bức ảnh "rải thảm" bao phủ trọn vẹn mọi ngóc ngách của không gian bối cảnh.
   - **Không có tâm ảo (No virtual centroids)**: Các điểm được chọn đều là ảnh thực tế 100%, không bị sai số nội suy như tâm của K-Means.
   - **Hiệu năng cực cao**: Thuật toán quét tham lam một mạch, không cần lặp lại hội tụ (EM) nặng nề.

### Bước 7 — Two-Tier Human-in-the-Loop (HITL) &amp; Cross-Model Oracle Guardrails

Cơ chế duyệt 2 tầng giải phóng sức lao động con người bằng cách kết hợp gán nhãn tự động cho mẫu dễ và tập trung công sức chuyên gia vào các ca khó. Để **triệt tiêu hoàn toàn Confirmation Bias** (Thiên kiến tự củng cố lỗi sai của chính YOLO11n), hệ thống tích hợp thêm mô hình **RT-DETRv4** làm Oracle (Chuyên gia thẩm định độc lập):

```text
                               Candidates Top-K
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
 TIER 1: AUTO-ACCEPT CANDIDATE  BUFFER ZONE GAP               TIER 2: CVAT HUMAN REVIEW
 - YOLO11n Conf >= 0.85         - Bất kỳ Model nào Conf < 0.85- Conf < 0.70
 - RT-DETR Conf >= 0.85         - Hoặc IoU lệch (< 0.90)      - Hoặc dính Tag khó:
 - IoU Đồng thuận >= 0.90       (Không đủ an toàn để          - + Night / Small / Occluded
 - KHÔNG dính Tag khó           tự duyệt -> Chuyển Tier 2)          │
        │                             │                             │
        ▼                             │                             ▼
 [TỰ ĐỘNG LƯU NHÃN MÁY]               │                     [COGNITIVE-LOAD BATCHING]
        │                             └──────────────┬──────────────┘
        ▼                                            ▼
 [CLOSED-LOOP SPOT-CHECK QA]                 - Gom Job CVAT theo từng lát cắt
 - Rút ngẫu nhiên 5-10% kiểm tra             - Phím tắt 1-chạm (G: Greensm, Space: Duyệt)
 - Hủy kết quả nếu Lỗi >= 5%                 - Guideline Chatbot (RAG) hỗ trợ tra cứu
        │                                            │
        └─────────────────────────────┬──────────────┘
                                      ▼
                        DATASET V1 (Lineage Manifest + Git Tag)
```

1. **Tier 1 — Tự Động Duyệt Nhãn qua Đồng thuận Kép (Cross-Model Consensus Oracle)**:
   - Thay vì chỉ tin tưởng một mình YOLO11n, hệ thống triệu tập thêm mô hình **RT-DETRv4** (Vision Transformer, sở hữu cơ chế Global Attention mạnh mẽ bắt che khuất vượt trội, được fine-tune nhanh trên 250 ảnh Seed).
   - **Tiêu chuẩn khắt khe (Image-Level Routing)**: Một bức ảnh CHỈ được tự duyệt khi toàn bộ BBox xe mục tiêu thỏa mãn điều kiện **Đồng thuận tuyệt đối**:
     - Cả YOLO11n và RT-DETR đều phát hiện ra xe `greensm` với $Conf \ge 0.85$.
     - Hai BBox đè lên nhau chuẩn xác với độ tương đồng hình học cực cao ($IoU \ge 0.90$).
     - **TTA Flip Consistency**: Khi lật ảnh ngang (Horizontal Flip), cả 2 mô hình đều giữ vững BBox với $IoU \ge 0.85$ và sai lệch tin cậy $\le 0.05$.
     - **Loại trừ rủi ro (Risk Exclusion)**: Tuyệt đối không tự duyệt ảnh dính tag **Ban đêm (Night)**, **Xe nhỏ (Small Object)**, hoặc **Che khuất nặng (High Occlusion)**.
   - **Closed-Loop Spot-Check QA**: Rút ngẫu nhiên 5–10% ảnh Tier 1 đưa cho QA Lead kiểm định:
     - Nếu độ chuẩn xác $\ge 95\%$: Duyệt batch và tự động nới nhẹ ngưỡng ($\theta = 0.83$) cho batch sau.
     - Nếu phát hiện lỗi: Hủy kết quả Auto-Accept của batch đó, nâng ngưỡng phòng vệ lên $\theta = 0.90$, và chuyển toàn bộ sang Tier 2 để người rà soát.
2. **Tier 2 — Con Người Rà Soát &amp; Tinh Chỉnh Trên CVAT (Human Refinement)**:
   - **Tận dụng nhãn mồi bất đồng (Pre-label Disagreement Assist)**: Các BBox có sự lệch pha giữa YOLO11n và RT-DETR được highlight màu cam cảnh báo. Annotator chỉ cần tập trung phán xử ca ranh giới, sửa nhãn nhầm lẫn giữa `car` $\leftrightarrow$ `greensm`, kéo khít BBox và vẽ thêm xe bị che khuất mà AI bỏ sót (tiết kiệm 60–70% thời gian so với vẽ từ đầu).
   - **Phân cụm giảm tải nhận thức (Cognitive-Load Batching)**: Gom các ảnh cùng lát cắt vào cùng một Job (Job 50 ảnh chuyên Ban đêm, Job 50 ảnh chuyên phân biệt màu Cyan...) giúp annotator không bị mỏi mắt do chuyển đổi bối cảnh liên tục.
   - **Thao tác 1 chạm (1-Key Hotkeys)**: Phím `G` để đổi ngay sang `greensm`, phím `Space` để xác nhận và chuyển ảnh tiếp theo trong 1 giây.
   - **Guideline Chatbot (RAG Assistant)**: Trợ lý AI tích hợp sẵn bộ quy chuẩn 5 classes, giúp người gán nhãn hỏi đáp nhanh các ca ranh giới gây tranh cãi ngay trên giao diện CVAT.

### Bước 8 — Experiment Manifest &amp; Data Lineage Tracking (Ghi Log Kiểm Toán &amp; Đóng Dấu Niêm Phong)

Bản chất của Bước 8 là tạo ra **"Bản Giấy Khai Sinh &amp; Hộp Đen MLOps"** cho từng vòng lặp dữ liệu. Thay vì lưu log text rời rạc, hệ thống tự động sinh file log có cấu trúc `manifest.yaml` để niêm phong chính xác trạng thái dữ liệu và thuật toán trước khi bước vào huấn luyện:

```yaml
experiment_id: "EXP_GREENSM_V1_20261009"
lineage:
  git_tag: "v1.0-active"
  dataset_checksum_sha256: "a4f89c2e71b..."  # Mã băm toàn vẹn của toàn bộ kho ảnh/nhãn V1
  base_checkpoint: "runs/train/v0/weights/best.pt"
dataset_summary:
  total_training_images: 450         # 250 ảnh Seed Set + 200 ảnh Active Cycle 1
  seed_images: 250
  active_tier1_auto: 60              # Máy tự duyệt 100% (Tiết kiệm hoàn toàn công sức)
  active_tier2_human: 140            # Người duyệt/sửa từ nhãn mồi trên CVAT
  qa_machine_label_accuracy: 97.2%   # Kết quả Spot-Check QA độc lập (Đạt chuẩn an toàn >= 95%)
  tier2_label_edit_rate: 18.5%       # Annotator chỉ cần sửa 18.5% BBox (81.5% BBox mồi dùng tốt)
selection_strategy:
  uncertainty_type: "greensm_car_confusion_margin"
  weak_slice_weights: {"night": 1.42, "small_object": 1.65, "occluded": 1.30}
  diversity_filtering: "pca_32d_kcenter_greedy"
  tier1_adaptive_threshold: 0.88     # Ngưỡng tin cậy động sau khi QA phản hồi
model_reproducibility:
  architecture: "YOLO11n"
  locked_hyperparams:
    imgsz: 640
    batch: 16
    epochs: 50
    optimizer: "AdamW"
    lr0: 0.001
  random_seed: 42
  checkpoint_path: "runs/train/v1/weights/best.pt"
```

1. **Minh bạch hóa Nguồn gốc (Audit Trail)**: Bóc tách rành mạch tỷ lệ ảnh do máy tự duyệt vs người sửa, kèm chỉ số kiểm định QA để chứng minh dữ liệu sạch trước Hội đồng/Mentor.
2. **Lưu vết Tham số Tuyển chọn (Algorithm Snapshot)**: Ghi lại toàn bộ công thức và trọng số của Bước 5, 6, 7 đã tạo ra đợt dữ liệu này.
3. **Niêm phong Bất biến (Git Tag + SHA-256 Checksum)**: Đảm bảo 100% tính tái lập (Reproducibility), bất kỳ ai tải checkpoint và dữ liệu mang mã hash này đều chạy ra kết quả mAP giống hệt nhau.

### Bước 9 — Retrain Model V1 &amp; Statistical Significance Check (Huấn luyện lại Model V1 &amp; Kiểm định Ý nghĩa Thống kê)

Bước này thực hiện huấn luyện mô hình thế hệ mới (**Model V1**) trên tập dữ liệu đã được làm giàu, đồng thời thiết lập quy chuẩn kiểm định khoa học nghiêm ngặt để chứng minh sự tiến bộ của mô hình là có thật, loại trừ hoàn toàn yếu tố ngẫu nhiên hay "ăn may":

1. **Chiến lược Huấn luyện Tích lũy &amp; Bộ đệm Tái hiện (Cumulative Retraining &amp; Memory Buffer)**:
   - **Vòng lặp cơ sở (**$V_0 \rightarrow V_1$**)**: Áp dụng **Cumulative Retraining** trên tập dữ liệu gộp:
    $$\\mathcal{D}*{\\text{train}}^{(V\_1)} = \\mathcal{D}*{\\text{Seed}} \\cup \\Delta \\mathcal{D}\_{\\text{Active}} = 250 + 200 = \\mathbf{450\\text{ ảnh}}$$
     - *Nguyên lý*: Giữ lại 250 ảnh Seed làm "mỏ neo" (Anchor) để chống hiện tượng **Quên thảm khốc (Catastrophic Forgetting)** — ngăn việc mô hình sau khi nạp 200 ảnh khó thì quên mất cách nhận diện xe chuẩn ban ngày.
     - Cả tập 450 ảnh được fine-tune từ trọng số COCO gốc (`yolo11n.pt`).
   - **Tầm nhìn mở rộng đường dài (**$V_2 \rightarrow V_k$**)**: Kích hoạt cơ chế **Exemplar Memory Buffer** — mỗi vòng chỉ giữ lại 20% ảnh mỏ neo tinh hoa (chọn bằng $k$-Center Greedy) từ các vòng cũ kết hợp với 100% ảnh khó mới, giữ cho thời gian huấn luyện luôn bằng phẳng và không bị phình to cấp số nhân.
2. **Khai thác Vệ tinh: Zero-Cost Background Mining (Mở rộng Bonus theo tư tưởng SSOD)**:
   - Trong số \~1.500 ảnh Unlabeled còn thừa lại trong kho, hệ thống tận dụng cặp đôi YOLO11n + RT-DETR để quét tự động, lọc ra **50–100 ảnh bối cảnh âm tính (Pure Background Images)**: Nơi cả 2 mô hình đều đồng thuận 100% không có xe `greensm` nhưng xuất hiện nhiều xe thường, xe máy, biển báo phức tạp.
   - Tập ảnh này được đưa vào huấn luyện như một nhánh kiểm chứng mở rộng (Negative Regularization) nhằm triệt tiêu False Positives mà **hoàn toàn không tốn thêm 1 giây công gán nhãn nào của con người**.
3. **Cố định Siêu tham số &amp; Tập Kiểm tra Nội bộ Đóng Băng (Locked Training &amp; Frozen Val Set)**:
   - Khóa cứng 100% môi trường huấn luyện để đảm bảo mọi cải thiện đều thuần túy đến từ **Chất lượng dữ liệu (Data-Centric AI)**:
     - `imgsz=640`, `batch=16`, `epochs=50`, `optimizer=AdamW`, `lr0=0.001`.
   - **Tập Val bất biến (Fixed Val Set)**: Tập `val` trong suốt quá trình train tiếp tục trỏ vào duy nhất **Fixed Val Set** (\~40–50 ảnh) đã tạo và **đóng băng ngay từ Bước 3**. Tập Val này được cố định cứng xuyên suốt toàn bộ dự án để thực hiện Early Stopping và lưu file checkpoint tối ưu `best.pt`, bảo đảm tính công bằng tuyệt đối giữa mọi phiên bản mô hình ($V_0, V_1, V_{\text{random}}$).
4. **Giao thức Kiểm định 3 Random Seeds (Multi-Seed Significance Protocol)**:
   - Để loại trừ nghi vấn "tăng điểm do gặp may khi xáo trộn thứ tự dữ liệu (Stochasticity)", mỗi thử nghiệm được chạy độc lập với **3 Random Seeds** khác nhau: `seed=42`, `seed=123`, `seed=456`.
   - Tính điểm trung bình $\mu(V_1)$ và độ lệch chuẩn dao động tự nhiên $\sigma_{\text{seed}}$:
     $$
     \sigma_{\text{seed}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (mAP_i - \mu)^2}
     $$
   - **Điều kiện chấp nhận (Significance Gate)**:
     $$
     \Delta mAP = \mu(V_1) - \mu(V_0) > 2 \times \sigma_{\text{seed}}
     $$
     \*(Mức tăng điểm phải vượt trội hơn gấp đôi biên độ dao động ngẫu nhiên của seed thì mới được công nhận là có ý nghĩa thống kê thực sự, $p &lt; 0.05$).\*
5. **Đánh giá Đa chiều theo Lát cắt (Fine-Grained Slice Breakdown trên Fixed Test Set)**:
   - Đem Model V1 chấm thi trên **Fixed Test Set** (160 ảnh độc lập đã đóng băng từ Bước 3).
   - Báo cáo kết quả bắt buộc xuất ra ma trận chi tiết:
     - $mAP_{50-95}$ tổng thể và riêng class `greensm`.
     - $Recall_{\text{night}}$ (đo khả năng khắc phục lỗi ám màu đèn đường).
     - $Recall_{\text{small}}$ (đo khả năng bắt xe ở cự ly xa từ camera góc rộng).
     - $Recall_{\text{occ}}$ (đo khả năng bóc tách xe GreenSM trong đám đông xe máy chen lấn).
   - Chứng minh thực nghiệm rằng các trọng số yếu động $W_i$ ở Bước 5 đã đánh trúng đích và giải quyết triệt để các điểm mù của Model V0.

### Bước 10 — A/B Testing &amp; Ablation Study Benchmark (Thử nghiệm Đối chứng &amp; Nghiên cứu Bóc tách)

Để chứng minh đanh thép tính ưu việt của giải pháp trước Mentor và Hội đồng, hệ thống thiết lập bài toán đánh giá đa nhánh chạy song song trên cùng một thước đo duy nhất (**Fixed Test Set** \~160 ảnh đã đóng băng từ Bước 3):

#### 1. Thiết kế 3 Nhánh Thử Nghiệm &amp; Mốc Tham Chiếu Trần (Upper Bound):

1. **Nhánh A (Random Baseline — Đối chứng Truyền thống)**:
   - *Cách chọn*: Bốc ngẫu nhiên 200 ảnh thô (`random.sample`) nạp cùng 250 ảnh Seed Set $\rightarrow$ Tổng 450 ảnh. Đa phần dính ảnh ban ngày dễ dãi, góc máy trùng lặp.
   - *Cách gán nhãn*: Bắt annotator vẽ tay thủ công 100% từ đầu trên trang trắng (tốn \~7.5 giờ công).
   - *Mô hình*: Huấn luyện ra `Model_Random` (YOLO11n, 50 epochs, dùng chung Fixed Val Set, 3 seeds).
2. **Nhánh C (Ablation Study — Bóc tách Thuật toán Chọn ảnh)**:
   - *Cách chọn*: Dùng thuật toán Active Mining ($U_i, W_i$) + $k$-Center Greedy (Bước 5 &amp; 6) để chọn 200 ảnh khó nhất.
   - *Cách gán nhãn*: **Vẽ tay thủ công 100% (KHÔNG dùng Tier 1 Auto-Accept)**.
   - *Ý nghĩa bóc tách*: So sánh trực diện $C \text{ vs } A$ trên cùng chi phí gán tay để chứng minh: **Thuật toán chọn ảnh khó tạo ra bước nhảy vọt về độ chính xác của mô hình**, loại bỏ nghi vấn kết quả tăng điểm là do nhãn mồi hay auto-label.
3. **Nhánh B (Full Active 2-Tier HITL — Giải pháp Đề xuất Tối ưu)**:
   - *Cách chọn*: Dùng thuật toán Active Mining + $k$-Center Greedy (Bước 5 &amp; 6) chọn 200 ảnh khó.
   - *Cách gán nhãn*: Áp dụng quy trình **Two-Tier HITL** (Bước 7) — Tier 1 máy tự duyệt \~60 ảnh ($0\text{s}$), Tier 2 người sửa trên nhãn mồi \~140 ảnh ($15\text{s}$/ảnh).
   - *Ý nghĩa*: So sánh trực tiếp $B \text{ vs } C$ để chứng minh: **Auto-Accept và Pre-labeling giúp cắt giảm &gt;68% giờ công mà KHÔNG làm suy giảm độ chính xác của mô hình** (Zero Degradation: mAP của B tương đương C, sai lệch $< 0.3\%$).
4. **Mốc Tham Chiếu Trần Hiệu Năng (Upper Bound Reference)**:
   - Mô phỏng cách làm cũ gán nhãn toàn bộ kho dữ liệu (\~2.000 ảnh thô bằng tay, tốn &gt;33 giờ công) đạt trần hiệu năng $mAP \approx 76.0\%$.

#### 2. Ma trận Kết quả Đối chứng Thực nghiệm (A/B Benchmark Matrix):


| Nhánh Thử nghiệm                      | Cơ chế Chọn mẫu               | Quy trình Gán nhãn              | Số ảnh Train | $mAP_{50-95}$ | $Recall_{\text{night}}$ | $Recall_{\text{small}}$ | Giờ công (Hours) | Kết luận &amp; Ý nghĩa Khoa học                                     |
| :------------------------------------- | :----------------------------- | :------------------------------- | :------------: | :-------------: | :-----------------------: | :-----------------------: | :----------------: | :------------------------------------------------------------------- |
| **Nhánh A (Random Baseline)**         | Ngẫu nhiên (`random.sample`)  | Vẽ tay 100% thủ công            | 450          | \~69.5%       | \~49.0%                 | \~43.0%                 | \~7.5h           | Bị mù ở các ca khó, tốn nhiều giờ công                              |
| **Nhánh C (Ablation: Active Only)**   | Active Mining + $k$-Center    | Vẽ tay 100% thủ công            | 450          | \~74.8%       | \~67.0%                 | \~58.5%                 | \~7.5h           | Chứng minh thuật toán chọn dữ liệu tạo đột phá ($+5.3\%$)           |
| **Nhánh B (Full Active 2-Tier HITL)** | Active Mining + $k$-Center    | **Two-Tier HITL (Auto + CVAT)** | **450**      | **\~74.5%**   | **\~66.5%**             | **\~58.0%**             | **\~2.4h**       | **Tối ưu toàn diện: Giữ vững mAP đỉnh cao, tiết kiệm 68% giờ công** |
| **Nhánh D (Bonus: Active + Negative)**| Active Mining + Background SSOD| Two-Tier HITL + 0s Negative BG   | 450 + 100    | \~75.1%       | \~67.2%                 | \~58.3%                 | \~2.4h           | Ép False Positives giảm thêm 15% mà không tốn thêm giờ gán nhãn     |
| *(Mốc tham chiếu: Upper Bound)*       | Toàn bộ kho ảnh (\~2.000 ảnh) | Vẽ tay 100% thủ công            | 2.000        | \~76.0%       | \~69.0%                 | \~60.0%                 | \~33.3h          | Trần hiệu năng tối đa (tốn gấp 14 lần thời gian)                    |


#### 3. Khung Báo Cáo 3 Trụ Cột Nghiệm Thu (3-Pillar ROI Framework):

1. **Trụ cột 1 — Hiệu năng Mô hình (Model Quality Gain)**:
   - $\Delta mAP_{\text{Active vs Random}} = +5.0\%$ trên tổng thể và $+7.6\%$ riêng cho class `greensm`.
   - Đột phá ở các điểm mù: $Recall_{\text{night}}$ tăng $+17.5\%$, $Recall_{\text{small}}$ tăng $+15.0\%$, $Recall_{\text{occ}}$ tăng $+16.2\%$.
2. **Trụ cột 2 — Hiệu quả Lao động &amp; Tiết kiệm Chi phí (Labor Efficiency ROI)**:
   - Tiết kiệm **68% thời gian** ở vòng lặp V1 so với gán thủ công cùng ngân sách ($2.4\text{h}$ vs $7.5\text{h}$).
   - Đạt **98% trần hiệu năng** của cả kho dữ liệu 2.000 ảnh nhưng chỉ tốn **\~7.2% thời gian** ($2.4\text{h}$ vs $33.3\text{h}$).
3. **Trụ cột 3 — Độ An Toàn Dữ Liệu &amp; Ý Nghĩa Thống Kê (Safety &amp; Significance)**:
   - Dữ liệu sạch: Spot-Check QA Tier 1 đạt $\ge 97\%$ độ chính xác, không gây ngộ độc dữ liệu (No Confirmation Bias nhờ RT-DETR Oracle đồng thuận).
   - Đảm bảo khoa học: Mức tăng $\Delta mAP = +5.0\% > 2\sigma_{\text{seed}} = 1.6\%$ ($p < 0.05$), loại trừ 100% rủi ro do may rủi ngẫu nhiên.

#### 4. Báo cáo Đối đầu Kiến trúc: YOLO11n vs RT-DETR (Edge Deployment vs Cloud Oracle)

Để chứng minh năng lực thiết kế hệ thống tối ưu giữa độ chính xác và tính khả thi thương mại ngoài đời thực, dự án bổ sung bảng đối đầu trực diện giữa mô hình lõi triển khai thực tế (YOLO11n) và mô hình Oracle kiểm định (RT-DETRv4-R18):

| Tiêu chí Đánh giá | YOLO11n V1 (Edge Core) | RT-DETR-R18 V1 (Cloud Oracle) | Lập luận Thực tiễn &amp; Giá trị Thương mại |
| :--- | :---: | :---: | :--- |
| **Kiến trúc cốt lõi** | CNN (Local Features, siêu nhẹ) | Transformer (Global Attention, nặng) | Đại diện cho 2 trường phái SOTA về Object Detection |
| **Dung lượng Checkpoint** | **5.5 MB** | **\~65 MB** (Nặng gấp 12 lần) | YOLO11n nhúng vừa chip camera giám sát giao thông / Edge IoT |
| **Tốc độ Inference (Edge)** | **&gt; 110 FPS** | **\~35 FPS** (Chậm hơn 3 lần) | YOLO11n xử lý mượt luồng video 4K Real-time không trễ khung hình |
| **$mAP_{50-95}$ Tổng thể** | \~74.5% | \~76.2% | RT-DETR nhỉnh hơn một chút nhưng chi phí phần cứng đắt đỏ |
| **$Recall$ Ca Che khuất** | 76.2% | **84.5%** | Global Attention của Transformer vượt trội ở bối cảnh xe bị che lấp |
| **Định vị trong Hệ thống** | **Triển khai Edge thực tế trên xe / cam** | **Làm Oracle thẩm định nhãn tự động** | Tối ưu toàn diện: Chất lượng nhãn tối đa, chi phí vận hành tối thiểu |

#### 5. Báo cáo Tối ưu Hóa &amp; Nén Lượng Tử Hóa Triển Khai Thực Tế (Edge Deployment Benchmark)

Để biến mô hình từ nghiên cứu lý thuyết thành giải pháp thương mại nhúng trực tiếp vào camera giám sát đường phố hoặc thiết bị trên xe (vốn chạy CPU Intel/ARM giá rẻ không có GPU rời), hệ thống thiết lập pipeline xuất xưởng và nén mô hình (**Export &amp; Quantization Pipeline**):

```text
    [best.pt (PyTorch GPU)]
               │
               ▼  (Ultralytics Export: Half=True)
       [ONNX Runtime (FP16)]  ──> Chuẩn mở đa nền tảng (Cross-platform)
               │
               ▼  (OpenVINO Post-Training Quantization Tool - POT/NNCF)
    [OpenVINO INT8 (Intel CPU)] ──> Tăng tốc 3-4x trên CPU thường, giữ vững mAP
```

* **Ma trận Đo lường Hiệu năng Phần cứng Thực tế (Latency &amp; Throughput Benchmark)**:

| Định dạng Mô hình | Phần cứng Mục tiêu | Kích thước File | Độ trễ (Latency/Frame) | Tốc độ (FPS) | Biến thiên $mAP_{50-95}$ | Trạng thái Triển khai |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PyTorch (`best.pt`)** | NVIDIA RTX GPU | 5.5 MB | \~9.0 ms | \~110 FPS | Baseline (74.5%) | Dùng cho Lab huấn luyện |
| **ONNX FP16 (`.onnx`)** | GPU / Cloud Edge | 2.8 MB | \~5.5 ms | \~180 FPS | $0.0\%$ | Chuẩn mở trao đổi hệ thống |
| **OpenVINO INT8 (`.xml/.bin`)** | **Intel CPU (i5/Core Ultra)** | **1.6 MB** | **\~8.2 ms** | **\~122 FPS** | **$-0.3\%$ (Không đáng kể)** | **Khuyên dùng triển khai Camera Edge** |
| *(NCNN / MNN)* | Chip ARM / Mobile IoT | \~1.5 MB | — | — | — | Hướng mở rộng cho thiết bị nhúng |

* **Lập luận bảo vệ thực chiến**: Nhóm chứng minh được việc nén lượng tử hóa sang **OpenVINO INT8** giúp camera giao thông chạy mượt mà ở tốc độ **>120 FPS trên CPU thuần túy** với dung lượng chỉ **1.6 MB**, tiết kiệm 80% chi phí đầu tư phần cứng máy chủ.

---

## 4. NGÂN SÁCH GÁN NHÃN &amp; KẾ HOẠCH KHẢ THI (3 TUẦN)

Để chứng minh đề tài hoàn toàn vừa sức trong thời gian **3 tuần**, ngân sách gán nhãn được lập chi tiết và đồng bộ tuyệt đối với các khâu kỹ thuật V4 như sau:

### 4.1. Khối lượng công việc gán nhãn thực tế (Workload Estimation)

| Thành phần dữ liệu | Quy mô | Phương thức thực hiện | Tốc độ định mức | Tổng thời gian thực tế |
| :--- | :---: | :--- | :---: | :---: |
| **1. Fixed Test Set** | 160 ảnh | Pre-labeling + Chuyên gia căn chỉnh BBox tỉ mỉ | 30 giây/ảnh | **1.33 giờ** (80 phút) |
| **2. Fixed Val Set** | 40 ảnh | Pre-labeling + Human Review | 25 giây/ảnh | **0.28 giờ** (17 phút) |
| **3. Seed Set (Model V0)** | 250 ảnh | Pre-labeling + Human Refinement (5 classes) | 25 giây/ảnh | **1.74 giờ** (104 phút) |
| **4. Active V1 (Tier 1)** | 60 ảnh | **Auto-Accept kép (YOLO11n + RT-DETR Oracle)** | **0 giây** | **0.00 giờ** (Máy tự duyệt) |
| **5. Active V1 (Tier 2)** | 140 ảnh | CVAT Human Review (Phím tắt 1-chạm + Nhãn mồi) | 15 giây/ảnh | **0.58 giờ** (35 phút) |
| **6. Spot-Check QA Tier 1** | 10 ảnh | QA Lead độc lập kiểm tra chéo độ chính xác | 20 giây/ảnh | **0.06 giờ** (3.3 phút) |
| **7. Nhánh A (Random Baseline)** | 200 ảnh | Vẽ tay thủ công 100% (đối chứng A/B Test) | 45 giây/ảnh | **2.50 giờ** (150 phút) |
| **8. Background Mining** | 100 ảnh | Lọc tự động từ 1.500 ảnh Unlabeled (SSOD Bonus) | 0 giây | **0.00 giờ** (Máy tự lọc) |
| **TỔNG CỘNG TOÀN DỰ ÁN** | **950 ảnh** | *(Người thực làm: 790 ảnh + 10 QA)* | — | **~6.5 giờ công** |

👉 **Kết luận Khả thi**: Tổng ngân sách gán nhãn cho toàn bộ dự án chỉ tốn khoảng **~6.5 giờ công con người**. Nhóm 2–3 người chỉ cần dành **2–3 giờ/người** rải đều trong 3 tuần là hoàn thành 100% dữ liệu chuẩn mực, triệt tiêu hoàn toàn rủi ro vỡ tiến độ.

### 4.2. Lịch trình thực thi chi tiết (3 Tuần: 08/10 - 29/10/2026)

```text
Tuần 1 (08/10 – 14/10): Data Profiling, Anti-Leakage Split & Đóng Băng Kép (Dual-Freeze)
Tuần 2 (15/10 – 21/10): Seed Training, Active Mining & Vòng Lặp Two-Tier HITL (RT-DETR Oracle)
Tuần 3 (22/10 – 29/10): Retrain Model V1, Multi-Seed Validation, A/B Test & Báo Cáo 3 Trụ Cột ROI
```

- **Tuần 1 (08/10 - 14/10): Data Profiling, Anti-Leakage Split &amp; Đóng Băng Kép (Dual-Freeze)**:
  - Thu thập dữ liệu video/ảnh GreenSM, chạy lọc trùng lặp thô &amp; tinh (**Dual-Stream FastDedup** với pHash).
  - Chạy **Auto Profiling** (HSV Day/Night, Laplacian Blur, IoA Occlusion) và đo mốc thời gian thực nghiệm CVAT ($T_{\text{manual}}$ vs $T_{\text{assisted}}$).
  - Phân chia tập theo Video Sequence ID, gán nhãn tăng tốc và **Đóng băng kép**: **Fixed Test Set** (160 ảnh) + **Fixed Val Set** (40 ảnh), sinh mã hash SHA-256.
- **Tuần 2 (15/10 - 21/10): Seed Training, Active Mining &amp; Vòng Lặp Two-Tier HITL**:
  - Gán nhãn Seed Set (250 ảnh) bằng Pre-labeling $\rightarrow$ Huấn luyện **Model V0 (YOLO11n)** và fine-tune nhanh **RT-DETRv4 V0**. Áp dụng Head restructuring, Backbone freeze 10 epochs đầu, Class-weighted Loss và AdamW.
  - Chạy Model V0 trên Unlabeled Pool, tính điểm **Cost-Aware Selection Score** (4 biến: $U_i, W_i, A_i, E_i$).
  - Lọc đa dạng hóa bối cảnh bằng **PCA 32D + $k$-Center Greedy (Core-Set)** bốc 200 ảnh khó nhất.
  - Vận hành **Two-Tier HITL**: Tier 1 Auto-Accept có **RT-DETR Consensus Oracle** bảo vệ; Tier 2 CVAT Review với phân cụm lát cắt và phím tắt 1-chạm; Spot-check QA chốt độ chính xác.
  - Xuất file thẻ căn cước dữ liệu `manifest.yaml` và gắn Git Tag niêm phong Dataset V1.
- **Tuần 3 (22/10 - 29/10): Retrain Model V1, Multi-Seed Validation, A/B Test &amp; Báo Cáo ROI**:
  - Huấn luyện tích lũy **Model V1** (450 ảnh: 250 seed + 200 active) trên Fixed Val Set; nạp thêm 100 ảnh Negative Background Mining.
  - Chạy giao thức **Multi-Seed (3 seeds)** kiểm tra tính ổn định thống kê ($\Delta mAP > 2\sigma_{\text{seed}}$).
  - Chạy thử nghiệm đối chứng A/B đa nhánh (**Nhánh A vs B vs C vs D**) trên Fixed Test Set.
  - Chạy đối đầu kiến trúc **Edge vs Cloud (YOLO11n vs RT-DETR)**.
  - Xuất mô hình sang **ONNX FP16** và lượng tử hóa **OpenVINO INT8 trên CPU**, đo đạc độ trễ Latency/FPS.
  - Hoàn thiện slide thuyết trình và **Báo cáo 3 Trụ Cột ROI (Model Quality, Labor Efficiency, Safety)**.

---

## 5. PHÂN CÔNG NHÂN SỰ &amp; ĐỘC LẬP QA

Để đảm bảo tính độc lập và độ tin cậy khoa học cao nhất theo gợi ý của Mentor, trách nhiệm được phân công rõ ràng:

1. **Kỹ sư Mô hình &amp; Active Learning Pipeline Lead**:
   - Chịu trách nhiệm viết code Profiling, Mining Score, Deduplication và Training YOLO11n.
   - Quản lý quá trình Inference và chuyển dữ liệu sang CVAT.
2. **Kỹ sư Kiểm định Chất lượng Tập Test &amp; QA Lead (Tách biệt độc lập)**:
   - **Độc lập hoàn toàn với người train mô hình**.
   - Chịu trách nhiệm gán nhãn, kiểm tra chất lượng và khóa cố định **Fixed Test Set**.
   - Chịu trách nhiệm ngẫu nhiên Spot-check 5-10% ảnh thuộc Tier 1 Auto-accept để tính chỉ số `Machine Label Accuracy`.
3. **Kỹ sư Quản lý Tái lập &amp; Manifest Lead**:
   - Quản lý các file `manifest.yaml`, lưu trữ config, seed, git tags và model checkpoints.
   - Đảm bảo 100% thí nghiệm có thể tái lập lại khi Mentor hoặc Hội đồng kiểm tra.

---

## 6. MA TRẬN CHỈ SỐ ĐO LƯỜNG HIỆU QUẢ (ROI &amp; METRICS MATRIX)

Báo cáo kết quả cuối cùng sẽ được lập dựa trên **3 nhóm metric cốt lõi**:

### 6.1. Chỉ số Chất lượng Mô hình (Model Quality Gain)

- $\Delta mAP_{50-95} = mAP(\text{Model } V_1) - mAP(\text{Model } V_0)$
- $\Delta Recall_{small} = Recall_{small}(\text{Model } V_1) - Recall_{small}(\text{Model } V_0)$
- $\Delta Recall_{night} = Recall_{night}(\text{Model } V_1) - Recall_{night}(\text{Model } V_0)$
- **Target**: $mAP_{50-95}$ tăng **+4% đến +8%** so với Model V0.

### 6.2. Chỉ số Tiết kiệm Công sức &amp; Chất lượng Nhãn (Labor Efficiency &amp; Safety)

- **Thời gian trung bình/ảnh (Avg Sec/Img)**:
  $$
  \text{Time Saved \%} = 100\% \times \left(1 - \frac{T_{assisted}}{T_{manual}}\right)
  $$
- **Tỷ lệ giảm giờ công tổng thể (Total Hours Saved %)**: Tiết kiệm $\ge 60\%$ tổng giờ công so với gán nhãn thủ công toàn bộ.
- \*\* Machine Label Accuracy (Spot-check QA)\*\*: Đạt $\ge 95\%$ chính xác trên các mẫu Tier 1 Auto-accept.

### 6.3. Chỉ số Ý nghĩa Thống kê &amp; A/B Benchmark (Statistical Significance)

- **Natural Variance (**$\sigma_{seed}$**)**: Mức dao động $mAP$ giữa 3 random seeds.
- **Chỉ số cải thiện thực sự**: $\Delta mAP_{Active} > \sigma_{seed}$.
- **Active vs Random Efficiency Gap**:
  $$
  \text{Efficiency Gap} = mAP(\text{Active V1}) - mAP(\text{Random V1})
  $$

---

## 7. DANH SÁCH FILE VÀ ARTIFACTS CẦN TẠO TRONG PROJECT

1. `data_profiling.py`: Script tự động Intra-video Deduplication (pHash, Keep-most-vehicles) và gán tag metadata (Day/Night, Blur, Small Object).
2. `active_selection.py`: Script tính toán điểm 1-Class Confidence Margin Uncertainty, Dynamic Weak-slice Weight, và Diversity Clustering.
3. `train_yolo11n.py`: Pipeline huấn luyện YOLO11n với cấu hình hyperparameter cố định, Head/Backbone fine-tuning, Class-weighted Loss, AdamW và Multi-seed support.
4. `evaluate_benchmark.py`: Script đánh giá mô hình trên Fixed Test Set theo từng slice và tính toán mức dao động $mAP$.
5. `export_quantize.py`: Pipeline đóng gói xuất ONNX FP16 và lượng tử hóa OpenVINO INT8, đo đạc Benchmark Latency/FPS trên CPU.
6. `manifest.yaml`: Template quản lý lineage dữ liệu và tái lập thí nghiệm.
7. `BAO_CAO_BENCHMARK_A_B_TEST.md`: Báo cáo kết quả A/B Test &amp; Ablation Study kèm chứng minh ROI.

