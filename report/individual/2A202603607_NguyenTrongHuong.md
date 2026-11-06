# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603607***
- **Họ và tên: Nguyễn Trọng Hưởng**
- **SEED** (in ở ô đầu notebook): 933

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

| Phương pháp | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.9171 | 0.9474 | 0.07 ms | $0.00 |
| **2. BoW + MLP (deep learning)** | 0.9681 | 1.0000 | 2.46 ms | $0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.8952 | 0.9268 | 56.00 ms | $0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.9637 | 0.9474 | 142.13 ms | $0.00 |
| **3c. Encoder (MiniLM) + MLP head** | 0.9479 | 0.9474 | 134.51 ms | $0.00 |
| **4a. LLM nhỏ zero-shot** | 1.0000 | 1.0000 | 1,036.69 ms | $0.0060 |
| **4b. LLM lớn zero-shot** | 1.0000 | 1.0000 | 1,522.52 ms | $0.2407 |
| **4c. LLM nhỏ few-shot** | 1.0000 | 1.0000 | 953.38 ms | $0.0176 |

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

1. **`dl_curve` (Đặc trưng BoW kết hợp Mạng Nơ-ron MLP):**
   * **Dự đoán:** Val loss sẽ sớm bị overfitting và F1 của MLP sẽ chỉ xấp xỉ bằng hoặc nhỉnh hơn Logistic Regression đôi chút vì vector BoW dạng thưa không hưởng lợi nhiều từ mạng nơ-ron sâu.
   * **Thực tế:** F1 của BoW + MLP đạt tới **0.9681** trên test (và **1.0000** trên SUB 40), vượt trội hoàn toàn so với BoW + LogReg (**0.9171**).
   * **Vì sao lệch:** Mặc dù mất thứ tự từ, không gian biểu diễn ẩn của MLP (2 lớp Linear kết hợp phi tuyến ReLU) vẫn học và kết hợp được các tương tác chéo (co-occurrence) giữa các từ khóa rất mạnh mẽ, giúp phân ranh giới quyết định tốt hơn hẳn bộ phân loại tuyến tính đơn thuần.

2. **`llm_models` (So sánh hiệu năng giữa các dòng LLM):**
   * **Dự đoán:** Model Opus 5.5 sẽ vượt trội hơn Haiku 5.5 khoảng 10 điểm F1-score.
   * **Thực tế:** Cả LLM nhỏ (Haiku) và LLM lớn (Opus) đều đạt F1 tuyệt đối **1.0000** trên tập đánh giá.
   * **Vì sao lệch:** Tác vụ phân loại nhị phân tin nhắn rác không đòi hỏi suy luận logic đa bước quá phức tạp. Khả năng đọc hiểu ngữ cảnh của một mô hình ngôn ngữ nhỏ như Haiku đã thừa sức giải quyết trọn vẹn tập test thông thường.

3. **`hard_eval` (Dự đoán độ tụt điểm trên tập 50 case khó):**
   * **Dự đoán:** Accuracy của các dòng BoW sẽ chỉ tụt nhẹ xuống mức 60–70%, còn Embedding sẽ giữ vững phong độ ở 85–95%.
   * **Thực tế:** F1 của dòng BoW tụt dốc thê thảm xuống **0.350** (BoW + LogReg) và **0.390** (BoW + MLP) với Accuracy chỉ đạt ~**52–54%**. Dòng Embedding cũng suy giảm nghiêm trọng với F1 chỉ còn **0.593 – 0.721** (Acc ~**60–69%**), cách rất xa mốc 85–95% dự đoán.
   * **Vì sao lệch:** Tập 50 case khó chứa nhiều mẫu "cố tình bẫy" từ khóa (như từ khóa spam nằm trong tin ham hoặc cố tình làm nhiễu chính tả để né lọc). BoW hoàn toàn mất phương hướng trước các bẫy từ khóa, trong khi biểu diễn của Embedding (MiniLM) vẫn bị nhiễu đáng kể bởi sự xuất hiện của các từ khóa có trọng số ngữ nghĩa mạnh.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

* **Mô hình tụt nhiều nhất:** Dòng **BoW (Bag-of-Words)** là nhóm tụt nặng nề nhất. Cụ thể, **BoW + LogReg** tụt F1 từ **0.947** xuống **0.350** (F1 giảm ~63%), và **BoW + MLP** tụt từ **1.000** xuống **0.390** (F1 giảm ~61%).
* **Nhóm case gây khó khăn nhất:**
  * **Với BoW:** Nhóm *Tưởng spam nhưng không phải* (`ham_looks_spam`) và *Né từ khoá* (`obfuscation`).
  * **Với Embedding / LLM:** Nhóm *Prompt injection* (đối với LLM) và *Lừa đảo xã hội / Thao túng ngữ nghĩa* (đối với Embedding).

* **2 case cụ thể và cơ chế khiến model sai:**
  1. **Case #9 (Nhóm: Tưởng spam nhưng không phải):**
     * *Văn bản:* `"Txt me ur address and I'll send the free tickets my boss gave me for the concert"` (Nhãn thật: **ham**).
     * *Cơ chế sai:* Model BoW nhìn thấy các token tần suất cao gắn chặt với tin rác như `txt`, `free`, `tickets` nên kích hoạt trọng số dương cực lớn, dẫn đến việc kết luận sai thành **spam**. Thậm chí cả mô hình LLM có phòng thủ cũng có thể bị phân vân ở trường hợp này do văn cảnh tặng vé miễn phí rất sát với kịch bản spam trúng thưởng.
  2. **Case dạng biến đổi ký tự né lọc (Nhóm: Né từ khoá):**
     * *Văn bản:* `"Urgent: your 0-T-P is 5821. C_a_l_l us now to change security settings"` (Nhãn thật: **spam**).
     * *Cơ chế sai:* Kẻ phát tán sử dụng dấu gạch ngang và dấu gạch dưới chèn giữa các ký tự (`0-T-P`, `C_a_l_l`). Bộ tách từ (tokenizer) của BoW tách thành các token độc lập (`0`, `T`, `P`, `C_a_l_l`) không nằm trong bảng từ vựng đã học, khiến vector đầu vào gần như bằng 0 và mô hình bỏ lọt spam (dự đoán nhầm thành **ham**).

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

* **Đề nhận được (SEED: 933, Challenge #4):**  
  *Embedding · Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?*

* **Cách làm:**  
  Chạy hàm `knn_predict(E_tr, y_tr, E_te, k=k_val)` trên tập embedding test với lần lượt $k \in \{1, 5, 25\}$, sau đó tính toán Accuracy, Precision, Recall và F1-score qua hàm `metrics(y_te, preds)`.

* **Kết quả thực nghiệm:**
  * **$k = 1$:** Accuracy = `0.9729`, Precision = `0.8393`, Recall = `0.9691`, **F1-Score = 0.8995**
  * **$k = 5$:** Accuracy = `0.9716`, Precision = `0.8319`, Recall = `0.9691`, **F1-Score = 0.8952**
  * **$k = 25$:** Accuracy = `0.9690`, Precision = `0.8288`, Recall = `0.9485`, **F1-Score = 0.8846**

* **Giải thích cơ chế:**
  * **Khi $k$ quá nhỏ ($k = 1$):** Mô hình rơi vào trạng thái phương sai cao (**High Variance / Overfitting**). Ranh giới phân chia rất nhạy cảm; chỉ cần một mẫu nhiễu hoặc điểm bị gán nhãn sai nằm sát mẫu kiểm thử cũng khiến mô hình đổi kết quả dự đoán.
  * **Khi $k$ quá lớn ($k = 25$):** Mô hình rơi vào trạng thái chệch cao (**High Bias / Underfitting**). Do tập dữ liệu mất cân bằng nghiêm trọng (~87% là ham), khi mở rộng bán kính tìm kiếm đến 25 láng giềng, các phiếu bầu sẽ bị áp đảo bởi lớp đa số (ham), dẫn đến ranh giới cục bộ bị làm phẳng, gây bỏ sót nhiều tin rác (Recall giảm từ `0.9691` xuống `0.9485`).

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

* **Yêu cầu hệ thống:** Lọc 10 triệu SMS/ngày (trung bình ~116 tin/giây), ràng buộc chất lượng dịch vụ **$p_{95} < 50\text{ ms/tin}$**.

* **Lựa chọn phương án:**  
  Áp dụng mô hình **Kiến trúc Lai (Router 2 tầng)** kết hợp giữa **BoW + MLP** (hoặc **Encoder MiniLM + LogReg**) và **LLM API**:
  * **Tầng 1 (Local Model - Tuyến đầu):** Sử dụng mô hình cục bộ phân loại xác suất $p$.
    * Nếu $p < 0.2$ (chắc chắn là tin nhắn thường) hoặc $p > 0.8$ (chắc chắn là spam): Xử lý ngay lập tức.
  * **Tầng 2 (LLM Fallback - Vùng lưỡng lự):** Chỉ chuyển tiếp các tin nhắn khó nằm trong vùng mập mờ ($0.2 \le p \le 0.8$) sang LLM nhỏ (như Haiku 5.5).

* **Dẫn chứng số liệu từ bài Lab:**
  * **Về độ trễ:** Phương án BoW + LogReg ($p_{95} = 0.07\text{ ms}$) và BoW + MLP ($p_{95} = 2.46\text{ ms}$) đáp ứng vượt mức cam kết SLA ($< 50\text{ ms}$). Ngược lại, nếu chỉ dựa vào LLM ($p_{95} > 1,000\text{ ms}$), hệ thống sẽ vi phạm nghiêm trọng SLA do độ trễ truyền tải mạng và thời gian sinh token.
  * **Về chi phí:** Với 10 triệu tin/ngày (khoảng 100 token input, 1 token output/tin):
    * Dùng thuần LLM (Haiku): Tốn khoảng **$105/ngày** $\approx$ **$3,150/tháng** (~75–80 triệu VND/tháng).
    * Dùng mô hình lai: >90% lưu lượng tin được xử lý cục bộ với chi phí điện toán gần như $0, chỉ ~5–10% tin khó được chuyển tiếp qua LLM, giảm chi phí vận hành hơn 90% (chỉ còn dưới $300/tháng).

* **Chấp nhận đánh đổi:**  
  Chấp nhận đánh đổi việc bỏ sót hoặc phân loại sai một tỷ lệ cực nhỏ các tin nhắn tinh vi ở Tầng 1 để đổi lấy việc bảo toàn SLA thời gian thực ($< 50\text{ ms}$) và tối ưu hóa hàng nghìn USD chi phí vận hành mỗi tháng.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| :--- | :--- | :--- | :--- |
| **Phần 1: Baseline** | *"Viết hàm vectorized và tính cross-entropy loss cho Logistic Regression trên PyTorch"* | Cung cấp hàm nén BoW và vòng lặp huấn luyện chuẩn sử dụng hàm mất mát `BCEWithLogitsLoss`. | Tinh chỉnh lại hàm tiền xử lý tokenizer để loại bỏ bớt ký tự đặc biệt, đối chiếu kiểm tra qua hàm `check_1_1` và `check_1_3`. |
| **Phần 3: kNN** | *"Cách vector hóa cosine similarity nhanh giữa 2 ma trận bằng PyTorch hoặc NumPy"* | Đưa ra công thức chuẩn hóa norm ma trận rồi nhân `A @ B.T`. | Kiểm tra lại kích thước ma trận `(N, D) x (M, D)` để tránh tràn RAM bộ nhớ GPU/CPU khi tập train phình to. |
| **Phần 4: LLM** | *"Thiết kế prompt phòng thủ chống prompt injection cho bài toán spam filter"* | Gợi ý prompt đặt nội dung SMS vào thẻ `<sms>...</sms>` và chỉ thị rõ không tuân theo các câu lệnh bên trong văn bản. | Thử nghiệm trực tiếp trên tập 50 case khó (mục 4d) để đối chiếu mức độ suy giảm accuracy của injection. |

* **Điều hiểu ra hôm nay mà trước đó chưa hiểu:**  
  Điểm số F1 cao ngất ngưởng trên tập test ngẫu nhiên dễ tạo ra sự "lạc quan giả tạo" vì dữ liệu test rơi vào cùng phân phối quen thuộc với tập train; chỉ khi đối mặt với các tình huống adversarial (như tập 50 case khó), các lỗ hổng của việc bám vào từ khóa rời rạc mới bộc lộ rõ nét. Ngoài ra, việc triển khai AI trong thực tế là bài toán cân bằng giữa điểm hiệu năng (F1), độ trễ phản hồi (latency SLA) và chi phí tài chính (cost/token) chứ không chỉ đơn thuần là chọn mô hình lớn nhất.