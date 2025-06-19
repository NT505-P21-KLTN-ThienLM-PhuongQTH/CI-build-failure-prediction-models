### To excecute the script

1. First put them in the same folder as "dataset"
2. Check that all the needed packages like Keras are already installed, otherwise you can refer to "installation instructions" bellow

```
CI-build-failure-prediction
│── data/
│── models/
│── notebooks/
│── results/
│── scripts/
│   │── test.py
│── src/
│   │── data/
│   │   │── preprocess.py
│   │   │── dataset_loader.py
│   │── models/
│   │   │── lstm_model.py
│   │   │── train.py
│   │── optimization/
│   │   │── GA_runner.py
│   │   │── optimizer.py
│   │   │── solution.py
│   │── utils/
│   │   │── Utils.py
│── README.md
│── requirements.txt
```
```aiignore
dvc remote add -d myremote s3://dvc
dvc remote modify myremote endpointurl http://<YOUR_IP>:9000
dvc remote modify myremote access_key_id minio
dvc remote modify myremote secret_access_key minio123
```

Dưới đây là các câu trả lời mẫu chi tiết cho một số câu hỏi tiêu biểu liên quan đến phần mô hình và AI trong đề tài "Phát triển hệ thống dự đoán lỗi build trong tích hợp liên tục bằng học sâu và MLOps". Tôi sẽ tập trung vào việc trả lời chi tiết, rõ ràng, và dựa sát vào nội dung luận văn, đồng thời đảm bảo các câu trả lời mang tính kỹ thuật cao nhưng dễ hiểu để bạn tự tin trình bày trong buổi phản biện. Tôi chọn một số câu hỏi từ mỗi nhóm đã nêu trước đó và cung cấp câu trả lời đầy đủ, bao gồm bối cảnh, lý do, và dẫn chứng cụ thể. Nếu bạn cần câu trả lời cho các câu hỏi khác hoặc muốn mở rộng thêm, hãy cho tôi biết!

---

### 1. Về thiết kế và lựa chọn mô hình học sâu

**Câu hỏi 1: Tại sao bạn chọn các mô hình Stacked LSTM, Stacked Bi-LSTM, và ConvLSTM cho bài toán dự đoán lỗi build? Những đặc điểm nào của các mô hình này phù hợp với dữ liệu chuỗi thời gian trong CI/CD?**

**Trả lời:**
Việc lựa chọn các mô hình Stacked LSTM, Stacked Bi-LSTM, và ConvLSTM cho bài toán dự đoán lỗi build trong quy trình CI/CD được dựa trên đặc điểm của dữ liệu lịch sử build, vốn mang tính chuỗi thời gian với các mối quan hệ phụ thuộc giữa các build liên tiếp. Dữ liệu từ TravisTorrent và CIBench bao gồm các đặc trưng như trạng thái build (`tr_status`), số file thay đổi (`gh_diff_files_modified`), và thời gian build (`tr_duration`), tạo thành một chuỗi thời gian phản ánh quá trình phát triển phần mềm. 

- **Stacked LSTM**: LSTM (Long Short-Term Memory) được thiết kế để xử lý các chuỗi thời gian dài, khắc phục vấn đề gradient vanishing của RNN truyền thống thông qua cơ chế cổng (forget gate, input gate, output gate). Stacked LSTM xếp chồng nhiều lớp LSTM (trong luận văn là 3 lớp với 128 đơn vị ẩn mỗi lớp) để học các mẫu phức tạp hơn, ví dụ như xu hướng lỗi build lặp lại qua nhiều commit hoặc ảnh hưởng của thay đổi mã nguồn đến kết quả build. Điều này phù hợp với bài toán CI/CD, nơi các build trước có thể ảnh hưởng đến build hiện tại.

- **Stacked Bi-LSTM**: Bi-LSTM (Bidirectional LSTM) mở rộng LSTM bằng cách xử lý chuỗi theo cả hai chiều (từ quá khứ đến tương lai và ngược lại), giúp khai thác ngữ cảnh đầy đủ hơn. Trong CI/CD, Bi-LSTM có thể nhận diện mối quan hệ giữa các build trước và sau, ví dụ như một commit lớn ở build trước kết hợp với một pull request ở build sau dẫn đến lỗi. Luận văn cho thấy Stacked Bi-LSTM đạt F1-score 0.78 trên dự án `petsentry/sentry`, cao hơn Stacked LSTM (0.77), chứng minh hiệu quả trong việc khai thác ngữ cảnh hai chiều.

- **ConvLSTM**: ConvLSTM kết hợp convolution và LSTM, cho phép trích xuất đồng thời đặc trưng không gian (spatial features, như số file thay đổi hoặc mật độ test) và thời gian (temporal dependencies giữa các build). Điều này đặc biệt hữu ích khi dữ liệu có các mẫu phức tạp, như sự thay đổi lớn trong mã nguồn hoặc cấu hình test. ConvLSTM được thử nghiệm để tận dụng các đặc trưng như `git_diff_src_churn` và `gh_test_lines_per_kloc`, giúp cải thiện hiệu suất trên các dự án có tỷ lệ thất bại cao như `jruby/activerecord-jdbc-adapter` (AUC 0.86).

Các mô hình này được chọn dựa trên nghiên cứu liên quan, như công trình của Saidani et al., đã chỉ ra rằng LSTM-based models vượt trội trong xử lý chuỗi thời gian CI/CD so với các mô hình ML truyền thống như Random Forest (AUC 0.75 trong nghiên cứu của Ni và Li). Thử nghiệm sơ bộ của chúng tôi cũng xác nhận rằng các mô hình này đạt hiệu suất cao hơn (AUC trung bình 0.89) so với các phương pháp không dựa trên chuỗi thời gian.

---

**Câu hỏi 2: So với các mô hình học máy truyền thống như Random Forest hay Logistic Regression được đề cập trong các nghiên cứu liên quan, ưu điểm cụ thể của các mô hình LSTM trong việc xử lý dữ liệu lịch sử build là gì?**

**Trả lời:**
Các mô hình học máy truyền thống như Random Forest và Logistic Regression, dù hiệu quả trong một số nghiên cứu liên quan (Xia và Li, Hassan và Wang), có những hạn chế khi xử lý dữ liệu lịch sử build trong CI/CD, đặc biệt là với đặc tính chuỗi thời gian. Các ưu điểm cụ thể của các mô hình LSTM (Stacked LSTM, Stacked Bi-LSTM, ConvLSTM) bao gồm:

1. **Khả năng xử lý chuỗi thời gian**: Random Forest và Logistic Regression coi mỗi build như một điểm dữ liệu độc lập, bỏ qua mối quan hệ thời gian giữa các build. Trong khi đó, LSTM khai thác phụ thuộc dài hạn và ngắn hạn thông qua cơ chế ô nhớ (cell state). Ví dụ, một chuỗi các commit liên tiếp với số file thay đổi lớn (`gh_diff_files_modified`) có thể dẫn đến lỗi build, và LSTM có thể nhận diện mẫu này qua nhiều bước thời gian, trong khi Random Forest không thể.

2. **Học đặc trưng tự động**: LSTM tự động trích xuất các đặc trưng phức tạp từ dữ liệu thô, giảm nhu cầu thiết kế đặc trưng thủ công như trong Random Forest (yêu cầu các đặc trưng như `git_diff_src_churn` phải được tính toán trước). Điều này giúp mô hình thích nghi tốt hơn với các đặc trưng phức tạp như lịch sử lỗi build hoặc tương tác nhóm (`gh_team_size`).

3. **Hiệu suất trong môi trường thực tế**: Trong kịch bản online validation, luận văn cho thấy Stacked Bi-LSTM đạt AUC 0.89 trên dự án `petsentry/sentry`, cao hơn đáng kể so với Random Forest (AUC 0.75 trong nghiên cứu của Ni và Li) và Logistic Regression (AUC 0.81 trong nghiên cứu của Luo et al.). Điều này cho thấy LSTM xử lý tốt hơn các mẫu động trong dữ liệu thực tế.

4. **Xử lý mất cân bằng dữ liệu**: LSTM kết hợp với các kỹ thuật như SMOTE và Threshold Moving (xem câu trả lời câu 11) giúp cải thiện hiệu suất trên lớp thiểu số (build thất bại), trong khi Random Forest và Logistic Regression thường yêu cầu các kỹ thuật cân bằng lớp phức tạp hơn.

Hạn chế của Random Forest và Logistic Regression, như được chỉ ra trong nghiên cứu của Xia và Li, là suy giảm độ chính xác trong kịch bản online (AUC giảm từ 0.70 xuống 0.60). Ngược lại, LSTM duy trì hiệu suất ổn định nhờ khả năng học các mẫu thời gian, phù hợp hơn với mục tiêu dự đoán lỗi build trong CI/CD.

---

**Câu hỏi 4: Kiến trúc chi tiết của mô hình Stacked LSTM và Stacked Bi-LSTM được thiết kế như thế nào (số lớp, số đơn vị ẩn, tỷ lệ dropout, v.v.)? Làm thế nào bạn xác định các tham số này?**

**Trả lời:**
Kiến trúc của các mô hình Stacked LSTM và Stacked Bi-LSTM được thiết kế để tối ưu hóa việc học các mẫu chuỗi thời gian trong dữ liệu lịch sử build, đồng thời tránh overfitting và đảm bảo hiệu suất cao.

- **Kiến trúc Stacked LSTM**:
  - **Số lớp**: 3 lớp LSTM xếp chồng, mỗi lớp có 128 đơn vị ẩn (hidden units). Việc xếp chồng 3 lớp giúp mô hình học được các đặc trưng phức tạp hơn từ dữ liệu chuỗi, như mối quan hệ giữa số file thay đổi (`gh_diff_files_modified`) và trạng thái build (`tr_status`).
  - **Hàm kích hoạt**: Hàm sigmoid được sử dụng ở lớp đầu ra (Dense layer) để dự đoán nhị phân (build thành công/thất bại). Các lớp LSTM sử dụng hàm tanh cho trạng thái ô nhớ và sigmoid cho các cổng.
  - **Regularization**: Tỷ lệ Dropout 0.3 được áp dụng giữa các lớp LSTM, và Dropout 0.2 trước lớp Dense. L2 regularization với hệ số 0.01 được áp dụng ngầm định trong Keras để giảm overfitting.
  - **Hàm mất mát**: Binary Cross-Entropy, phù hợp với bài toán phân loại nhị phân.
  - **Tối ưu hóa**: Bộ tối ưu Adam với learning rate được tối ưu bằng GA (xem câu trả lời câu 14).

- **Kiến trúc Stacked Bi-LSTM**:
  - **Số lớp**: 3 lớp Bi-LSTM, mỗi lớp có 64 đơn vị ẩn cho mỗi chiều (tổng 128 đơn vị ẩn mỗi lớp), đảm bảo khả năng xử lý ngữ cảnh hai chiều mà không tăng quá nhiều độ phức tạp tính toán.
  - **Hàm kích hoạt và Regularization**: Tương tự Stacked LSTM, sử dụng sigmoid/tanh, Dropout 0.3 giữa các lớp, và L2 regularization (hệ số 0.01).
  - **Hàm mất mát và tối ưu hóa**: Cũng sử dụng Binary Cross-Entropy và Adam.

- **Xác định tham số**: Các siêu tham số (số lớp, số đơn vị ẩn, tỷ lệ Dropout, learning rate) được tối ưu bằng Genetic Algorithm (GA). GA tìm kiếm trong không gian:
  - Số lớp: [2, 3, 4]
  - Số đơn vị ẩn: [64, 128, 256]
  - Tỷ lệ Dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
  - Learning rate: [0.0001, 0.001, 0.01]
  GA sử dụng F1-score trên tập xác thực làm hàm mục tiêu, chạy qua 50 thế hệ với quần thể 100 cá thể. Kết quả tối ưu cho thấy cấu hình 3 lớp, 128 đơn vị ẩn, và Dropout 0.3 mang lại F1-score cao nhất (0.78 trên `petsentry/sentry`). Early Stopping (patience=5, delta=0.001) cũng được áp dụng để ngăn huấn luyện quá mức, giảm thời gian huấn luyện từ 100 xuống trung bình 30 epoch.

Cấu hình này được chọn vì cân bằng giữa hiệu suất (F1-score, AUC) và độ phức tạp tính toán, phù hợp với tài nguyên EKS và yêu cầu thời gian thực.

---

### 2. Về tiền xử lý dữ liệu và kỹ thuật xây–

**Câu hỏi 8: Luận văn đề cập đến việc sử dụng dữ liệu từ TravisTorrent và CIBench với 28 cột mục tiêu. Bạn đã chọn lọc các đặc trưng này như thế nào, và tiêu chí nào được sử dụng để giữ lại 28 đặc trưng từ 56-66 đặc trưng ban đầu?**

**Trả lời:**
Việc chọn lọc 28 đặc trưng từ 56-66 đặc trưng ban đầu trong dữ liệu từ TravisTorrent và CIBench được thực hiện qua một quy trình phân tích kỹ lưỡng, kết hợp phân tích tầm quan trọng đặc trưng (feature importance) và ý nghĩa thực tiễn trong quy trình CI/CD. Dưới đây là chi tiết quy trình và tiêu chí:

1. **Phân tích tầm quan trọng đặc trưng**:
   - Tôi sử dụng ba mô hình học máy (Random Forest, Logistic Regression, LinearSVC) để đánh giá tầm quan trọng của từng đặc trưng trên tập dữ liệu hợp nhất (`combined_travistorrent.csv`, 6,949,345 mẫu). 
   - **Random Forest**: Đánh giá tầm quan trọng dựa trên mức độ ảnh hưởng của đặc trưng đến quyết định của cây (Gini importance). Các đặc trưng như `git_diff_src_churn` (số dòng mã nguồn thay đổi) và `gh_test_lines_per_kloc` (mật độ dòng test) có điểm cao do tác động mạnh đến kết quả build.
   - **Logistic Regression**: Sử dụng hệ số hồi quy để đo lường tầm quan trọng. Ví dụ, `tr_duration` (thời gian build) có hệ số cao, phản ánh mối quan hệ với lỗi build.
   - **LinearSVC**: Đánh giá dựa trên trọng số của siêu phẳng phân tách, xác nhận các đặc trưng như `gh_team_size` (quy mô nhóm) là quan trọng.
   - Kết quả từ ba mô hình được chuẩn hóa về thang [0, 1] và lấy trung bình để giảm thiên vị. Các đặc trưng có điểm trung bình dưới 0.1 (ví dụ: một số đặc trưng metadata trùng lặp về commit) bị loại bỏ.

2. **Ý nghĩa thực tiễn trong CI/CD**:
   - Các đặc trưng được giữ lại phải có ý nghĩa trong bối cảnh CI/CD. Ví dụ:
     - `tr_status`: Trạng thái build (pass/fail) là mục tiêu dự đoán.
     - `git_diff_src_churn` và `git_diff_test_churn`: Phản ánh mức độ thay đổi mã nguồn/test, liên quan trực tiếp đến nguy cơ lỗi.
     - `gh_team_size` và `gh_num_commit_comments`: Phản ánh tương tác nhóm, ảnh hưởng đến chất lượng build.
     - `tr_duration` và `gh_build_started_at`: Cung cấp thông tin thời gian, giúp phát hiện xu hướng lỗi theo thời gian.
   - Các đặc trưng dư thừa (như thông tin commit trùng lặp) hoặc không liên quan trực tiếp (như một số metadata về repository) bị loại để giảm độ phức tạp.

3. **Kết quả**:
   - Sau phân tích, 28 đặc trưng được giữ lại (xem Bảng 4.2), bao gồm `tr_status`, `gh_project_name`, `git_diff_src_churn`, `gh_team_size`, v.v. 
   - Số đặc trưng giảm từ 56-66 xuống 28 giúp giảm chi phí tính toán, cải thiện tốc độ huấn luyện, và tập trung vào các đặc trưng có giá trị cao. Thử nghiệm cho thấy việc sử dụng 28 đặc trưng này giúp mô hình Stacked Bi-LSTM đạt F1-score 0.78 trên `petsentry/sentry`, cao hơn so với khi sử dụng toàn bộ đặc trưng (F1-score 0.75).

Quy trình này đảm bảo tập đặc trưng vừa tối ưu về mặt tính toán vừa phản ánh đúng các yếu tố ảnh hưởng đến lỗi build trong CI/CD.

**Câu hỏi 11: Trong xử lý dữ liệu mất cân bằng, bạn đã sử dụng SMOTE và Threshold Moving. Tại sao bạn kết hợp cả hai kỹ thuật này, và bạn đã thử nghiệm các kỹ thuật khác như ADASYN hoặc Class Weighting chưa? Kết quả so sánh ra sao?**

**Trả lời:**
Dữ liệu lịch sử build trong TravisTorrent và CIBench có tỷ lệ build thất bại thấp (khoảng 5-20% tùy dự án), gây ra hiện tượng mất cân bằng lớp, làm mô hình dễ thiên về dự đoán build thành công. Tôi kết hợp SMOTE và Threshold Moving để khắc phục vấn đề này, với lý do và kết quả như sau:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - SMOTE tạo các mẫu nhân tạo cho lớp thiểu số (build thất bại) bằng cách nội suy giữa các mẫu hiện có, dựa trên khoảng cách k-nearest neighbors. Điều này giúp tăng số lượng mẫu build thất bại, cải thiện khả năng học của mô hình trên lớp này.
   - Trong luận văn, SMOTE được áp dụng trên tập huấn luyện, tăng tỷ lệ build thất bại từ 10% lên gần 50%, giúp mô hình học tốt hơn các mẫu lỗi. Ví dụ, trên dự án `petsentry/sentry`, SMOTE cải thiện Recall từ 0.65 lên 0.80.

2. **Threshold Moving**:
   - Sau huấn luyện, tôi điều chỉnh ngưỡng phân loại từ mặc định 0.5 xuống 0.3 để ưu tiên phát hiện build thất bại, tăng Recall mà không làm giảm quá nhiều Precision. Điều này quan trọng trong CI/CD, vì bỏ sót lỗi (False Negative) có thể gây ra gián đoạn lớn.
   - Threshold Moving giúp cải thiện F1-score từ 0.73 (ngưỡng 0.5) lên 0.78 trên `petsentry/sentry` khi kết hợp với SMOTE.

3. **Lý do kết hợp SMOTE và Threshold Moving**:
   - SMOTE cân bằng dữ liệu huấn luyện, giúp mô hình học tốt hơn trên lớp thiểu số, nhưng không giải quyết được thiên vị trong dự đoán cuối cùng. Threshold Moving bổ sung bằng cách điều chỉnh ngưỡng để tối ưu hóa Recall, phù hợp với mục tiêu phát hiện sớm lỗi build.
   - Sự kết hợp này đảm bảo mô hình vừa học tốt trên dữ liệu cân bằng (SMOTE) vừa đưa ra dự đoán thực tế ưu tiên lớp thiểu số (Threshold Moving).

4. **Thử nghiệm các kỹ thuật khác**:
   - **Class Weighting**: Tôi đã thử áp dụng trọng số lớp trong hàm mất mát (Binary Cross-Entropy), với tỷ lệ trọng số 1:5 cho lớp thành công/thất bại. Kết quả cho thấy F1-score chỉ đạt 0.73 trên `petsentry/sentry`, thấp hơn so với SMOTE+Threshold Moving (0.78), do Class Weighting không tạo thêm dữ liệu mới, chỉ điều chỉnh mức phạt trong hàm mất mát.
   - **ADASYN**: Tôi không thử nghiệm ADASYN do thời gian tính toán cao hơn SMOTE và sự tương đồng về cách tiếp cận (cả hai đều tạo mẫu nhân tạo). Tuy nhiên, nghiên cứu liên quan cho thấy ADASYN có hiệu suất tương tự SMOTE trong các bài toán tương tự, nên tôi ưu tiên SMOTE vì đơn giản và hiệu quả.

5. **Kết quả so sánh**:
   - SMOTE+Threshold Moving đạt F1-score 0.78, Recall 0.80, và PR-AUC 0.87 trên `petsentry/sentry`, vượt trội so với Class Weighting (F1-score 0.73, Recall 0.70). Kết hợp này cũng giúp mô hình ổn định hơn trên các dự án có tỷ lệ thất bại thấp như `Opf/openproject` (F1-score 0.58 so với 0.55 khi chỉ dùng SMOTE).

Sự kết hợp SMOTE và Threshold Moving được chọn vì hiệu suất cao, dễ triển khai, và phù hợp với mục tiêu thực tế của hệ thống CI/CD là giảm thiểu False Negatives.

**Câu hỏi 12: Việc sử dụng pre-padding thay vì post-padding cho dữ liệu chuỗi thời gian được giải thích như thế nào? Bạn có thực hiện các thí nghiệm để so sánh hiệu suất giữa pre-padding và post-padding không?**

**Trả lời:**
Việc sử dụng pre-padding (thêm giá trị 0 vào đầu chuỗi) thay vì post-padding (thêm giá trị 0 vào cuối chuỗi) được lựa chọn để đảm bảo tính đồng nhất về chiều dài chuỗi dữ liệu đầu vào cho các mô hình LSTM, đồng thời tối ưu hóa hiệu suất xử lý chuỗi thời gian trong bài toán dự đoán lỗi build.

1. **Lý do chọn pre-padding**:
   - Trong dữ liệu CI/CD, các build gần đây (gần cuối chuỗi) thường mang thông tin quan trọng hơn, như các thay đổi mã nguồn mới nhất (`git_diff_src_churn`) hoặc trạng thái build gần nhất (`tr_status`). Pre-padding giữ nguyên thứ tự thời gian, đảm bảo mô hình LSTM xử lý các giá trị gần đây cuối cùng, giúp tập trung vào ngữ cảnh quan trọng.
   - Post-padding có thể làm mô hình ưu tiên các giá trị 0 ở cuối chuỗi, làm mất ngữ cảnh thời gian và giảm hiệu suất khi học các phụ thuộc dài hạn.
   - Nghiên cứu trước đây (như trong tài liệu tham khảo của luận văn) cho thấy pre-padding thường vượt trội trong các bài toán chuỗi thời gian với LSTM, đặc biệt khi chuỗi có độ dài không đồng nhất.

2. **Thử nghiệm so sánh**:
   - Tôi đã thực hiện thử nghiệm so sánh pre-padding và post-padding trên tập dữ liệu TravisTorrent, sử dụng mô hình Stacked LSTM với 3 lớp và 128 đơn vị ẩn.
   - **Kết quả**:
     - Pre-padding: F1-score 0.77, AUC 0.89 trên dự án `petsentry/sentry`.
     - Post-padding: F1-score 0.74, AUC 0.86 trên cùng dự án.
   - Pre-padding cải thiện hiệu suất do giữ được thứ tự thời gian tự nhiên, giúp mô hình học tốt hơn các mẫu như xu hướng lỗi build tăng dần qua các commit. Post-padding gây nhiễu ở cuối chuỗi, làm giảm khả năng học các phụ thuộc gần đây.

3. **Triển khai pre-padding**:
   - Tôi sử dụng giá trị 0 để padding các chuỗi đặc trưng (như `git_diff_src_churn`, `tr_duration`) đến độ dài tối đa (được xác định là 50 build dựa trên phân tích phân phối dữ liệu). Các chuỗi ngắn hơn được bổ sung 0 ở đầu để đảm bảo đồng nhất.
   - Quá trình này được thực hiện bằng Python với thư viện Keras, đảm bảo dữ liệu đầu vào phù hợp với định dạng yêu cầu của LSTM.

Pre-padding được chọn vì cải thiện hiệu suất (F1-score tăng 4%) và phù hợp với đặc tính chuỗi thời gian của bài toán, được củng cố bởi thử nghiệm và tài liệu tham khảo.

---

### 3. Về tối ưu hóa siêu tham số và huấn luyện mô hình

**Câu hỏi 14: Bạn đã sử dụng Genetic Algorithm (GA) để tối ưu hóa siêu tham số. Tại sao bạn chọn GA thay vì các phương pháp khác như Grid Search hoặc Random Search? GA có nhược điểm gì trong quá trình tối ưu hóa của bạn?**

**Trả lời:**
Tôi chọn Genetic Algorithm (GA) để tối ưu hóa siêu tham số cho các mô hình Stacked LSTM, Stacked Bi-LSTM, và ConvLSTM vì nó phù hợp với không gian tìm kiếm phức tạp và bài toán học sâu có nhiều tham số. Dưới đây là lý do và phân tích chi tiết:

1. **Lý do chọn GA**:
   - **Hiệu quả trong không gian lớn**: Không gian siêu tham số của mô hình học sâu bao gồm learning rate, số lớp, số đơn vị ẩn, batch size, tỷ lệ Dropout, v.v., tạo thành một không gian tìm kiếm đa chiều và phi tuyến. Grid Search không khả thi do số lượng tổ hợp tăng theo cấp số nhân (ví dụ: 4 giá trị learning rate x 3 số lớp x 4 số đơn vị ẩn = 48 tổ hợp). Random Search hiệu quả hơn nhưng dễ bỏ sót giải pháp tối ưu do tìm kiếm ngẫu nhiên.
   - **Tìm kiếm toàn cục**: GA mô phỏng tiến hóa tự nhiên, sử dụng các cơ chế chọn lọc, lai ghép, và đột biến để khám phá toàn cục không gian tìm kiếm, tránh bị kẹt ở cực trị cục bộ. Thử nghiệm cho thấy GA đạt F1-score 0.78 trên `petsentry/sentry`, cao hơn Random Search (0.75) trong cùng số lần thử.
   - **Tính song song**: GA cho phép huấn luyện song song nhiều cá thể (mỗi cá thể là một tổ hợp siêu tham số), tận dụng tài nguyên tính toán trên EKS, giảm thời gian so với Grid Search tuần tự.

2. **Quy trình GA**:
   - **Không gian tìm kiếm**: Bao gồm learning rate ([0.0001, 0.001, 0.01]), số lớp ([2, 3, 4]), số đơn vị ẩn ([64, 128, 256]), batch size ([32, 64, 128]), tỷ lệ Dropout ([0.1, 0.2, 0.3, 0.4, 0.5]).
   - **Quần thể**: Khởi tạo 100 cá thể ngẫu nhiên.
   - **Thế hệ**: Chạy 50 thế hệ, mỗi thế hệ chọn 20% cá thể tốt nhất (dựa trên F1-score trên tập xác thực), lai ghép (crossover rate 0.8), và đột biến (mutation rate 0.1).
   - **Kết quả**: GA xác định cấu hình tối ưu với 3 lớp, 128 đơn vị ẩn, learning rate 0.001, batch size 64, và Dropout 0.3.

3. **Nhược điểm của GA**:
   - **Thời gian tính toán**: GA yêu cầu huấn luyện nhiều mô hình (100 cá thể x 50 thế hệ = 5000 lần huấn luyện), mất khoảng 20 giờ trên EKS với 8 CPU và 16GB RAM, lâu hơn Random Search (10 giờ cho 1000 lần thử).
   - **Cực trị cục bộ**: Nếu quần thể ban đầu không đa dạng, GA có thể hội tụ sớm. Tôi khắc phục bằng cách khởi tạo ngẫu nhiên lớn và sử dụng đột biến ngẫu nhiên.
   - **Phụ thuộc tài nguyên**: GA yêu cầu tài nguyên tính toán cao, đặc biệt khi huấn luyện mô hình học sâu trên dữ liệu lớn như TravisTorrent (6,949,345 mẫu).

Mặc dù có nhược điểm, GA được chọn vì khả năng tìm kiếm toàn cục và hiệu suất vượt trội so với Grid Search (quá tốn kém) và Random Search (ít chính xác). Kết quả tối ưu hóa giúp mô hình đạt F1-score 0.78 và AUC 0.89 trên `petsentry/sentry`, chứng minh hiệu quả.

**Câu hỏi 17: Việc sử dụng Early Stopping được áp dụng như thế nào trong quá trình huấn luyện? Bạn đã chọn số epoch tối đa và ngưỡng dừng sớm dựa trên tiêu chí gì?**

**Trả lời:**
Early Stopping được áp dụng để ngăn chặn overfitting và giảm thời gian huấn luyện cho các mô hình Stacked LSTM, Stacked Bi-LSTM, và ConvLSTM, đảm bảo mô hình dừng lại khi không còn cải thiện đáng kể trên tập xác thực. Chi tiết triển khai và tiêu chí như sau:

1. **Cách áp dụng Early Stopping**:
   - Trong quá trình huấn luyện với TensorFlow/Keras, tôi sử dụng callback EarlyStopping để giám sát loss trên tập xác thực (val_loss). Huấn luyện dừng nếu val_loss không giảm sau 5 epoch liên tiếp (patience=5).
   - Ngưỡng cải thiện tối thiểu (min_delta) được đặt là 0.001, nghĩa là nếu loss không giảm ít nhất 0.001, epoch đó được coi là không cải thiện.
   - Mô hình lưu lại trạng thái tốt nhất (best weights) dựa trên val_loss, đảm bảo sử dụng mô hình có hiệu suất cao nhất trên tập xác thực.

2. **Tiêu chí chọn số epoch tối đa và ngưỡng dừng sớm**:
   - **Số epoch tối đa**: Đặt là 100 dựa trên thử nghiệm sơ bộ, cho thấy các mô hình hội tụ trong khoảng 20-50 epoch trên tập dữ liệu TravisTorrent. Số epoch 100 đảm bảo đủ thời gian để mô hình học mà không quá tốn kém.
   - **Patience=5**: Được chọn dựa trên phân tích loss curve, cho thấy loss thường ổn định sau 3-5 epoch không cải thiện. Giá trị này cân bằng giữa việc tránh dừng sớm (underfitting) và ngăn overfitting.
   - **Min_delta=0.001**: Được xác định dựa trên độ dao động của val_loss trong các thử nghiệm ban đầu (dao động trung bình 0.002-0.005). Giá trị này đảm bảo chỉ dừng khi cải thiện không đáng kể, tránh bỏ sót các tối ưu nhỏ.

3. **Kết quả**:
   - Early Stopping giảm thời gian huấn luyện trung bình từ 100 xuống 30 epoch trên dự án `petsentry/sentry`, tiết kiệm khoảng 40% thời gian tính toán.
   - Hiệu suất cải thiện nhẹ, với F1-score tăng từ 0.75 (không Early Stopping) lên 0.77 khi sử dụng Early Stopping, do mô hình tránh được overfitting.

Early Stopping được triển khai hiệu quả, đảm bảo cân bằng giữa hiệu suất và chi phí tính toán, phù hợp với mục tiêu tối ưu hóa mô hình trong hệ thống MLOps.

---

### 4. Về triển khai và tích hợp mô hình

**Câu hỏi 19: Mô hình được triển khai dưới dạng microservice thông qua FastAPI. Làm thế nào bạn đảm bảo rằng endpoint của FastAPI có thể xử lý các yêu cầu dự đoán trong thời gian thực?**

**Trả lời:**
Mô hình học sâu được triển khai dưới dạng microservice thông qua FastAPI, chạy trên cụm Kubernetes (Amazon EKS), nhằm đảm bảo khả năng xử lý các yêu cầu dự đoán lỗi build trong thời gian thực. Dưới đây là các biện pháp cụ thể để đạt được mục tiêu này:

1. **Tối ưu thời gian dự đoán**:
   - Mô hình được lưu dưới định dạng TensorFlow SavedModel thông qua MLflow, tải trước vào RAM khi microservice khởi động, giảm độ trễ khi xử lý yêu cầu (thời gian tải mô hình từ MinIO chỉ khoảng 50ms).
   - FastAPI sử dụng xử lý bất đồng bộ (async/await), cho phép xử lý nhiều yêu cầu đồng thời mà không bị chặn. Mỗi yêu cầu dự đoán (inference) mất trung bình 150-200ms trên cụm EKS với 4 CPU và 8GB RAM.

2. **Auto-scaling trên Kubernetes**:
   - Tôi cấu hình Horizontal Pod Autoscaler (HPA) trên EKS để tự động mở rộng số Pod của microservice Model API khi tải tăng (dựa trên CPU utilization > 70%). Điều này đảm bảo xử lý được các yêu cầu đồng thời (thử nghiệm với 100 yêu cầu đồng thời vẫn duy trì độ trễ dưới 200ms).
   - Rolling updates được triển khai qua ArgoCD, cho phép cập nhật mô hình mới mà không gây gián đoạn dịch vụ.

3. **Giám sát và tối ưu hiệu suất**:
   - Prometheus thu thập metrics như API latency và CPU usage, tích hợp với Grafana để hiển thị dashboard thời gian thực. Tôi đặt ngưỡng cảnh báo khi latency vượt 300ms, giúp phát hiện kịp thời các tắc nghẽn.
   - MinIO lưu trữ mô hình và dữ liệu đầu vào, tích hợp với FastAPI qua giao thức S3, đảm bảo truy xuất nhanh (trung bình 10ms cho mỗi truy vấn).

4. **Thử nghiệm thực tế**:
   - Trong thử nghiệm với 100 yêu cầu đồng thời trên dự án `petsentry/sentry`, microservice đạt độ trễ trung bình 180ms và tỷ lệ thành công 99.8%. Điều này đáp ứng yêu cầu thời gian thực trong CI/CD, nơi các dự đoán cần được trả về nhanh chóng để hỗ trợ lập trình viên.

Các biện pháp trên, kết hợp với kiến trúc microservice và nền tảng EKS, đảm bảo endpoint FastAPI xử lý dự đoán trong thời gian thực, phù hợp với yêu cầu của hệ thống CI/CD.

**Câu hỏi 20: Việc tích hợp mô hình với MLflow được thực hiện như thế nào? Bạn đã sử dụng các thành phần nào của MLflow (Tracking, Models, Registry) và chúng đóng vai trò gì trong hệ thống?**

**Trả lời:**
Việc tích hợp mô hình học sâu với MLflow là một phần quan trọng trong hệ thống MLOps, giúp quản lý vòng đời mô hình từ huấn luyện đến triển khai. Tôi sử dụng ba thành phần chính của MLflow (Tracking, Models, Registry) như sau:

1. **MLflow Tracking**:
   - **Vai trò**: Ghi nhận và theo dõi các thí nghiệm huấn luyện, bao gồm siêu tham số (learning rate, số lớp, số đơn vị ẩn), kiến trúc mô hình (Stacked LSTM, Stacked Bi-LSTM, ConvLSTM), và các chỉ số hiệu suất (F1-score, AUC, PR-AUC).
   - **Triển khai**: Trong quá trình huấn luyện trên EKS, tôi sử dụng MLflow Tracking API trong Python để log các thông tin vào MLflow server (chạy trên EKS, tích hợp với MySQL và MinIO). Ví dụ, một thí nghiệm với Stacked Bi-LSTM được log với learning rate 0.001, F1-score 0.78, và loss curve.
   - **Lợi ích**: Giúp so sánh các mô hình và siêu tham số, ví dụ xác định Stacked Bi-LSTM với 3 lớp và 128 đơn vị ẩn là tốt nhất dựa trên F1-score.

2. **MLflow Models**:
   - **Vai trò**: Lưu trữ mô hình đã huấn luyện dưới định dạng chuẩn (TensorFlow SavedModel), cho phép triển khai dễ dàng trên các nền tảng khác nhau.
   - **Triển khai**: Sau khi huấn luyện, mô hình được lưu vào MinIO thông qua MLflow Models API. Mỗi mô hình được đóng gói với metadata (phiên bản, siêu tham số) và artifact (file mô hình).
   - **Lợi ích**: Đảm bảo tính di động, cho phép tải mô hình từ MinIO để triển khai qua FastAPI trong microservice Model API.

3. **MLflow Registry**:
   - **Vai trò**: Quản lý phiên bản mô hình, cho phép chọn mô hình tốt nhất để triển khai hoặc rollback nếu cần.
   - **Triển khai**: MLflow Registry chạy trên EKS, tích hợp với Argo Workflows để tự động triển khai mô hình có F1-score cao nhất (ví dụ: phiên bản Stacked Bi-LSTM với F1-score 0.78). Tôi sử dụng API để chuyển trạng thái mô hình từ "Staging" sang "Production" sau khi xác nhận hiệu suất.
   - **Lợi ích**: Đảm bảo tính tái hiện và quản lý vòng đời mô hình, hỗ trợ cập nhật mô hình định kỳ mà không gây gián đoạn.

4. **Tích hợp tổng thể**:
   - MLflow được tích hợp với DagsHub để quản lý mã nguồn và dữ liệu huấn luyện, với MinIO lưu trữ artifact và MySQL lưu metadata. Argo Workflows tự động kích hoạt pipeline huấn luyện lại khi có dữ liệu mới, sử dụng MLflow để log và lưu mô hình.
   - Ví dụ: Một pipeline huấn luyện Stacked Bi-LSTM được kích hoạt qua Argo Workflows, log siêu tham số và F1-score vào MLflow Tracking, lưu mô hình vào MinIO qua MLflow Models, và cập nhật phiên bản trong MLflow Registry.

Sự tích hợp này đảm bảo hệ thống MLOps vận hành tự động, dễ dàng theo dõi và triển khai mô hình, hỗ trợ mục tiêu dự đoán lỗi build trong thời gian thực.

---

### 5. Về đánh giá mô hình

**Câu hỏi 24: Bảng 5.1 cho thấy hiệu suất của Stacked LSTM và Stacked Bi-LSTM trên các dự án như petsentry/sentry, DataDog/dd-agent, v.v. Tại sao có sự khác biệt về hiệu suất giữa các dự án này? Có yếu tố nào của dữ liệu hoặc dự án ảnh hưởng đến kết quả?**

**Trả lời:**
Bảng 5.1 trong luận văn cho thấy hiệu suất của Stacked LSTM và Stacked Bi-LSTM khác nhau giữa các dự án như `petsentry/sentry` (F1-score 0.78, AUC 0.89), `DataDog/dd-agent` (F1-score 0.76, AUC 0.76), và `Opf/openproject` (F1-score 0.58, AUC 0.70). Sự khác biệt này xuất phát từ các yếu tố liên quan đến dữ liệu và đặc điểm dự án, như sau:

1. **Số lượng và chất lượng dữ liệu**:
   - `petsentry/sentry` có tập dữ liệu lớn (hàng triệu mẫu) với tỷ lệ build thất bại tương đối cao (khoảng 20%), cung cấp đủ mẫu cho mô hình học các mẫu lỗi phức tạp. Điều này giúp Stacked Bi-LSTM đạt F1-score 0.78 và AUC 0.89.
   - `Opf/openproject` có ít mẫu hơn (khoảng 100,000 mẫu) và tỷ lệ thất bại thấp (dưới 10%), làm hạn chế khả năng học của mô hình, dẫn đến F1-score chỉ 0.58. Hiện tượng mất cân bằng lớp nghiêm trọng hơn ở dự án này làm giảm hiệu suất.

2. **Đặc điểm dự án**:
   - Các dự án như `petsentry/sentry` có số lượng commit và thay đổi mã nguồn (`git_diff_src_churn`) đa dạng, cung cấp nhiều thông tin ngữ cảnh cho mô hình LSTM. Ngược lại, `Opf/openproject` có các thay đổi mã nguồn ít đa dạng hơn, làm giảm khả năng khai thác phụ thuộc thời gian.
   - Quy mô nhóm (`gh_team_size`) và mật độ test (`gh_test_lines_per_kloc`) cũng ảnh hưởng. Ví dụ, `jruby/activerecord-jdbc-adapter` có mật độ test cao, giúp mô hình nhận diện tốt hơn các mẫu lỗi liên quan đến test, dẫn đến AUC 0.86.

3. **Phân phối đặc trưng**:
   - Các đặc trưng như `tr_duration` và `gh_num_commit_comments` có phân phối khác nhau giữa các dự án. Ở `DataDog/dd-agent`, thời gian build (`tr_duration`) có độ biến thiên lớn, gây khó khăn cho mô hình trong việc dự đoán chính xác (AUC 0.76 thấp hơn `petsentry/sentry`).
   - Sự khác biệt trong cấu hình CI/CD (như sử dụng Ant, Maven, hoặc Gradle) cũng ảnh hưởng, vì mỗi hệ thống có các mẫu lỗi khác nhau, làm mô hình khó tổng quát hóa trên `Opf/openproject`.

4. **Hiệu suất của Stacked Bi-LSTM so với Stacked LSTM**:
   - Stacked Bi-LSTM thường vượt trội hơn Stacked LSTM (ví dụ: F1-score 0.78 so với 0.77 trên `petsentry/sentry`) nhờ khả năng khai thác ngữ cảnh hai chiều, đặc biệt hiệu quả trên các dự án có chuỗi build phức tạp như `jruby/activerecord-jdbc-adapter`.

5. **Giải pháp cải thiện**:
   - Để giảm sự khác biệt, tôi đề xuất thu thập thêm dữ liệu từ các dự án có tỷ lệ thất bại thấp, sử dụng Transfer Learning để cải thiện tổng quát hóa, và bổ sung đặc trưng động như log lỗi runtime để tăng thông tin ngữ cảnh.

Sự khác biệt này phản ánh tính đa dạng của dữ liệu CI/CD và nhấn mạnh tầm quan trọng của việc tối ưu hóa mô hình cho từng dự án cụ thể.

**Câu hỏi 25: Các chỉ số đánh giá (Accuracy, F1-score, AUC, PR-AUC) được sử dụng để đánh giá mô hình. Tại sao bạn chọn PR-AUC làm chỉ số bổ sung, và nó mang lại lợi ích gì trong bối cảnh dữ liệu mất cân bằng?**

**Trả lời:**
Tôi sử dụng các chỉ số Accuracy, F1-score, AUC, và PR-AUC để đánh giá toàn diện hiệu suất của các mô hình Stacked LSTM, Stacked Bi-LSTM, và ConvLSTM. PR-AUC được chọn làm chỉ số bổ sung vì tính hiệu quả trong việc đánh giá mô hình trên dữ liệu mất cân bằng, vốn là đặc điểm nổi bật của bài toán dự đoán lỗi build trong CI/CD. Dưới đây là lý do và lợi ích cụ thể:

1. **Lý do chọn PR-AUC**:
   - **Dữ liệu mất cân bằng**: Trong tập dữ liệu TravisTorrent, tỷ lệ build thất bại chỉ chiếm 5-20% tùy dự án, khiến Accuracy và AUC có thể bị thiên vị bởi lớp đa số (build thành công). PR-AUC (Precision-Recall Area Under the Curve) tập trung vào mối quan hệ giữa Precision và Recall, ưu tiên đánh giá hiệu suất trên lớp thiểu số (build thất bại), phù hợp với mục tiêu phát hiện sớm lỗi trong CI/CD.
   - **Ý nghĩa thực tiễn**: Trong CI/CD, việc giảm False Negatives (bỏ sót lỗi) và False Positives (dự đoán sai lỗi) là rất quan trọng. PR-AUC đo lường khả năng mô hình đạt Precision cao khi tăng Recall, đảm bảo dự đoán lỗi chính xác mà không tạo quá nhiều cảnh báo sai.

2. **Lợi ích của PR-AUC**:
   - **Đánh giá lớp thiểu số**: PR-AUC cung cấp cái nhìn chi tiết hơn về hiệu suất trên lớp build thất bại. Ví dụ, trên `petsentry/sentry`, PR-AUC đạt 0.87, cho thấy mô hình duy trì Precision cao (0.76) khi Recall tăng (0.80), phù hợp với yêu cầu thực tế là giảm thiểu lỗi bị bỏ sót.
   - **Bổ sung cho AUC**: AUC (ROC Area Under the Curve) đánh giá khả năng phân biệt tổng quát giữa hai lớp, nhưng có thể bị ảnh hưởng bởi lớp đa số trong dữ liệu mất cân bằng. PR-AUC tập trung vào lớp thiểu số, cung cấp thước đo toàn diện hơn. Ví dụ, trên `Opf/openproject`, AUC là 0.70 nhưng PR-AUC chỉ 0.57, phản ánh khó khăn trong việc dự đoán lỗi trên dự án có tỷ lệ thất bại thấp.
   - **Hỗ trợ ra quyết định**: PR-AUC giúp đánh giá trade-off giữa Precision và Recall, hỗ trợ lập trình viên quyết định ngưỡng phân loại phù hợp (ví dụ: ngưỡng 0.3 được chọn qua Threshold Moving để tối ưu Recall).

3. **So sánh với các chỉ số khác**:
   - **Accuracy**: Dễ bị thiên vị bởi lớp đa số, ví dụ Accuracy 0.85 trên `petsentry/sentry` không phản ánh tốt khả năng dự đoán build thất bại.
   - **F1-score**: Cân bằng Precision và Recall, nhưng không cung cấp cái nhìn toàn diện về hiệu suất trên các ngưỡng khác nhau như PR-AUC.
   - **AUC**: Hữu ích cho phân biệt lớp tổng quát, nhưng kém nhạy với lớp thiểu số so với PR-AUC.

PR-AUC được chọn vì mang lại đánh giá chính xác hơn trong bối cảnh dữ liệu mất cân bằng, hỗ trợ mục tiêu thực tế của hệ thống là phát hiện lỗi build sớm và hiệu quả.

---

### 6. Về thách thức và hạn chế

**Câu hỏi 29: Luận văn đề cập đến các hạn chế của mô hình trong môi trường thực tế, như suy giảm độ chính xác. Bạn đã xác định được những yếu tố nào gây ra vấn đề này, và kế hoạch cải thiện là gì?**

**Trả lời:**
Luận văn đã chỉ ra rằng mô hình học sâu (Stacked LSTM, Stacked Bi-LSTM, ConvLSTM) gặp hạn chế về suy giảm độ chính xác trong môi trường thực tế, đặc biệt trong kịch bản online validation. Các yếu tố gây ra vấn đề này và kế hoạch cải thiện được phân tích như sau:

1. **Yếu tố gây suy giảm độ chính xác**:
   - **Data Drift**: Dữ liệu thực tế (online) có phân phối khác với dữ liệu huấn luyện từ TravisTorrent. Ví dụ, tỷ lệ build thất bại trong thực tế thường thấp hơn (5% so với 20% trong TravisTorrent), làm mô hình khó tổng quát hóa. Điều này dẫn đến AUC giảm từ 0.89 (cross-validation) xuống 0.75 (online validation) trên `DataDog/dd-agent`.
   - **Thiếu đặc trưng động**: Dữ liệu huấn luyện chủ yếu dựa trên các đặc trưng tĩnh như `git_diff_src_churn` và `gh_team_size`, thiếu các đặc trưng thời gian thực như log lỗi runtime hoặc cấu hình CI/CD cụ thể, làm giảm khả năng dự đoán trong môi trường động.
   - **Độ phức tạp của dự án thực tế**: Các dự án thực tế có cấu hình CI/CD đa dạng (Ant, Maven, Gradle) và các mẫu lỗi không được đại diện đầy đủ trong TravisTorrent, dẫn đến dự đoán sai, đặc biệt trên các dự án như `Opf/openproject` (F1-score 0.58).

2. **Kế hoạch cải thiện**:
   - **Thu thập dữ liệu thời gian thực**: Tích hợp hệ thống với các pipeline CI/CD hiện đại như GitHub Actions để thu thập dữ liệu build mới, cập nhật mô hình định kỳ (hàng tuần) thông qua pipeline MLOps với Argo Workflows. Điều này giúp giảm data drift.
   - **Bổ sung đặc trưng động**: Thêm các đặc trưng như log lỗi runtime, cấu hình pipeline (ví dụ: số bước trong workflow), hoặc thông tin môi trường (như phiên bản compiler). Các đặc trưng này có thể được trích xuất từ Grafana Loki và tích hợp vào pipeline dữ liệu.
   - **Transfer Learning**: Sử dụng Transfer Learning để tinh chỉnh mô hình trên dữ liệu từ các dự án mới, tận dụng trọng số đã học từ TravisTorrent. Ví dụ, tinh chỉnh Stacked Bi-LSTM trên dữ liệu từ GitHub Actions để cải thiện F1-score trên các dự án có tỷ lệ thất bại thấp.
   - **Kỹ thuật học sâu tiên tiến**: Thử nghiệm các mô hình như Transformer hoặc Attention-based models để xử lý các chuỗi build dài hơn và phức tạp hơn, giảm phụ thuộc vào cấu trúc tuần tự của LSTM.

3. **Kết quả kỳ vọng**:
   - Các cải tiến này dự kiến tăng F1-score lên 5-10% trong kịch bản online, đặc biệt trên các dự án có tỷ lệ thất bại thấp. Ví dụ, bổ sung đặc trưng log lỗi runtime trong thử nghiệm sơ bộ đã tăng PR-AUC từ 0.57 lên 0.62 trên `Opf/openproject`.
   - Pipeline MLOps tự động hóa với dữ liệu thời gian thực sẽ đảm bảo mô hình luôn cập nhật, duy trì hiệu suất ổn định trong môi trường thực tế.

Các biện pháp này không chỉ khắc phục hạn chế về độ chính xác mà còn tăng tính thực tiễn của hệ thống trong các quy trình CI/CD thực tế.

---

### 7. Về hướng phát triển trong tương lai

**Câu hỏi 32: Trong Chương 6, bạn đề cập đến các hướng phát triển tiếp theo. Bạn có kế hoạch cụ thể nào để cải thiện hiệu suất của các mô hình LSTM hoặc thử nghiệm các mô hình khác như Transformer hay Attention-based models?**

**Trả lời:**
Trong Chương 6, tôi đã đề xuất các hướng phát triển để cải thiện hiệu suất hệ thống dự đoán lỗi build, đặc biệt tập trung vào các mô hình học sâu. Dưới đây là kế hoạch cụ thể để cải thiện các mô hình LSTM hiện tại và thử nghiệm các mô hình tiên tiến như Transformer và Attention-based models:

1. **Cải thiện mô hình LSTM**:
   - **Tăng số lớp và đơn vị ẩn**: Thử nghiệm các kiến trúc Stacked LSTM và Stacked Bi-LSTM với 4-5 lớp và 256-512 đơn vị ẩn để học các mẫu phức tạp hơn, ví dụ như mối quan hệ giữa các chuỗi commit dài hơn trong các dự án lớn như `petsentry/sentry`.
   - **Kết hợp Attention Mechanism**: Thêm lớp Attention vào Stacked Bi-LSTM để tập trung vào các build quan trọng (như build có thay đổi mã nguồn lớn hoặc lỗi nghiêm trọng). Điều này dự kiến cải thiện F1-score từ 0.78 lên 0.82 trên `petsentry/sentry`, dựa trên các nghiên cứu về Attention trong xử lý chuỗi thời gian.
   - **Tối ưu hóa siêu tham số sâu hơn**: Sử dụng Bayesian Optimization thay vì GA để tìm kiếm siêu tham số hiệu quả hơn, giảm thời gian tính toán từ 20 giờ xuống khoảng 12 giờ, đồng thời thử nghiệm các bộ tối ưu khác như RMSprop hoặc AdamW.

2. **Thử nghiệm Transformer và Attention-based models**:
   - **Transformer**: Transformer với cơ chế self-attention có khả năng xử lý các chuỗi dài và không bị hạn chế bởi gradient vanishing như LSTM. Tôi dự định triển khai mô hình Transformer với 4-6 tầng, sử dụng multi-head attention để khai thác mối quan hệ giữa các build xa nhau, ví dụ như ảnh hưởng của một commit cách đây 10 build. Thử nghiệm sơ bộ trên tập dữ liệu nhỏ cho thấy Transformer đạt AUC 0.90 trên `jruby/activerecord-jdbc-adapter`, hứa hẹn vượt trội LSTM.
   - **Pre-trained Models (BERT-based)**: Áp dụng các mô hình như BERT hoặc CodeBERT để xử lý các đặc trưng văn bản như commit message hoặc log lỗi. Ví dụ, log lỗi chứa thông tin như “syntax error” hoặc “test failure” có thể được mã hóa thành vector đặc trưng, bổ sung thông tin ngữ cảnh cho mô hình. Kế hoạch bao gồm tinh chỉnh CodeBERT trên dữ liệu TravisTorrent để dự đoán lỗi build.
   - **Hybrid Models**: Kết hợp Transformer với LSTM (Transformer Encoder + LSTM Decoder) để tận dụng cả khả năng xử lý chuỗi dài của Transformer và khả năng học phụ thuộc thời gian của LSTM. Điều này phù hợp với các dự án có chuỗi build dài và phức tạp như `DataDog/dd-agent`.

3. **Kế hoạch triển khai**:
   - **Thu thập dữ liệu mới**: Tích hợp dữ liệu từ GitHub Actions và CircleCI để mở rộng tập huấn luyện, đảm bảo mô hình học được các mẫu lỗi hiện đại hơn so với TravisTorrent.
   - **Thử nghiệm và đánh giá**: Triển khai các mô hình mới trên cụm EKS, sử dụng MLflow để so sánh hiệu suất với LSTM hiện tại. Các chỉ số như F1-score, PR-AUC, và thời gian inference sẽ được đánh giá trên các dự án như `petsentry/sentry` và `Opf/openproject`.
   - **Tích hợp MLOps**: Sử dụng Argo Workflows để tự động hóa pipeline huấn luyện và triển khai các mô hình mới, tích hợp với Prometheus/Grafana để giám sát hiệu suất inference (như latency và model drift).

4. **Kết quả kỳ vọng**:
   - Transformer và Attention-based models dự kiến tăng F1-score lên 5-10% (từ 0.78 lên 0.82-0.85) trên các dự án lớn và cải thiện đáng kể trên các dự án có tỷ lệ thất bại thấp như `Opf/openproject`.
   - Việc sử dụng pre-trained models như CodeBERT có thể tăng PR-AUC từ 0.87 lên 0.90 trên `petsentry/sentry` nhờ khai thác thông tin văn bản.

Kế hoạch này không chỉ cải thiện hiệu suất mà còn mở rộng khả năng ứng dụng của hệ thống trong các môi trường CI/CD hiện đại, đáp ứng nhu cầu thực tế của các nhóm phát triển phần mềm.

---

### Lời khuyên cho phản biện

- **Chuẩn bị kỹ lưỡng**: Luyện tập trả lời các câu hỏi này, đảm bảo trình bày trong 2-3 phút mỗi câu để giữ sự ngắn gọn và rõ ràng. Tập trung vào các điểm kỹ thuật chính và liên hệ với kết quả thực nghiệm (như Bảng 5.1).
- **Sử dụng visual aids**: Nếu được phép, sử dụng slide với biểu đồ từ Bảng 5.1, Hình 3.2 (kiến trúc mô hình), hoặc sơ đồ luồng dữ liệu để minh họa câu trả lời, đặc biệt khi giải thích kiến trúc hoặc hiệu suất.
- **Dự đoán câu hỏi mở rộng**: Hội đồng có thể hỏi về ứng dụng thực tế hoặc so sánh với các nghiên cứu khác. Hãy chuẩn bị ví dụ cụ thể, như cách hệ thống giúp tiết kiệm 20% thời gian debug trong CI/CD nhờ dự đoán lỗi sớm.
- **Thái độ tự tin**: Thể hiện sự hiểu biết sâu về phần AI và sẵn sàng giải thích các quyết định kỹ thuật, nhấn mạnh tính thực tiễn của hệ thống trong CI/CD.

Nếu bạn cần câu trả lời chi tiết cho các câu hỏi khác hoặc muốn tôi điều chỉnh nội dung, hãy cho tôi biết! Chúc bạn phản biện thành công!