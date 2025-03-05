project_name/
├── data/                  # Thư mục chứa dữ liệu
│   ├── raw/              # Dữ liệu thô ban đầu
│   ├── processed/        # Dữ liệu đã được tiền xử lý
│   └── external/         # Dữ liệu từ nguồn bên ngoài (nếu có)
├── models/               # Thư mục lưu mô hình đã huấn luyện
│   ├── single_lstm/      # Mô hình Single LSTM
│   ├── bilstm/           # Mô hình Bi-LSTM
│   └── convlstm/         # Mô hình ConvLSTM
├── src/                  # Thư mục chứa source code
│   ├── preprocess.py     # Tiền xử lý dữ liệu
│   ├── single_lstm.py    # Code cho Single LSTM
│   ├── bilstm.py         # Code cho Bi-LSTM
│   ├── convlstm.py       # Code cho ConvLSTM
│   ├── train.py          # Script huấn luyện chung
│   ├── evaluate.py       # Script đánh giá mô hình
│   └── utils.py          # Các hàm tiện ích (helper functions)
├── notebooks/            # Thư mục chứa Jupyter Notebooks (nếu dùng)
│   └── exploration.ipynb # Phân tích dữ liệu ban đầu
├── results/              # Thư mục lưu kết quả
│   ├── logs/             # Log huấn luyện (loss, accuracy,...)
│   └── visualizations/   # Biểu đồ, hình ảnh trực quan
├── requirements.txt      # Danh sách thư viện cần thiết
├── README.md             # Tài liệu mô tả dự án
└── .gitignore            # File để bỏ qua các file không cần commit (nếu dùng Git)