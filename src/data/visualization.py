import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_line(df, column, project, figsize=(10, 4)):
    """Vẽ biểu đồ đường cho một cột."""
    plt.figure(figsize=figsize)
    plt.plot(df[column].reset_index(drop=True), label=column, linewidth=1.2)
    plt.title(f"Line Plot of {column} for {project}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pie(df, column_counts, title, figsize=(3, 3)):
    """Vẽ biểu đồ tròn."""
    plt.figure(figsize=figsize)
    plt.pie(df['Ratio (%)'], labels=df[column_counts], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_multi_project(df, column, project_column='gh_project_name', figsize=(12, 6), dropna=True):
    """Vẽ biểu đồ cho nhiều dự án, tùy chọn xử lý NaN."""
    plt.figure(figsize=figsize)
    for project_name, group_df in df.groupby(project_column):
        # Chuyển đổi cột thành kiểu số, bỏ qua lỗi
        y_data = pd.to_numeric(group_df[column], errors='coerce')
        if dropna:
            # Chỉ vẽ các giá trị không phải NaN
            mask = ~y_data.isna()
            plt.plot(group_df.index[mask], y_data[mask], label=project_name, alpha=0.6)
        else:
            # Điền NaN bằng 0
            y_data = y_data.fillna(0)
            plt.plot(group_df.index, y_data, label=project_name, alpha=0.6)
    plt.title(f"Biểu đồ giá trị '{column}' theo project")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df, title="Feature Importance (Average Across Models)", top_n=None):
    """
    Vẽ biểu đồ feature importance chỉ dựa trên giá trị trung bình (Average_Importance).

    Parameters:
    - importance_df (pd.DataFrame): DataFrame chứa cột 'Feature' và 'Average_Importance'.
    - title (str): Tiêu đề của biểu đồ.
    - top_n (int, optional): Số lượng feature hàng đầu để hiển thị. Nếu None, hiển thị tất cả.
    """
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    n_features = len(importance_df)
    y_pos = np.arange(n_features)

    # Tạo figure và axes
    plt.figure(figsize=(10, max(6, n_features * 0.4)))
    bar_width = 0.5  # Độ rộng của thanh

    # Vẽ thanh ngang chỉ với Average_Importance
    bars = plt.barh(y_pos, importance_df['Average_Importance'], bar_width, color='#1f77b4', label='Average Importance')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(x=width + 0.02, y=bar.get_y() + bar.get_height() / 2, s=f'{width:.2f}',
                 ha='left', va='center', color='black')

    # Cài đặt trục và nhãn
    plt.yticks(y_pos, importance_df['Feature'])
    plt.xlabel('Feature Importance (Average)')
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()


def plot_class_distribution(train_sets, test_sets, proj_name):
    """Vẽ biểu đồ tỉ lệ 0 và 1 trong cột build_failed cho từng fold."""
    plt.figure(figsize=(12, 6))
    folds = range(len(train_sets))

    # Tính tỉ lệ cho train và test
    train_ratios = [train_set['build_failed'].value_counts(normalize=True) for train_set in train_sets]
    test_ratios = [test_set['build_failed'].value_counts(normalize=True) for test_set in test_sets]

    # Chuẩn bị dữ liệu để vẽ
    train_0 = [r.get(0, 0) for r in train_ratios]
    train_1 = [r.get(1, 0) for r in train_ratios]
    test_0 = [r.get(0, 0) for r in test_ratios]
    test_1 = [r.get(1, 0) for r in test_ratios]

    # Thiết lập vị trí cột
    bar_width = 0.35
    x = np.arange(len(folds))

    # Vẽ biểu đồ
    train_bars_0 = plt.bar(x - bar_width / 2, train_0, bar_width, label='Train 0', color='skyblue')
    train_bars_1 = plt.bar(x - bar_width / 2, train_1, bar_width, bottom=train_0, label='Train 1', color='lightcoral')
    test_bars_0 = plt.bar(x + bar_width / 2, test_0, bar_width, label='Test 0', color='lightgreen')
    test_bars_1 = plt.bar(x + bar_width / 2, test_1, bar_width, bottom=test_0, label='Test 1', color='salmon')

    # Thêm giá trị tỉ lệ trên cột
    for i in range(len(folds)):
        # Train 0
        plt.text(x[i] - bar_width / 2, train_0[i] / 2, f'{train_0[i]:.2f}', ha='center', va='center', color='black')
        # Train 1
        plt.text(x[i] - bar_width / 2, train_0[i] + train_1[i] / 2, f'{train_1[i]:.2f}', ha='center', va='center', color='black')
        # Test 0
        plt.text(x[i] + bar_width / 2, test_0[i] / 2, f'{test_0[i]:.2f}', ha='center', va='center', color='black')
        # Test 1
        plt.text(x[i] + bar_width / 2, test_0[i] + test_1[i] / 2, f'{test_1[i]:.2f}', ha='center', va='center', color='black')

    plt.xlabel('Fold')
    plt.ylabel('Ratio')
    plt.title(f'Class Distribution (0 and 1) in build_failed - {proj_name}')
    plt.xticks(x, [f'Fold {i + 1}' for i in folds])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_probs, proj_name, fold_idx, save_path=None):
    """
    Vẽ đường cong ROC và tính AUC.

    Parameters:
    - y_true: Nhãn thực tế (ground truth).
    - y_pred_probs: Xác suất dự đoán cho lớp positive (class 1).
    - proj_name: Tên dự án (dùng để đặt tiêu đề).
    - fold_idx: Chỉ số fold (dùng để đặt tiêu đề).
    - save_path: Đường dẫn để lưu biểu đồ (nếu None thì không lưu).
    """
    # Tính FPR, TPR và ngưỡng
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Vẽ đường cong ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {proj_name} (Fold {fold_idx + 1})')
    plt.legend(loc="lower right")

    # Lưu biểu đồ nếu có đường dẫn
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved at: {save_path}")

    plt.show()
    plt.close()

def plot_training_history(history, proj_name, fold_idx, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_title(f'Loss - {proj_name} (Fold {fold_idx + 1})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax2.set_title(f'Accuracy - {proj_name} (Fold {fold_idx + 1})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved at: {save_path}")

    plt.show()
    plt.close()