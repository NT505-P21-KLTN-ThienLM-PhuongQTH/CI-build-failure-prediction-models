import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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