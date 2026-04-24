import os
import sys

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

import numpy as np
from src.model_engine import ADFNetModelEngine, ModelEngine  # 修改导入
from training.mix_ import load_dataset, Config
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime



def print_confusion_matrix(y_true, y_pred, label_encoder: LabelEncoder):
    """
    打印分类评估报告和混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_encoder: 标签编码器（包含类别名称）
    """
    classes = label_encoder.classes_
    labels = label_encoder.transform(classes)

    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0
    ))

    print("\n" + "=" * 60)
    print("Confusion Matrix:")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)
    print("=" * 60 + "\n")


def save_evaluation_results(y_true, y_pred, label_encoder: LabelEncoder,
                            model_name: str = "model", save_dir: str = "./results"):
    """
    保存评估结果到 CSV 文件

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_encoder: 标签编码器
        model_name: 模型名称（用于文件名）
        save_dir: 保存目录
    """
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)

    classes = label_encoder.classes_
    labels = label_encoder.transform(classes)

    # 1. 保存分类报告
    report_dict = classification_report(
        y_true, y_pred, labels=labels,
        target_names=classes, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"{model_name}_classification_report_{timestamp}.csv")
    report_df.to_csv(report_path)
    print(f"分类报告已保存: {report_path}")

    # 2. 保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix_{timestamp}.csv")
    cm_df.to_csv(cm_path)
    print(f"混淆矩阵已保存: {cm_path}")

    # 3. 保存准确率
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    with open(os.path.join(save_dir, f"{model_name}_accuracy_{timestamp}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {len(y_true)}\n")
    print(f"准确率: {accuracy:.4f}")
