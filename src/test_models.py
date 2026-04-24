import os
import sys

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

import numpy as np
from src.model_engine import ADFNetModelEngine, ModelEngine
from training.mix_ import load_dataset, Config
from src.evaluation import print_confusion_matrix, save_evaluation_results


def evaluate_adfnet(model_path: str = "weights/ADFNet_best.pth"):
    """评估 ADF-Net 模型在测试集上的表现"""

    print("=" * 60)
    print("评估 ADF-Net 模型（测试集）")
    print("=" * 60)

    # 1. 加载测试集
    print("\n[1/4] 加载测试集...")
    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}, 类别数: {len(label_encoder.classes_)}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    engine = ADFNetModelEngine(model_path=model_path)

    # 3. 预测
    print("\n[3/4] 进行预测...")
    all_preds = []
    for i in range(len(X_test)):
        result = engine.predict(X_test[i])
        all_preds.append(result['label_id'])

    # 4. 评估
    print("\n[4/4] 评估结果...")
    print_confusion_matrix(np.array(all_preds), y_test, label_encoder)
    save_evaluation_results(np.array(all_preds), y_test, label_encoder, model_name="ADFNet_test")

    accuracy = np.sum(np.array(all_preds) == y_test) / len(y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")


def evaluate_standard_model():
    """评估标准双流CNN模型在测试集上的表现"""

    print("=" * 60)
    print("评估标准双流CNN模型（测试集）")
    print("=" * 60)

    # 1. 加载测试集
    print("\n[1/4] 加载测试集...")
    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}, 类别数: {len(label_encoder.classes_)}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    engine = ModelEngine(
        time_model_path="weights/TemporalCNN_best.pth",
        byte_model_path="weights/PayloadCNN_best.pth",
        nb_clf_path="weights/nb_classifier.pkl"
    )

    # 3. 预测
    print("\n[3/4] 进行预测...")
    all_preds = []
    for i in range(len(X_test)):
        result = engine.predict(X_test[i])
        all_preds.append(result['label_id'])

    # 4. 评估
    print("\n[4/4] 评估结果...")
    print_confusion_matrix(np.array(all_preds), y_test, label_encoder)
    save_evaluation_results(np.array(all_preds), y_test, label_encoder, model_name="StandardModel_test")

    accuracy = np.sum(np.array(all_preds) == y_test) / len(y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")


def compare_models():
    """对比两个模型在测试集上的表现"""

    print("=" * 60)
    print("模型对比（测试集）")
    print("=" * 60)

    # 加载测试集
    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}, 类别数: {len(label_encoder.classes_)}")

    # 标准模型预测
    print("\n标准模型预测中...")
    standard_engine = ModelEngine(
        time_model_path="weights/TemporalCNN_best.pth",
        byte_model_path="weights/PayloadCNN_best.pth",
        nb_clf_path="weights/nb_classifier.pkl"
    )
    standard_preds = []
    for i in range(len(X_test)):
        result = standard_engine.predict(X_test[i])
        standard_preds.append(result['label_id'])

    # ADF-Net 预测
    print("ADF-Net 预测中...")
    adfnet_engine = ADFNetModelEngine(model_path="weights/ADFNet_best.pth")
    adfnet_preds = []
    for i in range(len(X_test)):
        result = adfnet_engine.predict(X_test[i])
        adfnet_preds.append(result['label_id'])

    # 打印标准模型的详细报告
    print("\n" + "=" * 60)
    print("标准双流CNN 详细评估结果")
    print("=" * 60)
    print_confusion_matrix(np.array(standard_preds), y_test, label_encoder)
    save_evaluation_results(np.array(standard_preds), y_test, label_encoder, model_name="StandardModel_test")

    # 打印 ADF-Net 的详细报告
    print("\n" + "=" * 60)
    print("ADF-Net 详细评估结果")
    print("=" * 60)
    print_confusion_matrix(np.array(adfnet_preds), y_test, label_encoder)
    save_evaluation_results(np.array(adfnet_preds), y_test, label_encoder, model_name="ADFNet_test")

    # 计算准确率
    standard_acc = np.sum(np.array(standard_preds) == y_test) / len(y_test)
    adfnet_acc = np.sum(np.array(adfnet_preds) == y_test) / len(y_test)

    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"标准双流CNN 准确率: {standard_acc:.4f}")
    print(f"ADF-Net 准确率:     {adfnet_acc:.4f}")
    print(f"提升:                {(adfnet_acc - standard_acc) * 100:+.2f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "adfnet":
            evaluate_adfnet()
        elif sys.argv[1] == "standard":
            evaluate_standard_model()
        elif sys.argv[1] == "compare":
            compare_models()
        else:
            print("用法: python test_models.py [adfnet|standard|compare]")
    else:
        compare_models()
