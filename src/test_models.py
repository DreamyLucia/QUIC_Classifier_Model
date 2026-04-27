import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

import numpy as np
import torch
from src.model_engine import (
    ModelEngine,  # 基线双流CNN
    ADFNet,  # 完整ADF-Net
    ADFNet_AttentionOnly,  # 仅注意力融合
    ADFNet_InteractionOnly  # 仅跨粒度交互
)
from training.mix_ import load_dataset, Config
from src.evaluation import print_confusion_matrix, save_evaluation_results


class AblationModelEngine:
    """消融模型引擎（用于测试集评估）"""

    def __init__(self, model_path: str, model_type: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_classes = 8
        self.class_names = ['Nkiri', 'bilibili', 'edge', 'kwai',
                            'tencentnews', 'tencentvideo', 'tiktok', 'xiaohongshu']

        self.time_features_dim = 120
        self.byte_features_dim = 900
        self.stat_features_dim = 4

        # 根据模型类型选择对应的类
        if model_type == 'attention_only':
            self.model = ADFNet_AttentionOnly(num_classes=self.num_classes).to(self.device)
        elif model_type == 'interaction_only':
            self.model = ADFNet_InteractionOnly(num_classes=self.num_classes).to(self.device)
        elif model_type == 'full':
            self.model = ADFNet(num_classes=self.num_classes).to(self.device)
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ 模型加载成功: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")

    def predict(self, features: np.ndarray) -> dict:
        """预测单个样本"""
        time_feat = features[:self.time_features_dim].reshape(1, -1)
        byte_feat = features[self.time_features_dim:self.time_features_dim + self.byte_features_dim].reshape(1, -1)
        stats_feat = features[-self.stat_features_dim:].reshape(1, -1)

        time_tensor = torch.FloatTensor(time_feat).to(self.device)
        byte_tensor = torch.FloatTensor(byte_feat).to(self.device)
        stats_tensor = torch.FloatTensor(stats_feat).to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(time_tensor, byte_tensor, stats_tensor)
            probs = torch.softmax(outputs, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()

        return {
            'label': self.class_names[pred_id],
            'label_id': pred_id,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy().tolist()
        }


def evaluate_standard_model():
    """评估基线双流CNN模型"""
    print("=" * 60)
    print("评估基线双流CNN模型（测试集）")
    print("=" * 60)

    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}")

    engine = ModelEngine(
        time_model_path="weights/TemporalCNN_best.pth",
        byte_model_path="weights/PayloadCNN_best.pth",
        nb_clf_path="weights/nb_classifier.pkl"
    )

    print("预测中...")
    all_preds = []
    for i in range(len(X_test)):
        result = engine.predict(X_test[i])
        all_preds.append(result['label_id'])

    # ✅ 修改顺序：真实标签在前，预测标签在后
    print_confusion_matrix(y_test, np.array(all_preds), label_encoder)
    save_evaluation_results(y_test, np.array(all_preds), label_encoder, model_name="StandardModel_test")

    accuracy = np.sum(np.array(all_preds) == y_test) / len(y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")


def evaluate_ablation_model(model_path: str, model_type: str, model_name: str):
    """评估消融模型"""
    print("=" * 60)
    print(f"评估 {model_name}（测试集）")
    print("=" * 60)

    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}")

    engine = AblationModelEngine(model_path=model_path, model_type=model_type)

    print("预测中...")
    all_preds = []
    for i in range(len(X_test)):
        result = engine.predict(X_test[i])
        all_preds.append(result['label_id'])

    # ✅ 修改顺序：真实标签在前，预测标签在后
    print_confusion_matrix(y_test, np.array(all_preds), label_encoder)
    save_evaluation_results(y_test, np.array(all_preds), label_encoder, model_name=model_name)

    accuracy = np.sum(np.array(all_preds) == y_test) / len(y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")


def evaluate_full_adfnet():
    """评估完整ADF-Net"""
    evaluate_ablation_model("weights/ADFNet_best.pth", "full", "ADFNet_Full_test")


def evaluate_attention_only():
    """评估仅注意力融合模型"""
    evaluate_ablation_model("weights/ADFNet_AttentionOnly_best.pth", "attention_only", "ADFNet_AttentionOnly_test")


def evaluate_interaction_only():
    """评估仅跨粒度交互模型"""
    evaluate_ablation_model("weights/ADFNet_InteractionOnly_best.pth", "interaction_only",
                            "ADFNet_InteractionOnly_test")


def compare_all_models():
    """对比所有模型在测试集上的表现，并保存每个模型的评估结果"""
    print("=" * 60)
    print("所有模型对比（测试集）")
    print("=" * 60)

    X_test, y_test, label_encoder = load_dataset(Config.TEST_DATA_ROOT)
    print(f"测试集加载完成: {X_test.shape}")

    results = {}

    # 1. 基线模型
    print("\n[1/4] 基线双流CNN预测中...")
    baseline_engine = ModelEngine(
        time_model_path="weights/TemporalCNN_best.pth",
        byte_model_path="weights/PayloadCNN_best.pth",
        nb_clf_path="weights/nb_classifier.pkl"
    )
    baseline_preds = []
    for i in range(len(X_test)):
        result = baseline_engine.predict(X_test[i])
        baseline_preds.append(result['label_id'])
    baseline_preds = np.array(baseline_preds)
    baseline_acc = np.sum(baseline_preds == y_test) / len(y_test)
    results['基线双流CNN'] = baseline_acc

    # ✅ 修改顺序：真实标签在前，预测标签在后
    print_confusion_matrix(y_test, baseline_preds, label_encoder)
    save_evaluation_results(y_test, baseline_preds, label_encoder, model_name="StandardModel_test")

    # 2. 仅注意力融合（如果已训练）
    attention_path = "weights/ADFNet_AttentionOnly_best.pth"
    if os.path.exists(attention_path):
        print("\n[2/4] 仅注意力融合预测中...")
        attention_engine = AblationModelEngine(attention_path, "attention_only")
        attention_preds = []
        for i in range(len(X_test)):
            result = attention_engine.predict(X_test[i])
            attention_preds.append(result['label_id'])
        attention_preds = np.array(attention_preds)
        attention_acc = np.sum(attention_preds == y_test) / len(y_test)
        results['仅注意力融合'] = attention_acc

        # ✅ 修改顺序：真实标签在前，预测标签在后
        print_confusion_matrix(y_test, attention_preds, label_encoder)
        save_evaluation_results(y_test, attention_preds, label_encoder, model_name="ADFNet_AttentionOnly_test")
    else:
        print("\n[2/4] 仅注意力融合模型不存在，跳过")
        results['仅注意力融合'] = None

    # 3. 仅跨粒度交互（如果已训练）
    interaction_path = "weights/ADFNet_InteractionOnly_best.pth"
    if os.path.exists(interaction_path):
        print("\n[3/4] 仅跨粒度交互预测中...")
        interaction_engine = AblationModelEngine(interaction_path, "interaction_only")
        interaction_preds = []
        for i in range(len(X_test)):
            result = interaction_engine.predict(X_test[i])
            interaction_preds.append(result['label_id'])
        interaction_preds = np.array(interaction_preds)
        interaction_acc = np.sum(interaction_preds == y_test) / len(y_test)
        results['仅跨粒度交互'] = interaction_acc

        # ✅ 修改顺序：真实标签在前，预测标签在后
        print_confusion_matrix(y_test, interaction_preds, label_encoder)
        save_evaluation_results(y_test, interaction_preds, label_encoder, model_name="ADFNet_InteractionOnly_test")
    else:
        print("\n[3/4] 仅跨粒度交互模型不存在，跳过")
        results['仅跨粒度交互'] = None

    # 4. 完整ADF-Net
    print("\n[4/4] 完整ADF-Net预测中...")
    full_engine = AblationModelEngine("weights/ADFNet_best.pth", "full")
    full_preds = []
    for i in range(len(X_test)):
        result = full_engine.predict(X_test[i])
        full_preds.append(result['label_id'])
    full_preds = np.array(full_preds)
    full_acc = np.sum(full_preds == y_test) / len(y_test)
    results['完整ADF-Net'] = full_acc

    # ✅ 修改顺序：真实标签在前，预测标签在后
    print_confusion_matrix(y_test, full_preds, label_encoder)
    save_evaluation_results(y_test, full_preds, label_encoder, model_name="ADFNet_Full_test")

    # 打印对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    for name, acc in results.items():
        if acc is not None:
            print(f"{name:20} 准确率: {acc:.4f}")

    if results['完整ADF-Net'] is not None and results['基线双流CNN'] is not None:
        print(f"\n完整ADF-Net vs 基线: {(results['完整ADF-Net'] - results['基线双流CNN']) * 100:+.2f}%")

    print("\n✅ 所有模型的评估结果已保存至 results/ 目录")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "standard":
            evaluate_standard_model()
        elif sys.argv[1] == "full":
            evaluate_full_adfnet()
        elif sys.argv[1] == "attention_only":
            evaluate_attention_only()
        elif sys.argv[1] == "interaction_only":
            evaluate_interaction_only()
        elif sys.argv[1] == "compare":
            compare_all_models()
        else:
            print("用法: python test_models.py [standard|full|attention_only|interaction_only|compare]")
    else:
        compare_all_models()