"""
测试特征提取和模型引擎 - 批量测试所有pcap文件
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.feature_extractor import FeatureExtractor
from src.model_engine import ModelEngine

def test_all_pcaps():
    """测试test_pcaps目录下的所有pcap文件"""

    print("=" * 60)
    print("QUIC流量分类模型测试")
    print("=" * 60)

    # 初始化
    print("\n1. 初始化模型...")
    extractor = FeatureExtractor()
    engine = ModelEngine(
        time_model_path="../weights/TemporalCNN_best.pth",
        byte_model_path="../weights/PayloadCNN_best.pth",
        nb_clf_path="../weights/nb_classifier.pkl"
    )

    # 获取测试文件目录
    test_dir = os.path.join(os.path.dirname(__file__), "test_pcaps")
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        print("请先创建 tests/test_pcaps/ 目录")
        return

    # 获取所有pcap文件
    pcap_files = [f for f in os.listdir(test_dir) if f.endswith('.pcap')]
    if not pcap_files:
        print(f"❌ 测试目录中没有pcap文件: {test_dir}")
        print("请先在 tests/test_pcaps/ 目录下放置一些pcap文件")
        return

    print(f"\n2. 找到 {len(pcap_files)} 个测试文件")
    print("-" * 60)

    # 统计结果
    results = []
    success_count = 0
    fail_count = 0

    for i, pcap_file in enumerate(pcap_files, 1):
        pcap_path = os.path.join(test_dir, pcap_file)
        print(f"\n[{i}/{len(pcap_files)}] 测试文件: {pcap_file}")

        # 预测
        result = engine.predict_from_pcap(pcap_path, extractor)

        if result:
            success_count += 1
            results.append({
                'file': pcap_file,
                'result': result
            })
            print(f"   ✅ 预测成功")
            print(f"      预测类别: {result['label']}")
            print(f"      置信度: {result['confidence']:.4f}")
            print(f"      类别ID: {result['label_id']}")

            # 显示概率分布的前几名
            probs = result['probabilities']
            top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            print(f"      前三候选:")
            for idx, prob in top3:
                class_name = engine.class_names[idx]
                print(f"        - {class_name}: {prob:.4f}")
        else:
            fail_count += 1
            print(f"   ❌ 预测失败")

    # 输出统计结果
    print("\n" + "=" * 60)
    print("测试完成统计")
    print("=" * 60)
    print(f"总测试文件: {len(pcap_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"成功率: {success_count/len(pcap_files)*100:.2f}%")

    if results:
        print("\n详细结果:")
        print("-" * 60)
        for r in results:
            print(f"{r['file']:30} → {r['result']['label']:12} (置信度: {r['result']['confidence']:.4f})")

def test_single_file(file_name=None):
    """测试单个文件（可选）"""
    if not file_name:
        return

    print("=" * 60)
    print("测试单个文件")
    print("=" * 60)

    extractor = FeatureExtractor()
    engine = ModelEngine(
        time_model_path="../weights/TemporalCNN_best.pth",
        byte_model_path="../weights/PayloadCNN_best.pth",
        nb_clf_path="../weights/nb_classifier.pkl"
    )

    test_dir = os.path.join(os.path.dirname(__file__), "test_pcaps")
    pcap_path = os.path.join(test_dir, file_name)

    if not os.path.exists(pcap_path):
        print(f"文件不存在: {pcap_path}")
        return

    print(f"\n测试文件: {file_name}")
    result = engine.predict_from_pcap(pcap_path, extractor)

    if result:
        print(f"\n✅ 预测成功")
        print(f"   预测类别: {result['label']}")
        print(f"   置信度: {result['confidence']:.4f}")
        print(f"   类别ID: {result['label_id']}")

        print("\n   所有类别概率:")
        for i, prob in enumerate(result['probabilities']):
            class_name = engine.class_names[i]
            bar = "█" * int(prob * 50)
            print(f"   {class_name:12} [{bar:50}] {prob:.4f}")
    else:
        print(f"❌ 预测失败")

if __name__ == "__main__":
    # 测试所有文件
    test_all_pcaps()