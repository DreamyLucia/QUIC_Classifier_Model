"""
双流CNN + 朴素贝叶斯集成模型训练脚本
用于QUIC流量业务分类
"""

import os
import dpkt
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import multiprocessing
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# -------------------- 配置参数 --------------------
class Config:
    TRAIN_DATA_ROOT = "./data/train"   # 训练集目录
    VAL_DATA_ROOT = "./data/val"       # 验证集目录
    TEST_DATA_ROOT = "./data/test"     # 测试集目录
    MAX_PACKETS = 60
    TIME_FEATURES = 120
    BYTE_FEATURES = 900
    STAT_FEATURES = 4
    TOTAL_FEATURES = TIME_FEATURES + BYTE_FEATURES + STAT_FEATURES
    NUM_CLASSES = 8  # 数据集类别数


# -------------------- 数据预处理 --------------------
def process_pcap(pcap_path):
    """从pcap文件提取特征"""
    try:
        with open(pcap_path, 'rb') as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except (ValueError, dpkt.dpkt.NeedData):
                reader = dpkt.pcapng.Reader(f)

            # 预初始化特征数组
            time_features = np.zeros(Config.TIME_FEATURES, dtype=np.float32)
            byte_stream = np.zeros(Config.BYTE_FEATURES, dtype=np.uint8)
            size_counts = np.zeros(3, dtype=np.int32)
            time_deltas = []

            client_ip = None
            ip_type = None
            prev_ts = None
            byte_pos = 0
            packet_count = 0

            for ts, buf in reader:
                if packet_count >= Config.MAX_PACKETS:
                    break

                try:
                    # 优化链路层解析
                    if buf.startswith(b'\x00\x00'[:2]):  # SLL检测
                        sll = dpkt.sll.SLL(buf)
                        network_layer = sll.data
                    else:
                        eth = dpkt.ethernet.Ethernet(buf)
                        network_layer = eth.data

                    # 确定IP版本
                    if isinstance(network_layer, dpkt.ip.IP):
                        version = 4
                    elif isinstance(network_layer, dpkt.ip6.IP6):
                        version = 6
                    else:
                        continue

                    # 设置客户端IP（第一个遇到的IP包）
                    if client_ip is None:
                        client_ip = network_layer.src
                        ip_type = version

                    # 计算方向（直接比较原始字节）
                    direction = 1 if network_layer.src == client_ip else 0

                    # 时间特征处理
                    if prev_ts is not None:
                        delta = ts - prev_ts
                    else:
                        delta = 0.0
                    prev_ts = ts

                    # 填充时间特征数组
                    idx = packet_count * 2
                    if idx + 1 < Config.TIME_FEATURES:
                        pkt_size = -len(network_layer) if direction else len(network_layer)
                        time_features[idx] = delta
                        time_features[idx + 1] = pkt_size

                    # 包大小统计
                    size = len(network_layer)
                    if size < 400:
                        size_counts[0] += 1
                    elif 400 <= size <= 800:
                        size_counts[1] += 1
                    else:
                        size_counts[2] += 1
                    time_deltas.append(delta)

                    # 有效负载处理
                    if isinstance(network_layer.data, dpkt.udp.UDP):
                        payload = network_layer.data.data
                        payload_len = len(payload)
                        if byte_pos < Config.BYTE_FEATURES:
                            copy_len = min(payload_len, Config.BYTE_FEATURES - byte_pos)
                            byte_stream[byte_pos:byte_pos + copy_len] = np.frombuffer(
                                payload[:copy_len], dtype=np.uint8)
                            byte_pos += copy_len

                    packet_count += 1

                except Exception as e:
                    continue

            # 统计特征计算
            total = packet_count if packet_count > 0 else 1
            stats = [
                size_counts[0] / total,
                size_counts[1] / total,
                size_counts[2] / total,
                np.mean(time_deltas) if time_deltas else 0.0
            ]

            # 合并特征并转换为float32
            features = np.concatenate([
                time_features,
                byte_stream.astype(np.float32),
                np.array(stats, dtype=np.float32)
            ])

            return features.tolist() if len(features) == Config.TOTAL_FEATURES else None

    except Exception as e:
        print(f"处理文件{pcap_path}出错: {str(e)}")
        return None


def process_pcap_wrapper(pcap_path, label):
    """用于多进程的包装函数"""
    return process_pcap(pcap_path), label


def load_dataset(root_dir):
    """加载指定目录的数据集，返回特征、标签和标签编码器"""
    features = []
    labels = []

    # 初始化标签编码器
    le = LabelEncoder()
    class_names = []

    # 扫描有效类别
    valid_classes = []
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            valid_classes.append(class_name)
    valid_classes = sorted(valid_classes)

    # 创建进程池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for class_name in valid_classes:
        class_path = os.path.join(root_dir, class_name)
        pcap_files = [
            os.path.join(class_path, fname)
            for fname in os.listdir(class_path)
            if fname.lower().endswith('.pcap')
        ]

        # 并行处理文件
        results = pool.starmap(process_pcap_wrapper,
                               [(pcap, class_name) for pcap in pcap_files])

        valid = 0
        for feat, label in results:
            if feat is not None and len(feat) == Config.TOTAL_FEATURES:
                features.append(feat)
                class_names.append(label)
                valid += 1
        print(f"类别 [{class_name}] 加载完成，有效样本: {valid}")

    pool.close()
    pool.join()

    # 编码标签
    encoded_labels = le.fit_transform(class_names)

    return np.array(features, dtype=np.float32), encoded_labels, le


# -------------------- 模型定义 --------------------
class TemporalCNN(nn.Module):
    """时序特征模型"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(16, 32, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, temporal, stats):
        x = self.conv(temporal.unsqueeze(1))
        x = torch.mean(x, dim=2)
        return self.fc(torch.cat([x, stats], dim=1))


class PayloadCNN(nn.Module):
    """负载特征模型"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, payload, stats):
        x = self.conv(payload.unsqueeze(1))
        x = torch.mean(x, dim=2)
        return self.fc(torch.cat([x, stats], dim=1))


# -------------------- 训练流程 --------------------
def train_model(model, x_data, stats, labels, num_classes):
    """通用模型训练函数"""
    # 分割验证集
    x_train, x_val, s_train, s_val, y_train, y_val = train_test_split(
        x_data, stats, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train),
        torch.FloatTensor(s_train),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val),
        torch.FloatTensor(s_val),
        torch.LongTensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    for epoch in range(20):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for x, s, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x, s)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, s, y in val_loader:
                outputs = model(x, s)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        val_acc = correct / total

        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./weights/{model.__class__.__name__}_best.pth")

    # 加载最佳模型
    model.load_state_dict(torch.load(f"./weights/{model.__class__.__name__}_best.pth"))
    return model


def main():
    """主训练函数 - 使用训练集和验证集"""
    # 加载训练集
    print("=" * 50)
    print("加载训练集...")
    X_train, y_train, label_encoder = load_dataset(Config.TRAIN_DATA_ROOT)
    print(f"训练集加载完成，总样本: {len(X_train)}，类别数: {len(label_encoder.classes_)}")

    # 加载验证集
    print("\n加载验证集...")
    X_val, y_val, _ = load_dataset(Config.VAL_DATA_ROOT)
    print(f"验证集加载完成，总样本: {len(X_val)}")

    # 特征分解 - 训练集
    time_train = X_train[:, :Config.TIME_FEATURES]
    byte_train = X_train[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_train = X_train[:, -Config.STAT_FEATURES:]

    # 特征分解 - 验证集
    time_val = X_val[:, :Config.TIME_FEATURES]
    byte_val = X_val[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_val = X_val[:, -Config.STAT_FEATURES:]

    # 训练时序模型
    print("\n训练时序模型中...")
    time_model = TemporalCNN(len(label_encoder.classes_))
    time_model = train_model(time_model, time_train, stats_train, y_train, len(label_encoder.classes_))

    # 训练负载模型
    print("\n训练字节模型中...")
    byte_model = PayloadCNN(len(label_encoder.classes_))
    byte_model = train_model(byte_model, byte_train, stats_train, y_train, len(label_encoder.classes_))

    # 获取概率预测
    def get_probs(model, x, stats):
        model.eval()
        with torch.no_grad():
            outputs = model(torch.FloatTensor(x), torch.FloatTensor(stats))
            return torch.softmax(outputs, dim=1).numpy()

    # 训练集成模型
    print("\n训练集成模型...")
    time_train_probs = get_probs(time_model, time_train, stats_train)
    byte_train_probs = get_probs(byte_model, byte_train, stats_train)
    X_ensemble_train = np.concatenate([time_train_probs, byte_train_probs], axis=1)

    # 训练贝叶斯分类器
    nb_clf = MultinomialNB().fit(X_ensemble_train, y_train)

    # 保存集成模型
    import joblib
    joblib.dump(nb_clf, "./weights/nb_classifier.pkl")

    # 在验证集上评估
    print("\n在验证集上评估...")
    time_val_probs = get_probs(time_model, time_val, stats_val)
    byte_val_probs = get_probs(byte_model, byte_val, stats_val)
    X_ensemble_val = np.concatenate([time_val_probs, byte_val_probs], axis=1)
    final_preds = nb_clf.predict(X_ensemble_val)

    print("\n类别映射参考:")
    print({i: name for i, name in enumerate(label_encoder.classes_)})


def load_test_data():
    """单独加载测试集（用于评估脚本）"""
    return load_dataset(Config.TEST_DATA_ROOT)


if __name__ == "__main__":
    main()