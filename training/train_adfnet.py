import os
import sys

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.model_engine import ADFNet
from training.mix_ import load_dataset, Config


def train_adfnet():
    """训练 ADF-Net 模型"""

    print("=" * 60)
    print("训练 ADF-Net 注意力增强双流多粒度融合网络")
    print("=" * 60)

    # 1. 加载训练集
    print("\n[1/5] 加载训练集...")
    X_train, y_train, label_encoder = load_dataset(Config.TRAIN_DATA_ROOT)
    print(f"训练集加载完成: {X_train.shape}, 类别数: {len(label_encoder.classes_)}")

    # 2. 加载验证集
    print("\n[2/5] 加载验证集...")
    X_val, y_val, _ = load_dataset(Config.VAL_DATA_ROOT)
    print(f"验证集加载完成: {X_val.shape}")

    # 3. 特征分解 - 训练集
    time_train = X_train[:, :Config.TIME_FEATURES]
    byte_train = X_train[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_train = X_train[:, -Config.STAT_FEATURES:]

    # 特征分解 - 验证集
    time_val = X_val[:, :Config.TIME_FEATURES]
    byte_val = X_val[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_val = X_val[:, -Config.STAT_FEATURES:]

    # 4. 创建 DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(time_train),
        torch.FloatTensor(byte_train),
        torch.FloatTensor(stats_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(time_val),
        torch.FloatTensor(byte_val),
        torch.FloatTensor(stats_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 5. 创建模型
    print("\n[3/5] 创建 ADF-Net 模型...")
    model = ADFNet(num_classes=len(label_encoder.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    num_epochs = 30
    best_val_acc = 0.0

    print("\n[4/5] 开始训练...")
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for temporal, payload, stats, labels in train_loader:
            temporal = temporal.to(device)
            payload = payload.to(device)
            stats = stats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(temporal, payload, stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for temporal, payload, stats, labels in val_loader:
                temporal = temporal.to(device)
                payload = payload.to(device)
                stats = stats.to(device)
                labels = labels.to(device)
                outputs, _ = model(temporal, payload, stats)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "weights/ADFNet_best.pth")
            print(f"  → 保存最佳模型 (Acc: {val_acc:.4f})")

    # 5. 在验证集上评估（使用最佳模型）
    print("\n[5/5] 评估最佳模型...")

    # 加载最佳模型
    model.load_state_dict(torch.load("weights/ADFNet_best.pth"))
    model.eval()

    # 在验证集上预测
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for temporal, payload, stats, labels in val_loader:
            temporal = temporal.to(device)
            payload = payload.to(device)
            stats = stats.to(device)
            outputs, _ = model(temporal, payload, stats)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    test_acc = np.sum(all_preds == all_labels) / len(all_labels)
    print(f"验证集准确率: {test_acc:.4f}")
    print(f"模型已保存至: weights/ADFNet_best.pth")


if __name__ == "__main__":
    train_adfnet()