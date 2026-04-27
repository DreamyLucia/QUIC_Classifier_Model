import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

from src.model_engine import (
    ADFNet_AttentionOnly,
    ADFNet_InteractionOnly,
    ADFNet
)
from training.mix_ import load_dataset, Config


def train_ablation(model_type, model_name):
    """训练消融模型"""

    print("=" * 60)
    print(f"训练消融模型: {model_name}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载训练集...")
    X_train, y_train, label_encoder = load_dataset(Config.TRAIN_DATA_ROOT)
    print(f"训练集加载完成: {X_train.shape}")

    print("\n[2/5] 加载验证集...")
    X_val, y_val, _ = load_dataset(Config.VAL_DATA_ROOT)
    print(f"验证集加载完成: {X_val.shape}")

    # 2. 特征分解
    time_train = X_train[:, :Config.TIME_FEATURES]
    byte_train = X_train[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_train = X_train[:, -Config.STAT_FEATURES:]

    time_val = X_val[:, :Config.TIME_FEATURES]
    byte_val = X_val[:, Config.TIME_FEATURES:-Config.STAT_FEATURES]
    stats_val = X_val[:, -Config.STAT_FEATURES:]

    # 3. 创建 DataLoader
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

    # 4. 创建模型
    print("\n[3/5] 创建模型...")
    if model_type == 'attention_only':
        model = ADFNet_AttentionOnly(num_classes=len(label_encoder.classes_))
    elif model_type == 'interaction_only':
        model = ADFNet_InteractionOnly(num_classes=len(label_encoder.classes_))
    else:
        model = ADFNet(num_classes=len(label_encoder.classes_))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练配置
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"weights/{model_name}_best.pth")
            print(f"  → 保存最佳模型 (Acc: {val_acc:.4f})")

    # 6. 评估
    print("\n[5/5] 评估最佳模型...")
    model.load_state_dict(torch.load(f"weights/{model_name}_best.pth"))
    model.eval()

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

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['attention_only', 'interaction_only', 'full'],
                        help='模型类型')
    args = parser.parse_args()

    model_names = {
        'attention_only': 'ADFNet_AttentionOnly',
        'interaction_only': 'ADFNet_InteractionOnly',
        'full': 'ADFNet_Full'
    }

    train_ablation(args.model, model_names[args.model])