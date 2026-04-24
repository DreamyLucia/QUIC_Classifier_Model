"""
模型服务引擎
加载训练好的模型，提供预测接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import os
from typing import Dict, Any, Optional, List


# 导入模型定义
class TemporalCNN(torch.nn.Module):
    """时序特征模型"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(3),
            torch.nn.Conv1d(16, 32, 5),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 + 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, temporal, stats):
        x = self.conv(temporal.unsqueeze(1))
        x = torch.mean(x, dim=2)
        return self.fc(torch.cat([x, stats], dim=1))


class PayloadCNN(torch.nn.Module):
    """负载特征模型"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(3),
            torch.nn.Conv1d(8, 16, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),
            torch.nn.Conv1d(16, 32, 5),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 + 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, payload, stats):
        x = self.conv(payload.unsqueeze(1))
        x = torch.mean(x, dim=2)
        return self.fc(torch.cat([x, stats], dim=1))


class ModelEngine:
    """模型引擎，封装模型加载和预测"""

    def __init__(self,
                 time_model_path: str,
                 byte_model_path: str,
                 nb_clf_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        初始化模型引擎

        Args:
            time_model_path: 时序CNN模型权重路径
            byte_model_path: 负载CNN模型权重路径
            nb_clf_path: 朴素贝叶斯分类器路径
            device: 运行设备
        """
        self.device = torch.device(device)

        # 类别数
        self.num_classes = 8
        self.class_names = ['Nkiri', 'bilibili', 'edge', 'kwai',
                            'tencentnews', 'tencentvideo', 'tiktok', 'xiaohongshu']

        # 特征维度参数
        self.time_features_dim = 120
        self.byte_features_dim = 900
        self.stat_features_dim = 4

        # 加载模型
        self.time_model = TemporalCNN(self.num_classes).to(self.device)
        self.byte_model = PayloadCNN(self.num_classes).to(self.device)

        # 加载权重
        if os.path.exists(time_model_path):
            self.time_model.load_state_dict(torch.load(time_model_path, map_location=self.device))
            self.time_model.eval()
            print(f"时序模型加载成功: {time_model_path}")
        else:
            print(f"警告: 时序模型文件不存在 {time_model_path}")

        if os.path.exists(byte_model_path):
            self.byte_model.load_state_dict(torch.load(byte_model_path, map_location=self.device))
            self.byte_model.eval()
            print(f"负载模型加载成功: {byte_model_path}")
        else:
            print(f"警告: 负载模型文件不存在 {byte_model_path}")

        # 加载朴素贝叶斯分类器
        self.nb_clf = None
        if nb_clf_path and os.path.exists(nb_clf_path):
            self.nb_clf = joblib.load(nb_clf_path)
            print(f"集成模型加载成功: {nb_clf_path}")

    def _get_probs(self, model, x, stats):
        """获取模型输出的概率"""
        model.eval()
        with torch.no_grad():
            outputs = model(torch.FloatTensor(x).to(self.device),
                            torch.FloatTensor(stats).to(self.device))
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        预测单个样本

        Args:
            features: 1024维特征向量

        Returns:
            预测结果: {'label': str, 'confidence': float, 'probabilities': list}
        """
        # 检查特征维度
        if len(features) != self.time_features_dim + self.byte_features_dim + self.stat_features_dim:
            raise ValueError(f"特征维度错误: 期望 {self.total_features_dim}, 实际 {len(features)}")

        # 特征分解
        time_feat = features[:self.time_features_dim].reshape(1, -1)
        byte_feat = features[self.time_features_dim:self.time_features_dim + self.byte_features_dim].reshape(1, -1)
        stats_feat = features[-self.stat_features_dim:].reshape(1, -1)

        # 获取两个模型的概率
        time_probs = self._get_probs(self.time_model, time_feat, stats_feat)
        byte_probs = self._get_probs(self.byte_model, byte_feat, stats_feat)

        # 如果有集成模型，使用集成；否则用平均
        if self.nb_clf is not None:
            ensemble_feat = np.concatenate([time_probs, byte_probs], axis=1)
            pred_id = self.nb_clf.predict(ensemble_feat)[0]
            # 获取所有类别的概率（朴素贝叶斯不一定有概率）
            probs = (time_probs[0] + byte_probs[0]) / 2  # 还是用平均作为概率
        else:
            # 简单平均
            probs = (time_probs[0] + byte_probs[0]) / 2
            pred_id = np.argmax(probs)

        confidence = probs[pred_id]

        return {
            'label': self.class_names[pred_id],
            'label_id': int(pred_id),
            'confidence': float(confidence),
            'probabilities': probs.tolist()
        }

    def predict_from_pcap(self, pcap_path: str, feature_extractor) -> Optional[Dict[str, Any]]:
        """
        从pcap文件直接预测

        Args:
            pcap_path: pcap文件路径
            feature_extractor: FeatureExtractor实例

        Returns:
            预测结果
        """
        features = feature_extractor.extract_from_pcap(pcap_path)
        if features is None:
            return None
        return self.predict(features)

    def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        """批量预测"""
        return [self.predict(feats) for feats in features_list]


class CrossGranularityInteraction(nn.Module):
    """跨粒度交互模块"""

    def __init__(self, feat_dim=64):
        super().__init__()
        self.interaction = nn.Linear(feat_dim * 2, feat_dim)
        self.relu = nn.ReLU()

    def forward(self, f_t, f_p):
        concat = torch.cat([f_t, f_p], dim=-1)
        f_inter = self.interaction(concat)
        f_inter = self.relu(f_inter)
        return f_inter


class AttentionFusion(nn.Module):
    """注意力融合层"""

    def __init__(self, feat_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, f_t, f_p):
        concat = torch.cat([f_t, f_p], dim=-1)
        alpha = self.attention(concat)
        f_fused = alpha[:, 0:1] * f_t + alpha[:, 1:2] * f_p
        return f_fused, alpha


class ADFNet(nn.Module):
    """注意力增强的双流多粒度融合网络"""

    def __init__(self, num_classes=8, feat_dim=64):
        super().__init__()

        # 时间特征提取器
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(16, 32, 5),
            nn.ReLU()
        )
        self.temporal_fc = nn.Linear(32 + 4, feat_dim)

        # 字节特征提取器
        self.payload_conv = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.ReLU()
        )
        self.payload_fc = nn.Linear(32 + 4, feat_dim)

        # 跨粒度交互模块
        self.cross_interaction = CrossGranularityInteraction(feat_dim)

        # 注意力融合层
        self.attention_fusion = AttentionFusion(feat_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, temporal, payload, stats):
        """
        Args:
            temporal: 时间特征 [batch, 120]
            payload: 字节特征 [batch, 900]
            stats: 统计特征 [batch, 4]
        Returns:
            output: 分类结果 [batch, num_classes]
            alpha: 注意力权重
        """
        # 1. 时间特征提取
        t = temporal.unsqueeze(1)
        t = self.temporal_conv(t)
        t = torch.mean(t, dim=2)
        t = torch.cat([t, stats], dim=1)
        f_t = self.temporal_fc(t)

        # 2. 字节特征提取
        p = payload.unsqueeze(1)
        p = self.payload_conv(p)
        p = torch.mean(p, dim=2)
        p = torch.cat([p, stats], dim=1)
        f_p = self.payload_fc(p)

        # 3. 跨粒度交互
        f_inter = self.cross_interaction(f_t, f_p)

        # 4. 注意力融合
        f_fused, alpha = self.attention_fusion(f_inter, f_p)

        # 5. 分类
        output = self.classifier(f_fused)

        return output, alpha


class ADFNetModelEngine:
    """ADF-Net 模型引擎"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_classes = 8
        self.class_names = ['Nkiri', 'bilibili', 'edge', 'kwai',
                            'tencentnews', 'tencentvideo', 'tiktok', 'xiaohongshu']

        self.time_features_dim = 120
        self.byte_features_dim = 900
        self.stat_features_dim = 4

        self.model = ADFNet(num_classes=self.num_classes).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ ADF-Net 模型加载成功: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """预测单个样本"""
        time_feat = features[:self.time_features_dim].reshape(1, -1)
        byte_feat = features[self.time_features_dim:self.time_features_dim + self.byte_features_dim].reshape(1, -1)
        stats_feat = features[-self.stat_features_dim:].reshape(1, -1)

        time_tensor = torch.FloatTensor(time_feat).to(self.device)
        byte_tensor = torch.FloatTensor(byte_feat).to(self.device)
        stats_tensor = torch.FloatTensor(stats_feat).to(self.device)

        with torch.no_grad():
            outputs, alpha = self.model(time_tensor, byte_tensor, stats_tensor)
            probs = torch.softmax(outputs, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()

        return {
            'label': self.class_names[pred_id],
            'label_id': pred_id,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy().tolist(),
            'attention_weights': alpha[0].cpu().numpy().tolist()
        }

    def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [self.predict(feats) for feats in features_list]