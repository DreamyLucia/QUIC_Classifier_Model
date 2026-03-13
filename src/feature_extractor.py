"""
特征提取模块
从pcap文件中提取模型所需的特征
"""

import numpy as np
import dpkt
import os
from typing import Optional, List, Dict, Any


class FeatureExtractor:
    """特征提取器，封装process_pcap功能"""

    def __init__(self):
        # 特征维度参数（必须与训练时一致）
        self.time_features_dim = 120
        self.byte_features_dim = 900
        self.stat_features_dim = 4
        self.total_features_dim = self.time_features_dim + self.byte_features_dim + self.stat_features_dim
        self.max_packets = 60

    def _process_pcap(self, pcap_path: str) -> Optional[np.ndarray]:
        """
        从pcap文件提取特征（内部实现）
        这是从mix_.py中提取的核心代码
        """
        try:
            with open(pcap_path, 'rb') as f:
                try:
                    reader = dpkt.pcap.Reader(f)
                except (ValueError, dpkt.dpkt.NeedData):
                    reader = dpkt.pcapng.Reader(f)

                # 预初始化特征数组
                time_features = np.zeros(self.time_features_dim, dtype=np.float32)
                byte_stream = np.zeros(self.byte_features_dim, dtype=np.uint8)
                size_counts = np.zeros(3, dtype=np.int32)
                time_deltas = []

                client_ip = None
                prev_ts = None
                byte_pos = 0
                packet_count = 0

                for ts, buf in reader:
                    if packet_count >= self.max_packets:
                        break

                    try:
                        # 解析链路层
                        if buf.startswith(b'\x00\x00'[:2]):  # SLL检测
                            sll = dpkt.sll.SLL(buf)
                            network_layer = sll.data
                        else:
                            eth = dpkt.ethernet.Ethernet(buf)
                            network_layer = eth.data

                        # 确定IP版本
                        if not (isinstance(network_layer, dpkt.ip.IP) or isinstance(network_layer, dpkt.ip6.IP6)):
                            continue

                        # 设置客户端IP（第一个遇到的IP包）
                        if client_ip is None:
                            client_ip = network_layer.src

                        # 计算方向
                        direction = 1 if network_layer.src == client_ip else 0

                        # 时间特征处理
                        if prev_ts is not None:
                            delta = ts - prev_ts
                        else:
                            delta = 0.0
                        prev_ts = ts

                        # 填充时间特征数组
                        idx = packet_count * 2
                        if idx + 1 < self.time_features_dim:
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
                        if hasattr(network_layer, 'data') and hasattr(network_layer.data, 'data'):
                            if isinstance(network_layer.data, dpkt.udp.UDP):
                                payload = network_layer.data.data
                                payload_len = len(payload)
                                if byte_pos < self.byte_features_dim:
                                    copy_len = min(payload_len, self.byte_features_dim - byte_pos)
                                    byte_stream[byte_pos:byte_pos + copy_len] = np.frombuffer(
                                        payload[:copy_len], dtype=np.uint8)
                                    byte_pos += copy_len

                        packet_count += 1

                    except Exception:
                        continue

                # 统计特征计算
                total = packet_count if packet_count > 0 else 1
                stats = [
                    size_counts[0] / total,
                    size_counts[1] / total,
                    size_counts[2] / total,
                    np.mean(time_deltas) if time_deltas else 0.0
                ]

                # 合并特征
                features = np.concatenate([
                    time_features,
                    byte_stream.astype(np.float32),
                    np.array(stats, dtype=np.float32)
                ])

                return features if len(features) == self.total_features_dim else None

        except Exception as e:
            print(f"处理文件{pcap_path}出错: {str(e)}")
            return None

    def extract_from_pcap(self, pcap_path: str) -> Optional[np.ndarray]:
        """
        从pcap文件提取特征（对外接口）

        Args:
            pcap_path: pcap文件路径

        Returns:
            特征向量 (1024维) 或 None（提取失败）
        """
        if not os.path.exists(pcap_path):
            print(f"文件不存在: {pcap_path}")
            return None

        features = self._process_pcap(pcap_path)

        if features is not None:
            return np.array(features, dtype=np.float32)
        return None

    def extract_batch(self, pcap_paths: List[str]) -> List[Optional[np.ndarray]]:
        """批量提取特征"""
        return [self.extract_from_pcap(path) for path in pcap_paths]