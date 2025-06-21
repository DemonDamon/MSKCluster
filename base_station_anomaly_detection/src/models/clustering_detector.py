"""
基于聚类的异常检测器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .base_detector import UnsupervisedDetector
from ..utils.geo_utils import calculate_distance
from ..utils.time_utils import is_daytime


class ClusteringAnomalyDetector(UnsupervisedDetector):
    """基于聚类的异常检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化聚类检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # DBSCAN参数
        self.eps = self.config.get('eps', 0.02)  # 约2.2km
        self.min_samples = self.config.get('min_samples', 3)
        self.time_window = self.config.get('time_window', 24)  # 小时
        
        # LOF参数
        self.lof_neighbors = self.config.get('lof_neighbors', 20)
        self.lof_threshold = self.config.get('lof_threshold', 1.5)
        
        # 模型
        self.dbscan_model = None
        self.lof_model = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # 聚类结果
        self.cluster_centers = {}
        self.cluster_stats = {}
    
    def _fit_model(self, X: pd.DataFrame) -> None:
        """
        训练聚类模型
        
        Args:
            X: 特征数据
        """
        # 准备特征
        features = self._prepare_features(X)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练DBSCAN
        self.dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = self.dbscan_model.fit_predict(features_scaled)
        
        # 分析聚类结果
        self._analyze_clusters(X, cluster_labels)
        
        # 训练LOF
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination='auto',
            novelty=True
        )
        self.lof_model.fit(features_scaled)
        
        # 训练Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination='auto',
            random_state=42
        )
        self.isolation_forest.fit(features_scaled)
        
        self.logger.info(f"Clustering model trained with {len(np.unique(cluster_labels))} clusters")
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        准备聚类特征
        
        Args:
            X: 输入数据
        
        Returns:
            特征矩阵
        """
        features = []
        
        # 基础位置特征
        features.append(X['latitude'].values)
        features.append(X['longitude'].values)
        
        # 时间特征
        if 'timestamp' in X.columns:
            # 小时 (周期性编码)
            hours = X['timestamp'].dt.hour
            features.append(np.sin(2 * np.pi * hours / 24))
            features.append(np.cos(2 * np.pi * hours / 24))
            
            # 星期几 (周期性编码)
            weekdays = X['timestamp'].dt.weekday
            features.append(np.sin(2 * np.pi * weekdays / 7))
            features.append(np.cos(2 * np.pi * weekdays / 7))
        
        # 运动特征
        if 'speed_kmh' in X.columns:
            features.append(X['speed_kmh'].values)
        
        if 'distance' in X.columns:
            features.append(X['distance'].values)
        
        if 'bearing' in X.columns:
            # 方向 (周期性编码)
            bearings = X['bearing'].values
            features.append(np.sin(np.radians(bearings)))
            features.append(np.cos(np.radians(bearings)))
        
        # 滑动窗口统计特征
        window_features = self._calculate_window_features(X)
        if window_features is not None:
            features.extend(window_features)
        
        return np.column_stack(features)
    
    def _calculate_window_features(self, X: pd.DataFrame, window_size: int = 5) -> Optional[List[np.ndarray]]:
        """
        计算滑动窗口特征
        
        Args:
            X: 输入数据
            window_size: 窗口大小
        
        Returns:
            窗口特征列表
        """
        if len(X) < window_size:
            return None
        
        features = []
        
        # 位置方差
        lat_var = X['latitude'].rolling(window_size, center=True).var().fillna(0)
        lon_var = X['longitude'].rolling(window_size, center=True).var().fillna(0)
        features.extend([lat_var.values, lon_var.values])
        
        # 速度统计
        if 'speed_kmh' in X.columns:
            speed_mean = X['speed_kmh'].rolling(window_size, center=True).mean().fillna(0)
            speed_std = X['speed_kmh'].rolling(window_size, center=True).std().fillna(0)
            features.extend([speed_mean.values, speed_std.values])
        
        # 距离统计
        if 'distance' in X.columns:
            distance_sum = X['distance'].rolling(window_size, center=True).sum().fillna(0)
            features.append(distance_sum.values)
        
        return features
    
    def _analyze_clusters(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        """
        分析聚类结果
        
        Args:
            X: 输入数据
            cluster_labels: 聚类标签
        """
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = cluster_labels == label
            cluster_data = X[cluster_mask]
            
            # 计算聚类中心
            center_lat = cluster_data['latitude'].mean()
            center_lon = cluster_data['longitude'].mean()
            self.cluster_centers[label] = (center_lat, center_lon)
            
            # 计算聚类统计
            stats = {
                'size': len(cluster_data),
                'lat_std': cluster_data['latitude'].std(),
                'lon_std': cluster_data['longitude'].std(),
                'radius': self._calculate_cluster_radius(cluster_data, center_lat, center_lon)
            }
            
            # 时间统计
            if 'timestamp' in cluster_data.columns:
                stats['time_span'] = (cluster_data['timestamp'].max() - cluster_data['timestamp'].min()).total_seconds() / 3600
                stats['avg_hour'] = cluster_data['timestamp'].dt.hour.mean()
            
            # 运动统计
            if 'speed_kmh' in cluster_data.columns:
                stats['avg_speed'] = cluster_data['speed_kmh'].mean()
                stats['max_speed'] = cluster_data['speed_kmh'].max()
            
            self.cluster_stats[label] = stats
    
    def _calculate_cluster_radius(self, cluster_data: pd.DataFrame, center_lat: float, center_lon: float) -> float:
        """
        计算聚类半径
        
        Args:
            cluster_data: 聚类数据
            center_lat: 中心纬度
            center_lon: 中心经度
        
        Returns:
            聚类半径 (米)
        """
        distances = cluster_data.apply(
            lambda row: calculate_distance(
                row['latitude'], row['longitude'],
                center_lat, center_lon
            ), axis=1
        )
        return distances.quantile(0.95)  # 95分位数作为半径
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X: 特征数据
        
        Returns:
            异常概率
        """
        self._check_fitted()
        X = self._validate_input(X)
        
        # 准备特征
        features = self._prepare_features(X)
        features_scaled = self.scaler.transform(features)
        
        # 获取各种异常分数
        dbscan_scores = self._get_dbscan_scores(X, features_scaled)
        lof_scores = self._get_lof_scores(features_scaled)
        isolation_scores = self._get_isolation_scores(features_scaled)
        
        # 组合分数
        combined_scores = self._combine_scores(dbscan_scores, lof_scores, isolation_scores)
        
        return combined_scores
    
    def _get_dbscan_scores(self, X: pd.DataFrame, features_scaled: np.ndarray) -> np.ndarray:
        """
        获取DBSCAN异常分数
        
        Args:
            X: 原始数据
            features_scaled: 标准化特征
        
        Returns:
            DBSCAN异常分数
        """
        # 预测聚类标签
        cluster_labels = self.dbscan_model.fit_predict(features_scaled)
        
        scores = np.zeros(len(X))
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # 噪声点
                scores[i] = 1.0
            else:
                # 计算到聚类中心的距离
                if label in self.cluster_centers:
                    center_lat, center_lon = self.cluster_centers[label]
                    distance = calculate_distance(
                        X.iloc[i]['latitude'], X.iloc[i]['longitude'],
                        center_lat, center_lon
                    )
                    
                    # 基于距离和聚类半径计算分数
                    if label in self.cluster_stats:
                        radius = self.cluster_stats[label]['radius']
                        scores[i] = min(1.0, distance / max(radius, 100))  # 避免除零
                    else:
                        scores[i] = 0.5  # 未知聚类的中等分数
        
        return scores
    
    def _get_lof_scores(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        获取LOF异常分数
        
        Args:
            features_scaled: 标准化特征
        
        Returns:
            LOF异常分数
        """
        # LOF分数 (负值表示正常，正值表示异常)
        lof_scores = self.lof_model.decision_function(features_scaled)
        
        # 转换为0-1范围的异常概率
        # LOF分数越小(越负)越异常
        normalized_scores = 1 / (1 + np.exp(lof_scores))  # Sigmoid变换
        
        return normalized_scores
    
    def _get_isolation_scores(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        获取Isolation Forest异常分数
        
        Args:
            features_scaled: 标准化特征
        
        Returns:
            Isolation Forest异常分数
        """
        # Isolation Forest分数 (负值表示异常)
        isolation_scores = self.isolation_forest.decision_function(features_scaled)
        
        # 转换为0-1范围的异常概率
        # 分数越小越异常
        min_score = isolation_scores.min()
        max_score = isolation_scores.max()
        
        if max_score > min_score:
            normalized_scores = 1 - (isolation_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(isolation_scores)
        
        return normalized_scores
    
    def _combine_scores(self, dbscan_scores: np.ndarray, 
                       lof_scores: np.ndarray, 
                       isolation_scores: np.ndarray) -> np.ndarray:
        """
        组合多种异常分数
        
        Args:
            dbscan_scores: DBSCAN分数
            lof_scores: LOF分数
            isolation_scores: Isolation Forest分数
        
        Returns:
            组合异常分数
        """
        # 权重配置
        dbscan_weight = self.config.get('dbscan_weight', 0.4)
        lof_weight = self.config.get('lof_weight', 0.4)
        isolation_weight = self.config.get('isolation_weight', 0.2)
        
        # 加权平均
        combined_scores = (
            dbscan_weight * dbscan_scores +
            lof_weight * lof_scores +
            isolation_weight * isolation_scores
        )
        
        return combined_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        scores = self.predict_proba(X)
        threshold = self.config.get('threshold', 0.5)
        return (scores > threshold).astype(int)
    
    def get_cluster_info(self) -> Dict:
        """
        获取聚类信息
        
        Returns:
            聚类信息字典
        """
        return {
            'cluster_centers': self.cluster_centers,
            'cluster_stats': self.cluster_stats,
            'num_clusters': len(self.cluster_centers)
        }
    
    def _get_model_data(self) -> Dict:
        """获取模型特定数据"""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'time_window': self.time_window,
            'lof_neighbors': self.lof_neighbors,
            'lof_threshold': self.lof_threshold,
            'cluster_centers': self.cluster_centers,
            'cluster_stats': self.cluster_stats,
            'scaler': self.scaler,
            'dbscan_model': self.dbscan_model,
            'lof_model': self.lof_model,
            'isolation_forest': self.isolation_forest
        }
    
    def _set_model_data(self, model_data: Dict) -> None:
        """设置模型特定数据"""
        self.eps = model_data.get('eps', 0.02)
        self.min_samples = model_data.get('min_samples', 3)
        self.time_window = model_data.get('time_window', 24)
        self.lof_neighbors = model_data.get('lof_neighbors', 20)
        self.lof_threshold = model_data.get('lof_threshold', 1.5)
        self.cluster_centers = model_data.get('cluster_centers', {})
        self.cluster_stats = model_data.get('cluster_stats', {})
        self.scaler = model_data.get('scaler', StandardScaler())
        self.dbscan_model = model_data.get('dbscan_model')
        self.lof_model = model_data.get('lof_model')
        self.isolation_forest = model_data.get('isolation_forest')


class SpatioTemporalClusteringDetector(ClusteringAnomalyDetector):
    """时空聚类异常检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化时空聚类检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 时空参数
        self.spatial_eps = self.config.get('spatial_eps', 0.01)  # 空间邻域
        self.temporal_eps = self.config.get('temporal_eps', 3600)  # 时间邻域(秒)
        self.space_time_ratio = self.config.get('space_time_ratio', 1.0)  # 时空权重比
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        准备时空聚类特征
        
        Args:
            X: 输入数据
        
        Returns:
            时空特征矩阵
        """
        features = []
        
        # 空间特征 (标准化)
        features.append(X['latitude'].values)
        features.append(X['longitude'].values)
        
        # 时间特征
        if 'timestamp' in X.columns:
            # 转换为数值时间戳 (秒)
            timestamps = X['timestamp'].astype(np.int64) // 10**9
            # 标准化时间戳
            timestamps_normalized = (timestamps - timestamps.min()) / max(timestamps.max() - timestamps.min(), 1)
            features.append(timestamps_normalized * self.space_time_ratio)
        
        # 运动特征
        if 'speed_kmh' in X.columns:
            speed_normalized = X['speed_kmh'].values / 100  # 标准化到合理范围
            features.append(speed_normalized)
        
        return np.column_stack(features)
    
    def _analyze_clusters(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        """
        分析时空聚类结果
        
        Args:
            X: 输入数据
            cluster_labels: 聚类标签
        """
        super()._analyze_clusters(X, cluster_labels)
        
        # 添加时空特定的分析
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_mask = cluster_labels == label
            cluster_data = X[cluster_mask]
            
            if label in self.cluster_stats:
                # 时空密度
                if 'timestamp' in cluster_data.columns and len(cluster_data) > 1:
                    time_span = (cluster_data['timestamp'].max() - cluster_data['timestamp'].min()).total_seconds()
                    spatial_span = self._calculate_spatial_span(cluster_data)
                    
                    if time_span > 0 and spatial_span > 0:
                        spatiotemporal_density = len(cluster_data) / (time_span * spatial_span)
                        self.cluster_stats[label]['spatiotemporal_density'] = spatiotemporal_density
                
                # 活动模式
                if 'timestamp' in cluster_data.columns:
                    hours = cluster_data['timestamp'].dt.hour
                    self.cluster_stats[label]['active_hours'] = hours.value_counts().to_dict()
                    self.cluster_stats[label]['is_daytime_cluster'] = hours.apply(
                        lambda h: is_daytime(pd.Timestamp.now().replace(hour=h))
                    ).mean() > 0.5
    
    def _calculate_spatial_span(self, cluster_data: pd.DataFrame) -> float:
        """
        计算聚类的空间跨度
        
        Args:
            cluster_data: 聚类数据
        
        Returns:
            空间跨度 (平方米)
        """
        lat_range = cluster_data['latitude'].max() - cluster_data['latitude'].min()
        lon_range = cluster_data['longitude'].max() - cluster_data['longitude'].min()
        
        # 近似计算面积 (简化)
        lat_meters = lat_range * 111000  # 1度纬度约111km
        lon_meters = lon_range * 111000 * np.cos(np.radians(cluster_data['latitude'].mean()))
        
        return lat_meters * lon_meters
    
    def detect_spatiotemporal_anomalies(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        检测时空异常
        
        Args:
            X: 输入数据
        
        Returns:
            各种时空异常分数
        """
        self._check_fitted()
        
        anomaly_scores = {}
        
        # 空间异常
        anomaly_scores['spatial'] = self._detect_spatial_anomalies(X)
        
        # 时间异常
        anomaly_scores['temporal'] = self._detect_temporal_anomalies(X)
        
        # 时空密度异常
        anomaly_scores['density'] = self._detect_density_anomalies(X)
        
        # 活动模式异常
        anomaly_scores['pattern'] = self._detect_pattern_anomalies(X)
        
        return anomaly_scores
    
    def _detect_spatial_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """检测空间异常"""
        scores = np.zeros(len(X))
        
        for i, row in X.iterrows():
            min_distance = float('inf')
            
            # 计算到所有聚类中心的最小距离
            for center_lat, center_lon in self.cluster_centers.values():
                distance = calculate_distance(
                    row['latitude'], row['longitude'],
                    center_lat, center_lon
                )
                min_distance = min(min_distance, distance)
            
            # 基于距离计算异常分数
            if min_distance != float('inf'):
                # 使用最近聚类的半径作为参考
                nearest_cluster_radius = min(
                    stats['radius'] for stats in self.cluster_stats.values()
                )
                scores[i] = min(1.0, min_distance / max(nearest_cluster_radius, 100))
        
        return scores
    
    def _detect_temporal_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """检测时间异常"""
        scores = np.zeros(len(X))
        
        if 'timestamp' not in X.columns:
            return scores
        
        for i, row in X.iterrows():
            hour = row['timestamp'].hour
            max_prob = 0
            
            # 检查在各个聚类中该时间的活动概率
            for label, stats in self.cluster_stats.items():
                if 'active_hours' in stats:
                    hour_count = stats['active_hours'].get(hour, 0)
                    total_count = sum(stats['active_hours'].values())
                    prob = hour_count / max(total_count, 1)
                    max_prob = max(max_prob, prob)
            
            # 异常分数 = 1 - 最大活动概率
            scores[i] = 1 - max_prob
        
        return scores
    
    def _detect_density_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """检测密度异常"""
        scores = np.zeros(len(X))
        
        # 计算局部密度
        for i, row in X.iterrows():
            # 计算邻域内的点数
            neighbor_count = 0
            
            for j, other_row in X.iterrows():
                if i == j:
                    continue
                
                # 空间距离
                spatial_distance = calculate_distance(
                    row['latitude'], row['longitude'],
                    other_row['latitude'], other_row['longitude']
                )
                
                # 时间距离
                if 'timestamp' in X.columns:
                    time_distance = abs((row['timestamp'] - other_row['timestamp']).total_seconds())
                else:
                    time_distance = 0
                
                # 判断是否在邻域内
                if (spatial_distance <= self.spatial_eps * 111000 and  # 转换为米
                    time_distance <= self.temporal_eps):
                    neighbor_count += 1
            
            # 基于邻居数量计算异常分数
            expected_neighbors = self.min_samples
            if neighbor_count < expected_neighbors:
                scores[i] = 1 - neighbor_count / expected_neighbors
        
        return scores
    
    def _detect_pattern_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """检测活动模式异常"""
        scores = np.zeros(len(X))
        
        if 'timestamp' not in X.columns:
            return scores
        
        for i, row in X.iterrows():
            is_daytime = is_daytime(row['timestamp'])
            max_consistency = 0
            
            # 检查与各聚类的时间模式一致性
            for label, stats in self.cluster_stats.items():
                if 'is_daytime_cluster' in stats:
                    cluster_is_daytime = stats['is_daytime_cluster']
                    consistency = 1 if is_daytime == cluster_is_daytime else 0
                    max_consistency = max(max_consistency, consistency)
            
            # 异常分数 = 1 - 最大一致性
            scores[i] = 1 - max_consistency
        
        return scores