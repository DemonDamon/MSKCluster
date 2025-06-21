"""
基于统计方法的异常检测器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_detector import RuleBasedDetector
from ..utils.geo_utils import calculate_distance, calculate_speed, calculate_acceleration
from ..utils.time_utils import is_daytime


class StatisticalAnomalyDetector(RuleBasedDetector):
    """基于统计方法的异常检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化统计检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        self.day_speed_threshold = self.config.get('day_speed_threshold', 120)  # km/h
        self.night_speed_threshold = self.config.get('night_speed_threshold', 250)  # km/h
        self.acceleration_threshold = self.config.get('acceleration_threshold', 10)  # m/s²
        self.day_distance_threshold = self.config.get('day_distance_threshold', 10000)  # 米
        self.night_distance_threshold = self.config.get('night_distance_threshold', 50000)  # 米
        
        self.day_start = self.config.get('day_start_hour', 7)
        self.day_end = self.config.get('day_end_hour', 22)
        
        # 学习到的阈值
        self.learned_thresholds = {}
    
    def _learn_thresholds(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        从数据中学习动态阈值
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        """
        # 计算基础特征
        X_features = self._calculate_features(X)
        
        # 分离白天和夜间数据
        day_mask = X_features['is_daytime']
        day_data = X_features[day_mask]
        night_data = X_features[~day_mask]
        
        # 学习速度阈值 (使用95分位数)
        if len(day_data) > 0:
            self.learned_thresholds['day_speed_95'] = day_data['speed_kmh'].quantile(0.95)
            self.learned_thresholds['day_distance_95'] = day_data['distance'].quantile(0.95)
        
        if len(night_data) > 0:
            self.learned_thresholds['night_speed_95'] = night_data['speed_kmh'].quantile(0.95)
            self.learned_thresholds['night_distance_95'] = night_data['distance'].quantile(0.95)
        
        # 学习加速度阈值
        self.learned_thresholds['acceleration_95'] = X_features['acceleration'].abs().quantile(0.95)
        
        # 如果有标签，使用有监督方法优化阈值
        if y is not None:
            self._optimize_thresholds_supervised(X_features, y)
        
        self.logger.info(f"Learned thresholds: {self.learned_thresholds}")
    
    def _optimize_thresholds_supervised(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        使用有监督方法优化阈值
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        from sklearn.metrics import f1_score
        
        # 为每个特征寻找最优阈值
        features_to_optimize = ['speed_kmh', 'distance', 'acceleration']
        
        for feature in features_to_optimize:
            if feature not in X.columns:
                continue
            
            values = X[feature].values
            best_threshold = None
            best_f1 = 0
            
            # 尝试不同的阈值
            thresholds = np.percentile(values, np.arange(80, 99, 1))
            
            for threshold in thresholds:
                pred = (values > threshold).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            if best_threshold is not None:
                self.learned_thresholds[f'{feature}_optimal'] = best_threshold
    
    def _calculate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算统计特征
        
        Args:
            X: 输入数据
        
        Returns:
            包含特征的DataFrame
        """
        X = X.copy()
        
        # 确保数据按时间排序
        if 'timestamp' in X.columns:
            X = X.sort_values('timestamp').reset_index(drop=True)
        
        # 计算基础特征 (如果不存在)
        if 'distance' not in X.columns:
            X['prev_lat'] = X['latitude'].shift(1)
            X['prev_lon'] = X['longitude'].shift(1)
            X['distance'] = X.apply(
                lambda row: calculate_distance(
                    row['prev_lat'], row['prev_lon'],
                    row['latitude'], row['longitude']
                ) if not pd.isna(row['prev_lat']) else 0,
                axis=1
            )
        
        if 'time_diff' not in X.columns and 'timestamp' in X.columns:
            X['time_diff'] = X['timestamp'].diff().dt.total_seconds().fillna(0)
        
        if 'speed' not in X.columns:
            X['speed'] = X.apply(
                lambda row: calculate_speed(row['distance'], row['time_diff'])
                if 'time_diff' in row and row['time_diff'] > 0 else 0,
                axis=1
            )
        
        if 'speed_kmh' not in X.columns:
            X['speed_kmh'] = X['speed'] * 3.6
        
        if 'acceleration' not in X.columns:
            X['prev_speed'] = X['speed'].shift(1)
            X['acceleration'] = X.apply(
                lambda row: calculate_acceleration(
                    row['prev_speed'], row['speed'], row['time_diff']
                ) if not pd.isna(row['prev_speed']) and 'time_diff' in row and row['time_diff'] > 0 else 0,
                axis=1
            )
        
        if 'is_daytime' not in X.columns and 'timestamp' in X.columns:
            X['is_daytime'] = X['timestamp'].apply(
                lambda x: is_daytime(x, self.day_start, self.day_end)
            )
        
        return X
    
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        应用统计规则
        
        Args:
            X: 特征数据
        
        Returns:
            异常分数
        """
        X_features = self._calculate_features(X)
        
        # 初始化异常分数
        anomaly_scores = np.zeros(len(X_features))
        
        # 规则1: 速度异常
        speed_scores = self._check_speed_anomaly(X_features)
        anomaly_scores += speed_scores
        
        # 规则2: 距离异常
        distance_scores = self._check_distance_anomaly(X_features)
        anomaly_scores += distance_scores
        
        # 规则3: 加速度异常
        acceleration_scores = self._check_acceleration_anomaly(X_features)
        anomaly_scores += acceleration_scores
        
        # 规则4: 方向变化异常
        if 'direction_change' in X_features.columns:
            direction_scores = self._check_direction_anomaly(X_features)
            anomaly_scores += direction_scores
        
        # 规则5: 时间一致性异常
        time_scores = self._check_time_consistency(X_features)
        anomaly_scores += time_scores
        
        # 归一化分数到[0,1]
        max_score = 5.0  # 最大可能分数
        anomaly_scores = np.clip(anomaly_scores / max_score, 0, 1)
        
        return anomaly_scores
    
    def _check_speed_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查速度异常"""
        scores = np.zeros(len(X))
        
        if 'speed_kmh' not in X.columns or 'is_daytime' not in X.columns:
            return scores
        
        day_mask = X['is_daytime']
        
        # 白天速度检查
        day_threshold = min(
            self.day_speed_threshold,
            self.learned_thresholds.get('day_speed_95', self.day_speed_threshold)
        )
        day_anomaly = (X['speed_kmh'] > day_threshold) & day_mask
        scores[day_anomaly] += 1.0
        
        # 夜间速度检查
        night_threshold = min(
            self.night_speed_threshold,
            self.learned_thresholds.get('night_speed_95', self.night_speed_threshold)
        )
        night_anomaly = (X['speed_kmh'] > night_threshold) & (~day_mask)
        scores[night_anomaly] += 1.0
        
        # 使用优化阈值 (如果存在)
        if 'speed_kmh_optimal' in self.learned_thresholds:
            optimal_anomaly = X['speed_kmh'] > self.learned_thresholds['speed_kmh_optimal']
            scores[optimal_anomaly] += 0.5
        
        return scores
    
    def _check_distance_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查距离异常"""
        scores = np.zeros(len(X))
        
        if 'distance' not in X.columns or 'is_daytime' not in X.columns:
            return scores
        
        day_mask = X['is_daytime']
        
        # 白天距离检查
        day_threshold = min(
            self.day_distance_threshold,
            self.learned_thresholds.get('day_distance_95', self.day_distance_threshold)
        )
        day_anomaly = (X['distance'] > day_threshold) & day_mask
        scores[day_anomaly] += 1.0
        
        # 夜间距离检查
        night_threshold = min(
            self.night_distance_threshold,
            self.learned_thresholds.get('night_distance_95', self.night_distance_threshold)
        )
        night_anomaly = (X['distance'] > night_threshold) & (~day_mask)
        scores[night_anomaly] += 1.0
        
        return scores
    
    def _check_acceleration_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查加速度异常"""
        scores = np.zeros(len(X))
        
        if 'acceleration' not in X.columns:
            return scores
        
        threshold = min(
            self.acceleration_threshold,
            self.learned_thresholds.get('acceleration_95', self.acceleration_threshold)
        )
        
        anomaly = np.abs(X['acceleration']) > threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def _check_direction_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查方向变化异常"""
        scores = np.zeros(len(X))
        
        if 'direction_change' not in X.columns:
            return scores
        
        # 方向突变阈值 (度)
        direction_threshold = self.config.get('direction_change_threshold', 120)
        
        anomaly = X['direction_change'] > direction_threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def _check_time_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """检查时间一致性异常"""
        scores = np.zeros(len(X))
        
        if 'timestamp' not in X.columns:
            return scores
        
        # 检查时间间隔异常
        if 'time_diff' in X.columns:
            # 异常的时间间隔 (太短或太长)
            min_interval = 60  # 1分钟
            max_interval = 3600  # 1小时
            
            time_anomaly = (X['time_diff'] < min_interval) | (X['time_diff'] > max_interval)
            scores[time_anomaly] += 0.5
        
        # 检查夜间活动异常
        if 'is_daytime' in X.columns and 'speed_kmh' in X.columns:
            # 夜间高速移动
            night_activity = (~X['is_daytime']) & (X['speed_kmh'] > 50)
            scores[night_activity] += 0.5
        
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        return {
            'speed': 0.3,
            'distance': 0.25,
            'acceleration': 0.2,
            'direction_change': 0.15,
            'time_consistency': 0.1
        }
    
    def _get_model_data(self) -> Dict:
        """获取模型特定数据"""
        return {
            'learned_thresholds': self.learned_thresholds,
            'day_speed_threshold': self.day_speed_threshold,
            'night_speed_threshold': self.night_speed_threshold,
            'acceleration_threshold': self.acceleration_threshold,
            'day_distance_threshold': self.day_distance_threshold,
            'night_distance_threshold': self.night_distance_threshold
        }
    
    def _set_model_data(self, model_data: Dict) -> None:
        """设置模型特定数据"""
        self.learned_thresholds = model_data.get('learned_thresholds', {})
        self.day_speed_threshold = model_data.get('day_speed_threshold', 120)
        self.night_speed_threshold = model_data.get('night_speed_threshold', 250)
        self.acceleration_threshold = model_data.get('acceleration_threshold', 10)
        self.day_distance_threshold = model_data.get('day_distance_threshold', 10000)
        self.night_distance_threshold = model_data.get('night_distance_threshold', 50000)


class AdaptiveStatisticalDetector(StatisticalAnomalyDetector):
    """自适应统计检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化自适应统计检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 自适应参数
        self.adaptation_window = self.config.get('adaptation_window', 100)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        
        # 历史统计
        self.history_stats = {
            'speed_mean': [],
            'speed_std': [],
            'distance_mean': [],
            'distance_std': []
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常概率 (带自适应)
        
        Args:
            X: 特征数据
        
        Returns:
            异常概率
        """
        # 更新历史统计
        self._update_history_stats(X)
        
        # 自适应调整阈值
        self._adapt_thresholds()
        
        # 调用父类方法
        return super().predict_proba(X)
    
    def _update_history_stats(self, X: pd.DataFrame) -> None:
        """更新历史统计"""
        X_features = self._calculate_features(X)
        
        if 'speed_kmh' in X_features.columns:
            self.history_stats['speed_mean'].append(X_features['speed_kmh'].mean())
            self.history_stats['speed_std'].append(X_features['speed_kmh'].std())
        
        if 'distance' in X_features.columns:
            self.history_stats['distance_mean'].append(X_features['distance'].mean())
            self.history_stats['distance_std'].append(X_features['distance'].std())
        
        # 保持窗口大小
        for key in self.history_stats:
            if len(self.history_stats[key]) > self.adaptation_window:
                self.history_stats[key] = self.history_stats[key][-self.adaptation_window:]
    
    def _adapt_thresholds(self) -> None:
        """自适应调整阈值"""
        if len(self.history_stats['speed_mean']) < 10:
            return
        
        # 基于历史统计调整速度阈值
        recent_speed_mean = np.mean(self.history_stats['speed_mean'][-10:])
        recent_speed_std = np.mean(self.history_stats['speed_std'][-10:])
        
        # 动态调整阈值 (均值 + 3倍标准差)
        adaptive_speed_threshold = recent_speed_mean + 3 * recent_speed_std
        
        # 平滑更新
        current_threshold = self.learned_thresholds.get('day_speed_95', self.day_speed_threshold)
        new_threshold = (1 - self.adaptation_rate) * current_threshold + \
                       self.adaptation_rate * adaptive_speed_threshold
        
        self.learned_thresholds['day_speed_95'] = new_threshold
        
        # 类似地调整距离阈值
        if len(self.history_stats['distance_mean']) >= 10:
            recent_distance_mean = np.mean(self.history_stats['distance_mean'][-10:])
            recent_distance_std = np.mean(self.history_stats['distance_std'][-10:])
            
            adaptive_distance_threshold = recent_distance_mean + 3 * recent_distance_std
            
            current_threshold = self.learned_thresholds.get('day_distance_95', self.day_distance_threshold)
            new_threshold = (1 - self.adaptation_rate) * current_threshold + \
                           self.adaptation_rate * adaptive_distance_threshold
            
            self.learned_thresholds['day_distance_95'] = new_threshold