"""
基于轨迹分析的异常检测器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.interpolate import interp1d
from .base_detector import RuleBasedDetector
from ..utils.geo_utils import calculate_distance, calculate_bearing


class TrajectoryAnomalyDetector(RuleBasedDetector):
    """基于轨迹分析的异常检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化轨迹检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        self.window_size = self.config.get('window_size', 5)
        self.min_window_size = self.config.get('min_window_size', 3)
        self.residual_threshold = self.config.get('residual_threshold', 500)  # 米
        self.smoothing_method = self.config.get('smoothing_method', 'median')
        self.direction_change_threshold = self.config.get('direction_change_threshold', 120)  # 度
        
        # 学习到的参数
        self.learned_params = {}
    
    def _learn_thresholds(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        从数据中学习轨迹参数
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        """
        # 计算轨迹特征
        X_features = self._calculate_trajectory_features(X)
        
        # 学习残差阈值
        if 'residual' in X_features.columns:
            self.learned_params['residual_95'] = X_features['residual'].quantile(0.95)
            self.learned_params['residual_99'] = X_features['residual'].quantile(0.99)
        
        # 学习方向变化阈值
        if 'direction_change' in X_features.columns:
            self.learned_params['direction_95'] = X_features['direction_change'].quantile(0.95)
        
        # 学习轨迹平滑度参数
        if 'trajectory_smoothness' in X_features.columns:
            self.learned_params['smoothness_95'] = X_features['trajectory_smoothness'].quantile(0.95)
        
        self.logger.info(f"Learned trajectory parameters: {self.learned_params}")
    
    def _calculate_trajectory_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算轨迹特征
        
        Args:
            X: 输入数据
        
        Returns:
            包含轨迹特征的DataFrame
        """
        X = X.copy()
        
        # 确保数据按时间排序
        if 'timestamp' in X.columns:
            X = X.sort_values('timestamp').reset_index(drop=True)
        
        # 计算轨迹残差
        X['residual'] = self._calculate_residuals(X)
        
        # 计算方向变化
        X['direction_change'] = self._calculate_direction_changes(X)
        
        # 计算轨迹平滑度
        X['trajectory_smoothness'] = self._calculate_trajectory_smoothness(X)
        
        # 计算轨迹一致性
        X['trajectory_consistency'] = self._calculate_trajectory_consistency(X)
        
        # 计算预测误差
        X['prediction_error'] = self._calculate_prediction_error(X)
        
        return X
    
    def _calculate_residuals(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算轨迹残差 (实际位置与平滑轨迹的偏差)
        
        Args:
            X: 数据DataFrame
        
        Returns:
            残差数组
        """
        if len(X) < self.min_window_size:
            return np.zeros(len(X))
        
        residuals = np.zeros(len(X))
        
        # 使用滑动窗口计算残差
        for i in range(len(X)):
            # 确定窗口范围
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(X), i + self.window_size // 2 + 1)
            
            if end_idx - start_idx < self.min_window_size:
                continue
            
            # 提取窗口数据
            window_data = X.iloc[start_idx:end_idx]
            
            # 计算平滑位置
            if self.smoothing_method == 'median':
                smooth_lat = window_data['latitude'].median()
                smooth_lon = window_data['longitude'].median()
            elif self.smoothing_method == 'mean':
                smooth_lat = window_data['latitude'].mean()
                smooth_lon = window_data['longitude'].mean()
            elif self.smoothing_method == 'kalman':
                smooth_lat, smooth_lon = self._kalman_smooth(window_data, i - start_idx)
            else:
                smooth_lat = window_data['latitude'].median()
                smooth_lon = window_data['longitude'].median()
            
            # 计算残差
            actual_lat = X.iloc[i]['latitude']
            actual_lon = X.iloc[i]['longitude']
            
            residuals[i] = calculate_distance(actual_lat, actual_lon, smooth_lat, smooth_lon)
        
        return residuals
    
    def _kalman_smooth(self, window_data: pd.DataFrame, target_idx: int) -> Tuple[float, float]:
        """
        简化的卡尔曼滤波平滑
        
        Args:
            window_data: 窗口数据
            target_idx: 目标索引
        
        Returns:
            平滑后的坐标
        """
        # 简化实现：使用线性插值作为卡尔曼滤波的近似
        if len(window_data) < 2:
            return window_data.iloc[target_idx]['latitude'], window_data.iloc[target_idx]['longitude']
        
        # 时间序列
        if 'timestamp' in window_data.columns:
            times = pd.to_numeric(window_data['timestamp'])
            target_time = times.iloc[target_idx]
            
            # 线性插值
            f_lat = interp1d(times, window_data['latitude'], kind='linear', fill_value='extrapolate')
            f_lon = interp1d(times, window_data['longitude'], kind='linear', fill_value='extrapolate')
            
            smooth_lat = float(f_lat(target_time))
            smooth_lon = float(f_lon(target_time))
        else:
            # 如果没有时间戳，使用位置插值
            indices = np.arange(len(window_data))
            f_lat = interp1d(indices, window_data['latitude'], kind='linear', fill_value='extrapolate')
            f_lon = interp1d(indices, window_data['longitude'], kind='linear', fill_value='extrapolate')
            
            smooth_lat = float(f_lat(target_idx))
            smooth_lon = float(f_lon(target_idx))
        
        return smooth_lat, smooth_lon
    
    def _calculate_direction_changes(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算方向变化
        
        Args:
            X: 数据DataFrame
        
        Returns:
            方向变化数组
        """
        if len(X) < 3:
            return np.zeros(len(X))
        
        direction_changes = np.zeros(len(X))
        
        for i in range(1, len(X) - 1):
            # 前一段的方向
            bearing1 = calculate_bearing(
                X.iloc[i-1]['latitude'], X.iloc[i-1]['longitude'],
                X.iloc[i]['latitude'], X.iloc[i]['longitude']
            )
            
            # 后一段的方向
            bearing2 = calculate_bearing(
                X.iloc[i]['latitude'], X.iloc[i]['longitude'],
                X.iloc[i+1]['latitude'], X.iloc[i+1]['longitude']
            )
            
            # 计算方向变化
            diff = abs(bearing2 - bearing1)
            direction_changes[i] = min(diff, 360 - diff)
        
        return direction_changes
    
    def _calculate_trajectory_smoothness(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算轨迹平滑度 (基于二阶导数)
        
        Args:
            X: 数据DataFrame
        
        Returns:
            平滑度数组
        """
        if len(X) < 3:
            return np.zeros(len(X))
        
        smoothness = np.zeros(len(X))
        
        # 计算位置的二阶导数
        lat_diff2 = np.diff(X['latitude'], n=2)
        lon_diff2 = np.diff(X['longitude'], n=2)
        
        # 计算平滑度 (二阶导数的模长)
        for i in range(1, len(X) - 1):
            smoothness[i] = np.sqrt(lat_diff2[i-1]**2 + lon_diff2[i-1]**2)
        
        return smoothness
    
    def _calculate_trajectory_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算轨迹一致性 (与历史模式的一致性)
        
        Args:
            X: 数据DataFrame
        
        Returns:
            一致性分数数组
        """
        if len(X) < self.window_size:
            return np.ones(len(X))  # 默认一致
        
        consistency = np.ones(len(X))
        
        # 计算每个点与其邻域的一致性
        for i in range(self.window_size, len(X)):
            # 当前窗口
            current_window = X.iloc[i-self.window_size:i]
            
            # 计算窗口内的平均移动模式
            if 'speed_kmh' in X.columns:
                avg_speed = current_window['speed_kmh'].mean()
                current_speed = X.iloc[i]['speed_kmh'] if 'speed_kmh' in X.columns else 0
                
                # 速度一致性
                speed_consistency = 1.0 - min(1.0, abs(current_speed - avg_speed) / max(avg_speed, 1))
                consistency[i] *= speed_consistency
            
            # 方向一致性
            if 'bearing' in X.columns:
                # 计算方向的一致性 (使用圆形统计)
                bearings = current_window['bearing'].values
                if len(bearings) > 0:
                    # 转换为复数表示
                    complex_bearings = np.exp(1j * np.radians(bearings))
                    mean_bearing = np.angle(np.mean(complex_bearings), deg=True)
                    
                    current_bearing = X.iloc[i]['bearing'] if 'bearing' in X.columns else 0
                    bearing_diff = abs(current_bearing - mean_bearing)
                    bearing_diff = min(bearing_diff, 360 - bearing_diff)
                    
                    bearing_consistency = 1.0 - min(1.0, bearing_diff / 180)
                    consistency[i] *= bearing_consistency
        
        return consistency
    
    def _calculate_prediction_error(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算预测误差 (基于线性预测)
        
        Args:
            X: 数据DataFrame
        
        Returns:
            预测误差数组
        """
        if len(X) < 3:
            return np.zeros(len(X))
        
        prediction_errors = np.zeros(len(X))
        
        for i in range(2, len(X)):
            # 使用前两个点预测当前点
            prev2_lat, prev2_lon = X.iloc[i-2]['latitude'], X.iloc[i-2]['longitude']
            prev1_lat, prev1_lon = X.iloc[i-1]['latitude'], X.iloc[i-1]['longitude']
            actual_lat, actual_lon = X.iloc[i]['latitude'], X.iloc[i]['longitude']
            
            # 线性预测
            pred_lat = 2 * prev1_lat - prev2_lat
            pred_lon = 2 * prev1_lon - prev2_lon
            
            # 计算预测误差
            prediction_errors[i] = calculate_distance(actual_lat, actual_lon, pred_lat, pred_lon)
        
        return prediction_errors
    
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        应用轨迹规则
        
        Args:
            X: 特征数据
        
        Returns:
            异常分数
        """
        X_features = self._calculate_trajectory_features(X)
        
        # 初始化异常分数
        anomaly_scores = np.zeros(len(X_features))
        
        # 规则1: 残差异常
        residual_scores = self._check_residual_anomaly(X_features)
        anomaly_scores += residual_scores
        
        # 规则2: 方向变化异常
        direction_scores = self._check_direction_anomaly(X_features)
        anomaly_scores += direction_scores
        
        # 规则3: 平滑度异常
        smoothness_scores = self._check_smoothness_anomaly(X_features)
        anomaly_scores += smoothness_scores
        
        # 规则4: 一致性异常
        consistency_scores = self._check_consistency_anomaly(X_features)
        anomaly_scores += consistency_scores
        
        # 规则5: 预测误差异常
        prediction_scores = self._check_prediction_anomaly(X_features)
        anomaly_scores += prediction_scores
        
        # 归一化分数到[0,1]
        max_score = 5.0
        anomaly_scores = np.clip(anomaly_scores / max_score, 0, 1)
        
        return anomaly_scores
    
    def _check_residual_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查残差异常"""
        scores = np.zeros(len(X))
        
        if 'residual' not in X.columns:
            return scores
        
        # 使用学习到的阈值或默认阈值
        threshold = min(
            self.residual_threshold,
            self.learned_params.get('residual_95', self.residual_threshold)
        )
        
        anomaly = X['residual'] > threshold
        scores[anomaly] += 1.0
        
        # 极端异常
        extreme_threshold = self.learned_params.get('residual_99', threshold * 2)
        extreme_anomaly = X['residual'] > extreme_threshold
        scores[extreme_anomaly] += 0.5
        
        return scores
    
    def _check_direction_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查方向变化异常"""
        scores = np.zeros(len(X))
        
        if 'direction_change' not in X.columns:
            return scores
        
        threshold = min(
            self.direction_change_threshold,
            self.learned_params.get('direction_95', self.direction_change_threshold)
        )
        
        anomaly = X['direction_change'] > threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def _check_smoothness_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查平滑度异常"""
        scores = np.zeros(len(X))
        
        if 'trajectory_smoothness' not in X.columns:
            return scores
        
        threshold = self.learned_params.get('smoothness_95', X['trajectory_smoothness'].quantile(0.95))
        
        anomaly = X['trajectory_smoothness'] > threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def _check_consistency_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查一致性异常"""
        scores = np.zeros(len(X))
        
        if 'trajectory_consistency' not in X.columns:
            return scores
        
        # 一致性低的点是异常的
        threshold = 0.5  # 一致性阈值
        
        anomaly = X['trajectory_consistency'] < threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def _check_prediction_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """检查预测误差异常"""
        scores = np.zeros(len(X))
        
        if 'prediction_error' not in X.columns:
            return scores
        
        threshold = X['prediction_error'].quantile(0.95)
        
        anomaly = X['prediction_error'] > threshold
        scores[anomaly] += 1.0
        
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        return {
            'residual': 0.3,
            'direction_change': 0.25,
            'trajectory_smoothness': 0.2,
            'trajectory_consistency': 0.15,
            'prediction_error': 0.1
        }
    
    def _get_model_data(self) -> Dict:
        """获取模型特定数据"""
        return {
            'learned_params': self.learned_params,
            'window_size': self.window_size,
            'min_window_size': self.min_window_size,
            'residual_threshold': self.residual_threshold,
            'smoothing_method': self.smoothing_method,
            'direction_change_threshold': self.direction_change_threshold
        }
    
    def _set_model_data(self, model_data: Dict) -> None:
        """设置模型特定数据"""
        self.learned_params = model_data.get('learned_params', {})
        self.window_size = model_data.get('window_size', 5)
        self.min_window_size = model_data.get('min_window_size', 3)
        self.residual_threshold = model_data.get('residual_threshold', 500)
        self.smoothing_method = model_data.get('smoothing_method', 'median')
        self.direction_change_threshold = model_data.get('direction_change_threshold', 120)


class AdvancedTrajectoryDetector(TrajectoryAnomalyDetector):
    """高级轨迹检测器 (包含更复杂的分析方法)"""
    
    def __init__(self, config: Dict = None):
        """
        初始化高级轨迹检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 高级参数
        self.use_fourier_analysis = self.config.get('use_fourier_analysis', True)
        self.use_wavelet_analysis = self.config.get('use_wavelet_analysis', False)
        self.pattern_memory_size = self.config.get('pattern_memory_size', 1000)
        
        # 模式记忆
        self.normal_patterns = []
    
    def _calculate_trajectory_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级轨迹特征
        
        Args:
            X: 输入数据
        
        Returns:
            包含高级特征的DataFrame
        """
        # 调用父类方法
        X = super()._calculate_trajectory_features(X)
        
        # 添加高级特征
        if self.use_fourier_analysis:
            X['fourier_anomaly'] = self._fourier_analysis(X)
        
        if self.use_wavelet_analysis:
            X['wavelet_anomaly'] = self._wavelet_analysis(X)
        
        # 模式匹配
        X['pattern_similarity'] = self._pattern_matching(X)
        
        return X
    
    def _fourier_analysis(self, X: pd.DataFrame) -> np.ndarray:
        """
        傅里叶分析检测周期性异常
        
        Args:
            X: 数据DataFrame
        
        Returns:
            傅里叶异常分数
        """
        if len(X) < 16:  # 需要足够的数据点
            return np.zeros(len(X))
        
        # 对纬度和经度分别进行FFT
        lat_fft = np.fft.fft(X['latitude'].values)
        lon_fft = np.fft.fft(X['longitude'].values)
        
        # 计算功率谱
        lat_power = np.abs(lat_fft) ** 2
        lon_power = np.abs(lon_fft) ** 2
        
        # 检测异常频率成分
        lat_anomaly = self._detect_frequency_anomaly(lat_power)
        lon_anomaly = self._detect_frequency_anomaly(lon_power)
        
        # 合并异常分数
        fourier_scores = np.maximum(lat_anomaly, lon_anomaly)
        
        return fourier_scores
    
    def _detect_frequency_anomaly(self, power_spectrum: np.ndarray) -> np.ndarray:
        """
        检测频率域异常
        
        Args:
            power_spectrum: 功率谱
        
        Returns:
            异常分数
        """
        # 简化实现：检测异常的高频成分
        n = len(power_spectrum)
        scores = np.zeros(n)
        
        # 计算高频能量比例
        low_freq_energy = np.sum(power_spectrum[:n//4])
        high_freq_energy = np.sum(power_spectrum[n//4:])
        
        if low_freq_energy > 0:
            high_freq_ratio = high_freq_energy / low_freq_energy
            
            # 如果高频能量比例过高，可能存在异常
            if high_freq_ratio > 0.5:
                scores[:] = min(1.0, high_freq_ratio)
        
        return scores
    
    def _wavelet_analysis(self, X: pd.DataFrame) -> np.ndarray:
        """
        小波分析检测局部异常
        
        Args:
            X: 数据DataFrame
        
        Returns:
            小波异常分数
        """
        # 简化实现：使用差分近似小波变换
        if len(X) < 8:
            return np.zeros(len(X))
        
        lat_signal = X['latitude'].values
        lon_signal = X['longitude'].values
        
        # 多尺度差分
        scales = [1, 2, 4]
        anomaly_scores = np.zeros(len(X))
        
        for scale in scales:
            if len(X) > scale * 2:
                # 计算差分
                lat_diff = np.diff(lat_signal, n=scale)
                lon_diff = np.diff(lon_signal, n=scale)
                
                # 计算局部异常
                diff_magnitude = np.sqrt(lat_diff**2 + lon_diff**2)
                
                # 检测异常值
                threshold = np.percentile(diff_magnitude, 95)
                scale_anomaly = diff_magnitude > threshold
                
                # 映射回原始长度
                for i, is_anomaly in enumerate(scale_anomaly):
                    if is_anomaly and i + scale < len(anomaly_scores):
                        anomaly_scores[i + scale] += 1.0 / len(scales)
        
        return anomaly_scores
    
    def _pattern_matching(self, X: pd.DataFrame) -> np.ndarray:
        """
        模式匹配检测
        
        Args:
            X: 数据DataFrame
        
        Returns:
            模式相似度分数
        """
        if len(X) < self.window_size or len(self.normal_patterns) == 0:
            return np.ones(len(X))  # 默认相似
        
        similarity_scores = np.ones(len(X))
        
        # 提取当前轨迹的模式
        for i in range(self.window_size, len(X)):
            current_pattern = self._extract_pattern(X.iloc[i-self.window_size:i])
            
            # 与历史正常模式比较
            max_similarity = 0
            for normal_pattern in self.normal_patterns:
                similarity = self._calculate_pattern_similarity(current_pattern, normal_pattern)
                max_similarity = max(max_similarity, similarity)
            
            similarity_scores[i] = max_similarity
        
        return similarity_scores
    
    def _extract_pattern(self, window_data: pd.DataFrame) -> Dict:
        """
        提取轨迹模式特征
        
        Args:
            window_data: 窗口数据
        
        Returns:
            模式特征字典
        """
        pattern = {}
        
        # 统计特征
        pattern['mean_speed'] = window_data['speed_kmh'].mean() if 'speed_kmh' in window_data.columns else 0
        pattern['std_speed'] = window_data['speed_kmh'].std() if 'speed_kmh' in window_data.columns else 0
        pattern['mean_distance'] = window_data['distance'].mean() if 'distance' in window_data.columns else 0
        
        # 几何特征
        pattern['total_distance'] = window_data['distance'].sum() if 'distance' in window_data.columns else 0
        pattern['displacement'] = calculate_distance(
            window_data.iloc[0]['latitude'], window_data.iloc[0]['longitude'],
            window_data.iloc[-1]['latitude'], window_data.iloc[-1]['longitude']
        )
        
        # 方向特征
        if 'bearing' in window_data.columns:
            bearings = window_data['bearing'].values
            # 计算主要方向
            complex_bearings = np.exp(1j * np.radians(bearings))
            mean_direction = np.angle(np.mean(complex_bearings), deg=True)
            pattern['main_direction'] = mean_direction
            pattern['direction_variance'] = np.var(bearings)
        
        return pattern
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        计算模式相似度
        
        Args:
            pattern1: 模式1
            pattern2: 模式2
        
        Returns:
            相似度分数 (0-1)
        """
        similarity = 0
        count = 0
        
        # 比较各个特征
        for key in pattern1:
            if key in pattern2:
                val1, val2 = pattern1[key], pattern2[key]
                
                if key in ['mean_speed', 'std_speed', 'mean_distance', 'total_distance', 'displacement']:
                    # 数值特征：使用相对差异
                    if max(val1, val2) > 0:
                        diff = abs(val1 - val2) / max(val1, val2)
                        similarity += 1 - min(1, diff)
                        count += 1
                elif key in ['main_direction']:
                    # 角度特征：使用圆形距离
                    diff = abs(val1 - val2)
                    diff = min(diff, 360 - diff)
                    similarity += 1 - min(1, diff / 180)
                    count += 1
                elif key in ['direction_variance']:
                    # 方差特征
                    if max(val1, val2) > 0:
                        diff = abs(val1 - val2) / max(val1, val2)
                        similarity += 1 - min(1, diff)
                        count += 1
        
        return similarity / count if count > 0 else 0
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'AdvancedTrajectoryDetector':
        """
        训练高级轨迹检测器
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        
        Returns:
            训练后的检测器
        """
        # 调用父类方法
        super().fit(X, y)
        
        # 学习正常模式
        self._learn_normal_patterns(X, y)
        
        return self
    
    def _learn_normal_patterns(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        学习正常轨迹模式
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        """
        # 如果有标签，只使用正常样本
        if y is not None:
            normal_data = X[y == 0]
        else:
            normal_data = X
        
        # 提取正常模式
        self.normal_patterns = []
        
        for i in range(self.window_size, len(normal_data), self.window_size // 2):
            if i + self.window_size <= len(normal_data):
                window_data = normal_data.iloc[i-self.window_size:i]
                pattern = self._extract_pattern(window_data)
                self.normal_patterns.append(pattern)
                
                # 限制模式数量
                if len(self.normal_patterns) >= self.pattern_memory_size:
                    break
        
        self.logger.info(f"Learned {len(self.normal_patterns)} normal patterns")