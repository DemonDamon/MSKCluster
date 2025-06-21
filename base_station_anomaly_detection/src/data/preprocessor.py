"""
数据预处理模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..utils.geo_utils import calculate_trajectory_features
from ..utils.time_utils import calculate_time_features, is_daytime


class TrajectoryPreprocessor:
    """轨迹数据预处理器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.scaler = None
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.day_start = self.config.get('day_start_hour', 7)
        self.day_end = self.config.get('day_end_hour', 22)
        self.coordinate_precision = self.config.get('coordinate_precision', 6)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            清洗后的DataFrame
        """
        df = df.copy()
        
        # 检查必要列
        required_cols = ['timestamp', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 清理坐标数据
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # 坐标精度处理
        df['latitude'] = df['latitude'].round(self.coordinate_precision)
        df['longitude'] = df['longitude'].round(self.coordinate_precision)
        
        # 删除无效坐标
        valid_coords = (
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180)) &
            (df['latitude'].notna()) &
            (df['longitude'].notna())
        )
        
        invalid_count = (~valid_coords).sum()
        if invalid_count > 0:
            self.logger.warning(f"Removed {invalid_count} invalid coordinate points")
            df = df[valid_coords]
        
        # 删除重复记录
        duplicate_count = df.duplicated(['timestamp', 'latitude', 'longitude']).sum()
        if duplicate_count > 0:
            self.logger.warning(f"Removed {duplicate_count} duplicate records")
            df = df.drop_duplicates(['timestamp', 'latitude', 'longitude'])
        
        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征提取
        
        Args:
            df: 清洗后的数据DataFrame
        
        Returns:
            包含特征的DataFrame
        """
        df = df.copy()
        
        # 计算轨迹特征
        df = calculate_trajectory_features(df)
        
        # 计算时间特征
        df = calculate_time_features(df)
        
        # 计算滑动窗口统计特征
        df = self._calculate_window_features(df)
        
        # 计算位置聚集特征
        df = self._calculate_location_features(df)
        
        return df
    
    def _calculate_window_features(self, df: pd.DataFrame, window_sizes: List[int] = None) -> pd.DataFrame:
        """
        计算滑动窗口统计特征
        
        Args:
            df: 数据DataFrame
            window_sizes: 窗口大小列表
        
        Returns:
            添加了窗口特征的DataFrame
        """
        if window_sizes is None:
            window_sizes = [3, 5, 10]
        
        df = df.copy()
        
        for window in window_sizes:
            if len(df) < window:
                continue
                
            # 位置统计
            df[f'lat_mean_{window}'] = df['latitude'].rolling(window, center=True).mean()
            df[f'lon_mean_{window}'] = df['longitude'].rolling(window, center=True).mean()
            df[f'lat_std_{window}'] = df['latitude'].rolling(window, center=True).std()
            df[f'lon_std_{window}'] = df['longitude'].rolling(window, center=True).std()
            
            # 速度统计
            df[f'speed_mean_{window}'] = df['speed_kmh'].rolling(window, center=True).mean()
            df[f'speed_max_{window}'] = df['speed_kmh'].rolling(window, center=True).max()
            df[f'speed_std_{window}'] = df['speed_kmh'].rolling(window, center=True).std()
            
            # 距离统计
            df[f'distance_sum_{window}'] = df['distance'].rolling(window, center=True).sum()
            df[f'distance_mean_{window}'] = df['distance'].rolling(window, center=True).mean()
        
        return df
    
    def _calculate_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算位置相关特征
        
        Args:
            df: 数据DataFrame
        
        Returns:
            添加了位置特征的DataFrame
        """
        df = df.copy()
        
        # 计算到质心的距离
        centroid_lat = df['latitude'].mean()
        centroid_lon = df['longitude'].mean()
        
        from ..utils.geo_utils import calculate_distance
        df['distance_to_centroid'] = df.apply(
            lambda row: calculate_distance(
                row['latitude'], row['longitude'],
                centroid_lat, centroid_lon
            ), axis=1
        )
        
        # 计算位置密度 (简化版，避免O(n²)计算)
        # 使用网格方法近似计算密度
        df['local_density'] = self._calculate_grid_density(df)
        
        return df
    
    def _calculate_grid_density(self, df: pd.DataFrame, grid_size: float = 0.01) -> np.ndarray:
        """
        使用网格方法计算局部密度
        
        Args:
            df: 数据DataFrame
            grid_size: 网格大小 (度)
        
        Returns:
            密度数组
        """
        # 创建网格
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # 计算网格索引
        lat_grid = ((df['latitude'] - lat_min) / grid_size).astype(int)
        lon_grid = ((df['longitude'] - lon_min) / grid_size).astype(int)
        
        # 计算每个网格的点数
        grid_counts = {}
        for i, (lat_idx, lon_idx) in enumerate(zip(lat_grid, lon_grid)):
            key = (lat_idx, lon_idx)
            if key not in grid_counts:
                grid_counts[key] = 0
            grid_counts[key] += 1
        
        # 为每个点分配密度值
        densities = np.zeros(len(df))
        for i, (lat_idx, lon_idx) in enumerate(zip(lat_grid, lon_grid)):
            key = (lat_idx, lon_idx)
            densities[i] = grid_counts.get(key, 0)
        
        return densities
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据DataFrame
            strategy: 处理策略 ('interpolate', 'forward_fill', 'drop')
        
        Returns:
            处理后的DataFrame
        """
        df = df.copy()
        
        if strategy == 'interpolate':
            # 对数值列进行插值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
            
        elif strategy == 'drop':
            df = df.dropna()
            
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            df: 数据DataFrame
            method: 标准化方法 ('standard', 'minmax')
        
        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        
        # 选择需要标准化的数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['timestamp', 'latitude', 'longitude', 'label']  # 排除不需要标准化的列
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            return df
        
        # 初始化标准化器
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # 拟合和转换
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        self.feature_columns = feature_cols
        
        return df
    
    def apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用已训练的标准化器
        
        Args:
            df: 数据DataFrame
        
        Returns:
            标准化后的DataFrame
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call normalize_features first.")
        
        df = df.copy()
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df
    
    def split_by_time_period(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间段分割数据 (白天/夜间)
        
        Args:
            df: 数据DataFrame
        
        Returns:
            (白天数据, 夜间数据)
        """
        df = df.copy()
        
        # 添加时间段标记
        df['is_daytime'] = df['timestamp'].apply(
            lambda x: is_daytime(x, self.day_start, self.day_end)
        )
        
        day_data = df[df['is_daytime']].copy()
        night_data = df[~df['is_daytime']].copy()
        
        return day_data, night_data
    
    def create_sequences(self, df: pd.DataFrame, 
                        sequence_length: int = 10,
                        target_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据 (用于RNN/LSTM)
        
        Args:
            df: 数据DataFrame
            sequence_length: 序列长度
            target_col: 目标列名
        
        Returns:
            (特征序列, 标签序列)
        """
        df = df.copy()
        
        # 选择特征列
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'latitude', 'longitude', target_col]]
        
        X, y = [], []
        
        for i in range(len(df) - sequence_length + 1):
            # 提取序列特征
            seq_features = df.iloc[i:i+sequence_length][feature_cols].values
            seq_target = df.iloc[i+sequence_length-1][target_col] if target_col in df.columns else 0
            
            X.append(seq_features)
            y.append(seq_target)
        
        return np.array(X), np.array(y)
    
    def process(self, df: pd.DataFrame, 
                normalize: bool = True,
                handle_missing: str = 'interpolate') -> pd.DataFrame:
        """
        完整的数据预处理流程
        
        Args:
            df: 原始数据DataFrame
            normalize: 是否进行特征标准化
            handle_missing: 缺失值处理策略
        
        Returns:
            处理后的DataFrame
        """
        self.logger.info("Starting data preprocessing...")
        
        # 数据清洗
        df = self.clean_data(df)
        self.logger.info(f"After cleaning: {len(df)} records")
        
        # 特征提取
        df = self.extract_features(df)
        self.logger.info("Features extracted")
        
        # 处理缺失值
        df = self.handle_missing_values(df, strategy=handle_missing)
        self.logger.info("Missing values handled")
        
        # 特征标准化
        if normalize:
            df = self.normalize_features(df)
            self.logger.info("Features normalized")
        
        self.logger.info("Data preprocessing completed")
        
        return df


def load_and_preprocess_data(file_path: str, 
                           config: Dict = None,
                           **kwargs) -> pd.DataFrame:
    """
    加载和预处理数据的便捷函数
    
    Args:
        file_path: 数据文件路径
        config: 配置字典
        **kwargs: 其他参数
    
    Returns:
        处理后的DataFrame
    """
    # 加载数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 预处理
    preprocessor = TrajectoryPreprocessor(config)
    df = preprocessor.process(df, **kwargs)
    
    return df