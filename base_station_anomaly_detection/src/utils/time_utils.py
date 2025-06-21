"""
时间处理工具模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Tuple


def parse_timestamp(timestamp: Union[str, pd.Timestamp, datetime]) -> pd.Timestamp:
    """
    解析时间戳
    
    Args:
        timestamp: 时间戳 (字符串、pandas.Timestamp或datetime)
    
    Returns:
        pandas.Timestamp对象
    """
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, datetime):
        return pd.Timestamp(timestamp)
    elif isinstance(timestamp, pd.Timestamp):
        return timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def is_daytime(timestamp: pd.Timestamp, day_start: int = 7, day_end: int = 22) -> bool:
    """
    判断是否为白天时间
    
    Args:
        timestamp: 时间戳
        day_start: 白天开始时间 (小时)
        day_end: 白天结束时间 (小时)
    
    Returns:
        是否为白天
    """
    hour = timestamp.hour
    return day_start <= hour < day_end


def get_time_period(timestamp: pd.Timestamp) -> str:
    """
    获取时间段
    
    Args:
        timestamp: 时间戳
    
    Returns:
        时间段 ('morning', 'afternoon', 'evening', 'night')
    """
    hour = timestamp.hour
    
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'


def get_weekday_type(timestamp: pd.Timestamp) -> str:
    """
    获取工作日类型
    
    Args:
        timestamp: 时间戳
    
    Returns:
        工作日类型 ('weekday', 'weekend')
    """
    weekday = timestamp.weekday()
    return 'weekend' if weekday >= 5 else 'weekday'


def calculate_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    计算时间特征
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        添加了时间特征的DataFrame
    """
    df = df.copy()
    
    # 确保时间戳列是datetime类型
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 基础时间特征
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['weekday'] = df[timestamp_col].dt.weekday
    df['month'] = df[timestamp_col].dt.month
    df['day_of_year'] = df[timestamp_col].dt.dayofyear
    
    # 时间段特征
    df['is_daytime'] = df[timestamp_col].apply(is_daytime)
    df['time_period'] = df[timestamp_col].apply(get_time_period)
    df['weekday_type'] = df[timestamp_col].apply(get_weekday_type)
    
    # 周期性特征 (正弦余弦编码)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def resample_trajectory(df: pd.DataFrame, 
                       timestamp_col: str = 'timestamp',
                       freq: str = '5T',
                       method: str = 'interpolate') -> pd.DataFrame:
    """
    重采样轨迹数据
    
    Args:
        df: 轨迹数据DataFrame
        timestamp_col: 时间戳列名
        freq: 重采样频率 ('5T'表示5分钟)
        method: 重采样方法 ('interpolate', 'forward_fill', 'backward_fill')
    
    Returns:
        重采样后的DataFrame
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    
    # 创建新的时间索引
    start_time = df.index.min()
    end_time = df.index.max()
    new_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # 重采样
    if method == 'interpolate':
        df_resampled = df.reindex(new_index).interpolate(method='time')
    elif method == 'forward_fill':
        df_resampled = df.reindex(new_index).fillna(method='ffill')
    elif method == 'backward_fill':
        df_resampled = df.reindex(new_index).fillna(method='bfill')
    else:
        raise ValueError(f"Unsupported resampling method: {method}")
    
    return df_resampled.reset_index().rename(columns={'index': timestamp_col})


def detect_sampling_intervals(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.Series:
    """
    检测采样间隔
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        采样间隔序列 (秒)
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    intervals = df[timestamp_col].diff().dt.total_seconds()
    return intervals


def identify_time_gaps(df: pd.DataFrame, 
                      timestamp_col: str = 'timestamp',
                      max_gap: int = 3600) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    识别时间间隔过大的位置
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
        max_gap: 最大允许间隔 (秒)
    
    Returns:
        时间间隔列表 [(start_time, end_time), ...]
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    intervals = df[timestamp_col].diff().dt.total_seconds()
    gap_indices = intervals[intervals > max_gap].index
    
    gaps = []
    for idx in gap_indices:
        start_time = df.loc[idx - 1, timestamp_col]
        end_time = df.loc[idx, timestamp_col]
        gaps.append((start_time, end_time))
    
    return gaps


def split_by_time_gaps(df: pd.DataFrame,
                      timestamp_col: str = 'timestamp',
                      max_gap: int = 3600) -> List[pd.DataFrame]:
    """
    根据时间间隔分割轨迹
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
        max_gap: 最大允许间隔 (秒)
    
    Returns:
        分割后的DataFrame列表
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    intervals = df[timestamp_col].diff().dt.total_seconds()
    gap_indices = intervals[intervals > max_gap].index.tolist()
    
    # 添加起始和结束索引
    split_indices = [0] + gap_indices + [len(df)]
    
    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()
        if len(segment) > 1:  # 只保留有多个点的段
            segments.append(segment)
    
    return segments


def calculate_time_statistics(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> dict:
    """
    计算时间统计信息
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        时间统计信息字典
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    intervals = df[timestamp_col].diff().dt.total_seconds().dropna()
    
    stats = {
        'total_points': len(df),
        'time_span_hours': (df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600,
        'avg_interval_seconds': intervals.mean(),
        'median_interval_seconds': intervals.median(),
        'min_interval_seconds': intervals.min(),
        'max_interval_seconds': intervals.max(),
        'std_interval_seconds': intervals.std(),
        'start_time': df[timestamp_col].min(),
        'end_time': df[timestamp_col].max()
    }
    
    return stats


def generate_time_windows(start_time: pd.Timestamp,
                         end_time: pd.Timestamp,
                         window_size: int = 3600,
                         overlap: int = 0) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    生成时间窗口
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        window_size: 窗口大小 (秒)
        overlap: 重叠时间 (秒)
    
    Returns:
        时间窗口列表 [(start, end), ...]
    """
    windows = []
    current_start = start_time
    step = window_size - overlap
    
    while current_start < end_time:
        current_end = min(current_start + timedelta(seconds=window_size), end_time)
        windows.append((current_start, current_end))
        current_start += timedelta(seconds=step)
    
    return windows