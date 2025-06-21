"""
地理计算工具模块
"""

import numpy as np
import pandas as pd
from haversine import haversine, Unit
from typing import Tuple, List, Union
import math


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = 'meters') -> float:
    """
    计算两点间的地理距离
    
    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
        unit: 距离单位 ('meters', 'kilometers')
    
    Returns:
        距离值
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    
    unit_map = {
        'meters': Unit.METERS,
        'kilometers': Unit.KILOMETERS
    }
    
    return haversine((lat1, lon1), (lat2, lon2), unit=unit_map.get(unit, Unit.METERS))


def calculate_speed(distance: float, time_diff: float) -> float:
    """
    计算速度 (m/s)
    
    Args:
        distance: 距离 (米)
        time_diff: 时间差 (秒)
    
    Returns:
        速度 (m/s)
    """
    if time_diff <= 0 or pd.isna(distance) or pd.isna(time_diff):
        return 0.0
    
    return distance / time_diff


def calculate_acceleration(speed1: float, speed2: float, time_diff: float) -> float:
    """
    计算加速度 (m/s²)
    
    Args:
        speed1: 前一时刻速度 (m/s)
        speed2: 当前时刻速度 (m/s)
        time_diff: 时间差 (秒)
    
    Returns:
        加速度 (m/s²)
    """
    if time_diff <= 0 or pd.isna(speed1) or pd.isna(speed2) or pd.isna(time_diff):
        return 0.0
    
    return (speed2 - speed1) / time_diff


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的方位角 (度)
    
    Args:
        lat1, lon1: 起点纬度和经度
        lat2, lon2: 终点纬度和经度
    
    Returns:
        方位角 (0-360度)
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    
    # 转换为弧度
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    # 计算方位角
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # 转换为0-360度
    return (bearing_deg + 360) % 360


def calculate_direction_change(bearing1: float, bearing2: float) -> float:
    """
    计算方向变化角度
    
    Args:
        bearing1: 前一段方位角
        bearing2: 当前段方位角
    
    Returns:
        方向变化角度 (0-180度)
    """
    if pd.isna(bearing1) or pd.isna(bearing2):
        return 0.0
    
    diff = abs(bearing2 - bearing1)
    return min(diff, 360 - diff)


def is_point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    判断点是否在多边形内 (射线法)
    
    Args:
        lat, lon: 点的纬度和经度
        polygon: 多边形顶点列表 [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        是否在多边形内
    """
    if pd.isna(lat) or pd.isna(lon) or len(polygon) < 3:
        return False
    
    x, y = lon, lat
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def calculate_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算轨迹特征
    
    Args:
        df: 包含timestamp, latitude, longitude的DataFrame
    
    Returns:
        添加了特征的DataFrame
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    
    # 计算前一个点的坐标
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    
    # 计算时间差 (秒)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 计算距离 (米)
    df['distance'] = df.apply(
        lambda row: calculate_distance(
            row['prev_lat'], row['prev_lon'],
            row['latitude'], row['longitude']
        ) if not pd.isna(row['prev_lat']) else 0,
        axis=1
    )
    
    # 计算速度 (m/s)
    df['speed'] = df.apply(
        lambda row: calculate_speed(row['distance'], row['time_diff']),
        axis=1
    )
    
    # 计算速度 (km/h)
    df['speed_kmh'] = df['speed'] * 3.6
    
    # 计算加速度 (m/s²)
    df['prev_speed'] = df['speed'].shift(1)
    df['acceleration'] = df.apply(
        lambda row: calculate_acceleration(
            row['prev_speed'], row['speed'], row['time_diff']
        ) if not pd.isna(row['prev_speed']) else 0,
        axis=1
    )
    
    # 计算方位角
    df['bearing'] = df.apply(
        lambda row: calculate_bearing(
            row['prev_lat'], row['prev_lon'],
            row['latitude'], row['longitude']
        ) if not pd.isna(row['prev_lat']) else 0,
        axis=1
    )
    
    # 计算方向变化
    df['prev_bearing'] = df['bearing'].shift(1)
    df['direction_change'] = df.apply(
        lambda row: calculate_direction_change(row['prev_bearing'], row['bearing'])
        if not pd.isna(row['prev_bearing']) else 0,
        axis=1
    )
    
    # 清理临时列
    df = df.drop(['prev_lat', 'prev_lon', 'prev_speed', 'prev_bearing'], axis=1)
    
    return df


def calculate_centroid(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    计算坐标点的质心
    
    Args:
        coordinates: 坐标点列表 [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        质心坐标 (lat, lon)
    """
    if not coordinates:
        return 0.0, 0.0
    
    lats, lons = zip(*coordinates)
    return sum(lats) / len(lats), sum(lons) / len(lons)


def calculate_bounding_box(coordinates: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    计算坐标点的边界框
    
    Args:
        coordinates: 坐标点列表 [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        边界框 (min_lat, min_lon, max_lat, max_lon)
    """
    if not coordinates:
        return 0.0, 0.0, 0.0, 0.0
    
    lats, lons = zip(*coordinates)
    return min(lats), min(lons), max(lats), max(lons)


def meters_to_degrees(meters: float, latitude: float) -> float:
    """
    将米转换为度数 (近似)
    
    Args:
        meters: 距离 (米)
        latitude: 纬度
    
    Returns:
        度数
    """
    # 地球半径 (米)
    earth_radius = 6371000
    
    # 纬度方向: 1度 ≈ 111km
    lat_degree = meters / 111000
    
    # 经度方向: 随纬度变化
    lon_degree = meters / (111000 * math.cos(math.radians(latitude)))
    
    return max(lat_degree, lon_degree)


def degrees_to_meters(degrees: float, latitude: float) -> float:
    """
    将度数转换为米 (近似)
    
    Args:
        degrees: 度数
        latitude: 纬度
    
    Returns:
        距离 (米)
    """
    # 纬度方向: 1度 ≈ 111km
    lat_meters = degrees * 111000
    
    # 经度方向: 随纬度变化
    lon_meters = degrees * 111000 * math.cos(math.radians(latitude))
    
    return max(lat_meters, lon_meters)