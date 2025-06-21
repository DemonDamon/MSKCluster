"""
数据生成器模块 - 用于生成合成的基站定位数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from ..utils.geo_utils import calculate_distance, calculate_bearing


class TrajectoryGenerator:
    """轨迹数据生成器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化生成器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认配置
        self.center_lat = self.config.get('center_lat', 39.9042)  # 北京
        self.center_lon = self.config.get('center_lon', 116.4074)
        self.area_radius = self.config.get('area_radius', 10000)  # 10km
        
        self.day_interval = self.config.get('day_interval', 5)    # 白天5分钟
        self.night_interval = self.config.get('night_interval', 30)  # 夜间30分钟
        
        self.anomaly_rate = self.config.get('anomaly_rate', 0.0017)  # 0.17%
        
    def generate_normal_trajectory(self, 
                                 start_time: datetime,
                                 duration_hours: int = 24,
                                 user_type: str = 'commuter') -> pd.DataFrame:
        """
        生成正常轨迹
        
        Args:
            start_time: 开始时间
            duration_hours: 持续时间 (小时)
            user_type: 用户类型 ('commuter', 'resident', 'tourist')
        
        Returns:
            轨迹DataFrame
        """
        trajectory = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        # 根据用户类型设置行为模式
        if user_type == 'commuter':
            locations = self._generate_commuter_locations()
        elif user_type == 'resident':
            locations = self._generate_resident_locations()
        elif user_type == 'tourist':
            locations = self._generate_tourist_locations()
        else:
            locations = self._generate_random_locations()
        
        current_location_idx = 0
        current_lat, current_lon = locations[current_location_idx]
        
        while current_time < end_time:
            # 确定采样间隔
            hour = current_time.hour
            if 7 <= hour < 22:  # 白天
                interval = self.day_interval
            else:  # 夜间
                interval = self.night_interval
            
            # 添加当前位置点
            trajectory.append({
                'timestamp': current_time,
                'latitude': current_lat,
                'longitude': current_lon,
                'label': 0  # 正常点
            })
            
            # 移动到下一个位置
            if user_type == 'commuter':
                current_lat, current_lon, current_location_idx = self._move_commuter(
                    current_time, current_lat, current_lon, locations, current_location_idx
                )
            elif user_type == 'resident':
                current_lat, current_lon = self._move_resident(
                    current_time, current_lat, current_lon, locations
                )
            elif user_type == 'tourist':
                current_lat, current_lon = self._move_tourist(
                    current_time, current_lat, current_lon, locations
                )
            else:
                current_lat, current_lon = self._random_walk(
                    current_lat, current_lon, max_distance=100
                )
            
            # 更新时间
            current_time += timedelta(minutes=interval)
        
        return pd.DataFrame(trajectory)
    
    def _generate_commuter_locations(self) -> List[Tuple[float, float]]:
        """生成通勤者的关键位置"""
        # 家
        home_lat = self.center_lat + np.random.normal(0, 0.05)
        home_lon = self.center_lon + np.random.normal(0, 0.05)
        
        # 公司
        office_lat = self.center_lat + np.random.normal(0, 0.08)
        office_lon = self.center_lon + np.random.normal(0, 0.08)
        
        # 其他地点 (商场、餐厅等)
        other_locations = []
        for _ in range(3):
            lat = self.center_lat + np.random.normal(0, 0.06)
            lon = self.center_lon + np.random.normal(0, 0.06)
            other_locations.append((lat, lon))
        
        return [(home_lat, home_lon), (office_lat, office_lon)] + other_locations
    
    def _generate_resident_locations(self) -> List[Tuple[float, float]]:
        """生成居民的关键位置"""
        # 家附近的多个位置
        home_lat = self.center_lat + np.random.normal(0, 0.03)
        home_lon = self.center_lon + np.random.normal(0, 0.03)
        
        locations = [(home_lat, home_lon)]
        
        # 附近的商店、公园等
        for _ in range(5):
            lat = home_lat + np.random.normal(0, 0.01)
            lon = home_lon + np.random.normal(0, 0.01)
            locations.append((lat, lon))
        
        return locations
    
    def _generate_tourist_locations(self) -> List[Tuple[float, float]]:
        """生成游客的关键位置"""
        # 多个景点
        locations = []
        for _ in range(8):
            lat = self.center_lat + np.random.normal(0, 0.1)
            lon = self.center_lon + np.random.normal(0, 0.1)
            locations.append((lat, lon))
        
        return locations
    
    def _generate_random_locations(self) -> List[Tuple[float, float]]:
        """生成随机位置"""
        locations = []
        for _ in range(10):
            lat = self.center_lat + np.random.normal(0, 0.05)
            lon = self.center_lon + np.random.normal(0, 0.05)
            locations.append((lat, lon))
        
        return locations
    
    def _move_commuter(self, current_time: datetime, 
                      current_lat: float, current_lon: float,
                      locations: List[Tuple[float, float]],
                      current_location_idx: int) -> Tuple[float, float, int]:
        """通勤者移动模式"""
        hour = current_time.hour
        weekday = current_time.weekday()
        
        home_lat, home_lon = locations[0]
        office_lat, office_lon = locations[1]
        
        # 工作日模式
        if weekday < 5:  # 周一到周五
            if 7 <= hour < 9:  # 上班时间
                target_lat, target_lon = office_lat, office_lon
                current_location_idx = 1
            elif 18 <= hour < 20:  # 下班时间
                target_lat, target_lon = home_lat, home_lon
                current_location_idx = 0
            elif 12 <= hour < 14:  # 午餐时间
                if random.random() < 0.3:  # 30%概率外出
                    target_lat, target_lon = random.choice(locations[2:])
                    current_location_idx = random.randint(2, len(locations)-1)
                else:
                    target_lat, target_lon = locations[current_location_idx]
            else:
                target_lat, target_lon = locations[current_location_idx]
        else:  # 周末
            if random.random() < 0.2:  # 20%概率外出
                target_lat, target_lon = random.choice(locations[2:])
                current_location_idx = random.randint(2, len(locations)-1)
            else:
                target_lat, target_lon = home_lat, home_lon
                current_location_idx = 0
        
        # 逐步移动到目标位置
        new_lat, new_lon = self._move_towards(
            current_lat, current_lon, target_lat, target_lon, max_speed=50
        )
        
        return new_lat, new_lon, current_location_idx
    
    def _move_resident(self, current_time: datetime,
                      current_lat: float, current_lon: float,
                      locations: List[Tuple[float, float]]) -> Tuple[float, float]:
        """居民移动模式"""
        hour = current_time.hour
        
        # 大部分时间在家
        if 22 <= hour or hour < 7:  # 夜间在家
            target_lat, target_lon = locations[0]
        elif random.random() < 0.8:  # 80%时间在家附近
            target_lat, target_lon = random.choice(locations[:3])
        else:  # 偶尔外出
            target_lat, target_lon = random.choice(locations)
        
        new_lat, new_lon = self._move_towards(
            current_lat, current_lon, target_lat, target_lon, max_speed=20
        )
        
        return new_lat, new_lon
    
    def _move_tourist(self, current_time: datetime,
                     current_lat: float, current_lon: float,
                     locations: List[Tuple[float, float]]) -> Tuple[float, float]:
        """游客移动模式"""
        hour = current_time.hour
        
        if 22 <= hour or hour < 8:  # 夜间休息
            # 选择一个固定位置 (酒店)
            target_lat, target_lon = locations[0]
        else:  # 白天游览
            target_lat, target_lon = random.choice(locations)
        
        new_lat, new_lon = self._move_towards(
            current_lat, current_lon, target_lat, target_lon, max_speed=30
        )
        
        return new_lat, new_lon
    
    def _move_towards(self, current_lat: float, current_lon: float,
                     target_lat: float, target_lon: float,
                     max_speed: float = 50) -> Tuple[float, float]:
        """向目标位置移动"""
        distance = calculate_distance(current_lat, current_lon, target_lat, target_lon)
        
        if distance < 50:  # 已经很接近目标
            return target_lat + np.random.normal(0, 0.0001), target_lon + np.random.normal(0, 0.0001)
        
        # 计算移动方向
        bearing = calculate_bearing(current_lat, current_lon, target_lat, target_lon)
        
        # 随机移动距离 (不超过最大速度)
        move_distance = min(distance, np.random.uniform(10, max_speed))
        
        # 计算新位置
        bearing_rad = np.radians(bearing)
        
        # 简化的坐标计算 (适用于小距离)
        lat_change = move_distance * np.cos(bearing_rad) / 111000  # 1度纬度约111km
        lon_change = move_distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(current_lat)))
        
        new_lat = current_lat + lat_change + np.random.normal(0, 0.0001)  # 添加噪声
        new_lon = current_lon + lon_change + np.random.normal(0, 0.0001)
        
        return new_lat, new_lon
    
    def _random_walk(self, current_lat: float, current_lon: float,
                    max_distance: float = 100) -> Tuple[float, float]:
        """随机游走"""
        # 随机方向和距离
        bearing = np.random.uniform(0, 360)
        distance = np.random.uniform(0, max_distance)
        
        bearing_rad = np.radians(bearing)
        
        lat_change = distance * np.cos(bearing_rad) / 111000
        lon_change = distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(current_lat)))
        
        new_lat = current_lat + lat_change
        new_lon = current_lon + lon_change
        
        return new_lat, new_lon
    
    def inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        注入异常点
        
        Args:
            df: 正常轨迹DataFrame
        
        Returns:
            包含异常点的DataFrame
        """
        df = df.copy()
        
        # 计算需要注入的异常点数量
        total_points = len(df)
        anomaly_count = max(1, int(total_points * self.anomaly_rate))
        
        # 随机选择异常点位置
        anomaly_indices = np.random.choice(
            range(1, total_points-1),  # 避免第一个和最后一个点
            size=anomaly_count,
            replace=False
        )
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'large_jump',      # 大幅跳跃
                'drift_sequence',  # 连续漂移
                'wrong_direction', # 错误方向
                'impossible_speed' # 不可能的速度
            ], p=[0.4, 0.3, 0.2, 0.1])
            
            if anomaly_type == 'large_jump':
                df = self._inject_large_jump(df, idx)
            elif anomaly_type == 'drift_sequence':
                df = self._inject_drift_sequence(df, idx)
            elif anomaly_type == 'wrong_direction':
                df = self._inject_wrong_direction(df, idx)
            elif anomaly_type == 'impossible_speed':
                df = self._inject_impossible_speed(df, idx)
        
        return df
    
    def _inject_large_jump(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """注入大幅跳跃异常"""
        # 跳跃到远处位置
        jump_distance = np.random.uniform(5000, 20000)  # 5-20km
        jump_bearing = np.random.uniform(0, 360)
        
        current_lat = df.loc[idx, 'latitude']
        current_lon = df.loc[idx, 'longitude']
        
        bearing_rad = np.radians(jump_bearing)
        lat_change = jump_distance * np.cos(bearing_rad) / 111000
        lon_change = jump_distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(current_lat)))
        
        df.loc[idx, 'latitude'] = current_lat + lat_change
        df.loc[idx, 'longitude'] = current_lon + lon_change
        df.loc[idx, 'label'] = 1
        
        return df
    
    def _inject_drift_sequence(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """注入连续漂移异常"""
        sequence_length = min(3, len(df) - idx)
        
        # 漂移方向和距离
        drift_bearing = np.random.uniform(0, 360)
        drift_distance = np.random.uniform(1000, 5000)  # 1-5km
        
        for i in range(sequence_length):
            if idx + i >= len(df):
                break
                
            current_lat = df.loc[idx + i, 'latitude']
            current_lon = df.loc[idx + i, 'longitude']
            
            bearing_rad = np.radians(drift_bearing)
            step_distance = drift_distance * (i + 1) / sequence_length
            
            lat_change = step_distance * np.cos(bearing_rad) / 111000
            lon_change = step_distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(current_lat)))
            
            df.loc[idx + i, 'latitude'] = current_lat + lat_change
            df.loc[idx + i, 'longitude'] = current_lon + lon_change
            df.loc[idx + i, 'label'] = 1
        
        return df
    
    def _inject_wrong_direction(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """注入错误方向异常"""
        if idx == 0 or idx >= len(df) - 1:
            return df
        
        # 计算正常方向
        prev_lat = df.loc[idx - 1, 'latitude']
        prev_lon = df.loc[idx - 1, 'longitude']
        next_lat = df.loc[idx + 1, 'latitude']
        next_lon = df.loc[idx + 1, 'longitude']
        
        normal_bearing = calculate_bearing(prev_lat, prev_lon, next_lat, next_lon)
        
        # 反方向移动
        wrong_bearing = (normal_bearing + 180) % 360
        move_distance = np.random.uniform(500, 2000)
        
        current_lat = df.loc[idx, 'latitude']
        current_lon = df.loc[idx, 'longitude']
        
        bearing_rad = np.radians(wrong_bearing)
        lat_change = move_distance * np.cos(bearing_rad) / 111000
        lon_change = move_distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(current_lat)))
        
        df.loc[idx, 'latitude'] = current_lat + lat_change
        df.loc[idx, 'longitude'] = current_lon + lon_change
        df.loc[idx, 'label'] = 1
        
        return df
    
    def _inject_impossible_speed(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """注入不可能速度异常"""
        if idx == 0:
            return df
        
        # 计算时间差
        current_time = df.loc[idx, 'timestamp']
        prev_time = df.loc[idx - 1, 'timestamp']
        time_diff = (current_time - prev_time).total_seconds()
        
        if time_diff <= 0:
            return df
        
        # 设置不可能的移动距离 (对应速度>200km/h)
        impossible_distance = max(10000, time_diff * 200 / 3.6)  # 200km/h转m/s
        
        # 随机方向
        bearing = np.random.uniform(0, 360)
        
        prev_lat = df.loc[idx - 1, 'latitude']
        prev_lon = df.loc[idx - 1, 'longitude']
        
        bearing_rad = np.radians(bearing)
        lat_change = impossible_distance * np.cos(bearing_rad) / 111000
        lon_change = impossible_distance * np.sin(bearing_rad) / (111000 * np.cos(np.radians(prev_lat)))
        
        df.loc[idx, 'latitude'] = prev_lat + lat_change
        df.loc[idx, 'longitude'] = prev_lon + lon_change
        df.loc[idx, 'label'] = 1
        
        return df
    
    def generate_dataset(self, 
                        num_users: int = 100,
                        days_per_user: int = 7,
                        user_types: List[str] = None) -> pd.DataFrame:
        """
        生成完整数据集
        
        Args:
            num_users: 用户数量
            days_per_user: 每个用户的天数
            user_types: 用户类型列表
        
        Returns:
            完整数据集DataFrame
        """
        if user_types is None:
            user_types = ['commuter', 'resident', 'tourist']
        
        all_trajectories = []
        
        for user_id in range(num_users):
            user_type = np.random.choice(user_types)
            
            for day in range(days_per_user):
                start_time = datetime.now() - timedelta(days=days_per_user-day-1)
                start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # 生成正常轨迹
                trajectory = self.generate_normal_trajectory(
                    start_time, duration_hours=24, user_type=user_type
                )
                
                # 注入异常
                trajectory = self.inject_anomalies(trajectory)
                
                # 添加用户ID
                trajectory['user_id'] = user_id
                trajectory['user_type'] = user_type
                
                all_trajectories.append(trajectory)
        
        # 合并所有轨迹
        dataset = pd.concat(all_trajectories, ignore_index=True)
        dataset = dataset.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        return dataset


def generate_sample_data(output_path: str = None, **kwargs) -> pd.DataFrame:
    """
    生成示例数据的便捷函数
    
    Args:
        output_path: 输出文件路径
        **kwargs: 其他参数
    
    Returns:
        生成的数据集
    """
    generator = TrajectoryGenerator()
    dataset = generator.generate_dataset(**kwargs)
    
    if output_path:
        dataset.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return dataset