"""
Badcase分析模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.geo_utils import calculate_distance
from ..utils.time_utils import is_daytime


class BadcaseAnalyzer:
    """Badcase分析器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化Badcase分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.day_start = self.config.get('day_start_hour', 7)
        self.day_end = self.config.get('day_end_hour', 22)
    
    def analyze_badcases(self, X: pd.DataFrame, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_scores: np.ndarray = None) -> Dict:
        """
        分析Badcase
        
        Args:
            X: 特征数据
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测分数 (可选)
        
        Returns:
            Badcase分析结果
        """
        # 识别Badcase
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        results = {
            'false_positives': self._analyze_false_positives(X, false_positives, y_scores),
            'false_negatives': self._analyze_false_negatives(X, false_negatives, y_scores),
            'summary': self._generate_summary(X, y_true, y_pred, false_positives, false_negatives)
        }
        
        return results
    
    def _analyze_false_positives(self, X: pd.DataFrame, 
                                fp_mask: np.ndarray,
                                y_scores: np.ndarray = None) -> Dict:
        """分析假阳性"""
        if not fp_mask.any():
            return {'count': 0, 'analysis': {}}
        
        fp_data = X[fp_mask].copy()
        
        analysis = {
            'count': len(fp_data),
            'time_distribution': self._analyze_time_distribution(fp_data),
            'spatial_distribution': self._analyze_spatial_distribution(fp_data),
            'movement_patterns': self._analyze_movement_patterns(fp_data),
            'feature_analysis': self._analyze_feature_patterns(fp_data)
        }
        
        if y_scores is not None:
            fp_scores = y_scores[fp_mask]
            analysis['score_distribution'] = {
                'mean': float(np.mean(fp_scores)),
                'std': float(np.std(fp_scores)),
                'min': float(np.min(fp_scores)),
                'max': float(np.max(fp_scores)),
                'median': float(np.median(fp_scores))
            }
        
        return analysis
    
    def _analyze_false_negatives(self, X: pd.DataFrame, 
                                fn_mask: np.ndarray,
                                y_scores: np.ndarray = None) -> Dict:
        """分析假阴性"""
        if not fn_mask.any():
            return {'count': 0, 'analysis': {}}
        
        fn_data = X[fn_mask].copy()
        
        analysis = {
            'count': len(fn_data),
            'time_distribution': self._analyze_time_distribution(fn_data),
            'spatial_distribution': self._analyze_spatial_distribution(fn_data),
            'movement_patterns': self._analyze_movement_patterns(fn_data),
            'feature_analysis': self._analyze_feature_patterns(fn_data)
        }
        
        if y_scores is not None:
            fn_scores = y_scores[fn_mask]
            analysis['score_distribution'] = {
                'mean': float(np.mean(fn_scores)),
                'std': float(np.std(fn_scores)),
                'min': float(np.min(fn_scores)),
                'max': float(np.max(fn_scores)),
                'median': float(np.median(fn_scores))
            }
        
        return analysis
    
    def _analyze_time_distribution(self, data: pd.DataFrame) -> Dict:
        """分析时间分布"""
        if 'timestamp' not in data.columns:
            return {}
        
        # 小时分布
        hours = data['timestamp'].dt.hour
        hour_counts = hours.value_counts().sort_index()
        
        # 白天/夜间分布
        is_day = data['timestamp'].apply(
            lambda x: is_daytime(x, self.day_start, self.day_end)
        )
        day_count = is_day.sum()
        night_count = len(data) - day_count
        
        # 星期几分布
        weekdays = data['timestamp'].dt.weekday
        weekday_counts = weekdays.value_counts().sort_index()
        
        return {
            'hour_distribution': hour_counts.to_dict(),
            'day_night': {
                'day': int(day_count),
                'night': int(night_count)
            },
            'weekday_distribution': weekday_counts.to_dict()
        }
    
    def _analyze_spatial_distribution(self, data: pd.DataFrame) -> Dict:
        """分析空间分布"""
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            return {}
        
        # 基本统计
        lat_stats = {
            'mean': float(data['latitude'].mean()),
            'std': float(data['latitude'].std()),
            'min': float(data['latitude'].min()),
            'max': float(data['latitude'].max())
        }
        
        lon_stats = {
            'mean': float(data['longitude'].mean()),
            'std': float(data['longitude'].std()),
            'min': float(data['longitude'].min()),
            'max': float(data['longitude'].max())
        }
        
        # 计算到中心点的距离分布
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        distances = data.apply(
            lambda row: calculate_distance(
                row['latitude'], row['longitude'],
                center_lat, center_lon
            ), axis=1
        )
        
        distance_stats = {
            'mean': float(distances.mean()),
            'std': float(distances.std()),
            'min': float(distances.min()),
            'max': float(distances.max()),
            'median': float(distances.median())
        }
        
        return {
            'latitude': lat_stats,
            'longitude': lon_stats,
            'distance_from_center': distance_stats
        }
    
    def _analyze_movement_patterns(self, data: pd.DataFrame) -> Dict:
        """分析运动模式"""
        analysis = {}
        
        # 速度分析
        if 'speed_kmh' in data.columns:
            speed_stats = {
                'mean': float(data['speed_kmh'].mean()),
                'std': float(data['speed_kmh'].std()),
                'min': float(data['speed_kmh'].min()),
                'max': float(data['speed_kmh'].max()),
                'median': float(data['speed_kmh'].median())
            }
            analysis['speed'] = speed_stats
        
        # 距离分析
        if 'distance' in data.columns:
            distance_stats = {
                'mean': float(data['distance'].mean()),
                'std': float(data['distance'].std()),
                'min': float(data['distance'].min()),
                'max': float(data['distance'].max()),
                'median': float(data['distance'].median())
            }
            analysis['distance'] = distance_stats
        
        # 加速度分析
        if 'acceleration' in data.columns:
            acc_stats = {
                'mean': float(data['acceleration'].mean()),
                'std': float(data['acceleration'].std()),
                'min': float(data['acceleration'].min()),
                'max': float(data['acceleration'].max()),
                'median': float(data['acceleration'].median())
            }
            analysis['acceleration'] = acc_stats
        
        # 方向变化分析
        if 'direction_change' in data.columns:
            direction_stats = {
                'mean': float(data['direction_change'].mean()),
                'std': float(data['direction_change'].std()),
                'min': float(data['direction_change'].min()),
                'max': float(data['direction_change'].max()),
                'median': float(data['direction_change'].median())
            }
            analysis['direction_change'] = direction_stats
        
        return analysis
    
    def _analyze_feature_patterns(self, data: pd.DataFrame) -> Dict:
        """分析特征模式"""
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['latitude', 'longitude', 'user_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            return {}
        
        feature_analysis = {}
        
        for col in feature_cols:
            if col in data.columns:
                feature_analysis[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median())
                }
        
        return feature_analysis
    
    def _generate_summary(self, X: pd.DataFrame, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         fp_mask: np.ndarray, 
                         fn_mask: np.ndarray) -> Dict:
        """生成分析摘要"""
        total_samples = len(X)
        total_positives = y_true.sum()
        total_negatives = total_samples - total_positives
        
        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        
        summary = {
            'total_samples': int(total_samples),
            'total_positives': int(total_positives),
            'total_negatives': int(total_negatives),
            'false_positives': int(fp_count),
            'false_negatives': int(fn_count),
            'fp_rate': float(fp_count / total_negatives) if total_negatives > 0 else 0,
            'fn_rate': float(fn_count / total_positives) if total_positives > 0 else 0
        }
        
        # 主要问题类型
        main_issues = []
        if fp_count > fn_count:
            main_issues.append('高误报率')
        elif fn_count > fp_count:
            main_issues.append('高漏报率')
        
        if fp_count > total_negatives * 0.1:
            main_issues.append('误报过多')
        
        if fn_count > total_positives * 0.3:
            main_issues.append('漏报过多')
        
        summary['main_issues'] = main_issues
        
        return summary
    
    def generate_badcase_report(self, analysis_results: Dict) -> str:
        """生成Badcase分析报告"""
        report = []
        report.append("=" * 60)
        report.append("Badcase 分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 摘要
        summary = analysis_results['summary']
        report.append("分析摘要:")
        report.append(f"  总样本数: {summary['total_samples']:,}")
        report.append(f"  正样本数: {summary['total_positives']:,}")
        report.append(f"  负样本数: {summary['total_negatives']:,}")
        report.append(f"  假阳性数: {summary['false_positives']:,} (误报率: {summary['fp_rate']:.2%})")
        report.append(f"  假阴性数: {summary['false_negatives']:,} (漏报率: {summary['fn_rate']:.2%})")
        
        if summary['main_issues']:
            report.append(f"  主要问题: {', '.join(summary['main_issues'])}")
        
        report.append("")
        
        # 假阳性分析
        fp_analysis = analysis_results['false_positives']
        if fp_analysis['count'] > 0:
            report.append("假阳性 (误报) 分析:")
            report.append("-" * 30)
            
            # 时间分布
            if 'time_distribution' in fp_analysis:
                time_dist = fp_analysis['time_distribution']
                if 'day_night' in time_dist:
                    day_count = time_dist['day_night']['day']
                    night_count = time_dist['day_night']['night']
                    report.append(f"  时间分布: 白天 {day_count}, 夜间 {night_count}")
            
            # 运动模式
            if 'movement_patterns' in fp_analysis:
                movement = fp_analysis['movement_patterns']
                if 'speed' in movement:
                    speed_stats = movement['speed']
                    report.append(f"  平均速度: {speed_stats['mean']:.1f} km/h (最大: {speed_stats['max']:.1f})")
            
            report.append("")
        
        # 假阴性分析
        fn_analysis = analysis_results['false_negatives']
        if fn_analysis['count'] > 0:
            report.append("假阴性 (漏报) 分析:")
            report.append("-" * 30)
            
            # 时间分布
            if 'time_distribution' in fn_analysis:
                time_dist = fn_analysis['time_distribution']
                if 'day_night' in time_dist:
                    day_count = time_dist['day_night']['day']
                    night_count = time_dist['day_night']['night']
                    report.append(f"  时间分布: 白天 {day_count}, 夜间 {night_count}")
            
            # 运动模式
            if 'movement_patterns' in fn_analysis:
                movement = fn_analysis['movement_patterns']
                if 'speed' in movement:
                    speed_stats = movement['speed']
                    report.append(f"  平均速度: {speed_stats['mean']:.1f} km/h (最大: {speed_stats['max']:.1f})")
            
            report.append("")
        
        # 改进建议
        report.append("改进建议:")
        report.append("-" * 20)
        
        if summary['fp_rate'] > 0.1:
            report.append("  - 考虑提高异常检测阈值以减少误报")
            report.append("  - 分析误报的特征模式，调整特征权重")
        
        if summary['fn_rate'] > 0.3:
            report.append("  - 考虑降低异常检测阈值以减少漏报")
            report.append("  - 增加更多判别性特征")
            report.append("  - 考虑使用集成方法提高召回率")
        
        if fp_analysis['count'] > 0 and 'movement_patterns' in fp_analysis:
            movement = fp_analysis['movement_patterns']
            if 'speed' in movement and movement['speed']['mean'] < 50:
                report.append("  - 误报主要发生在低速场景，考虑调整速度阈值")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_badcase_distribution(self, X: pd.DataFrame, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> plt.Figure:
        """绘制Badcase分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 识别Badcase
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        # 1. 时间分布
        if 'timestamp' in X.columns:
            hours = X['timestamp'].dt.hour
            
            fp_hours = hours[fp_mask]
            fn_hours = hours[fn_mask]
            
            axes[0, 0].hist(fp_hours, bins=24, alpha=0.7, label='False Positives', color='red')
            axes[0, 0].hist(fn_hours, bins=24, alpha=0.7, label='False Negatives', color='blue')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Badcase Time Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 空间分布
        if 'latitude' in X.columns and 'longitude' in X.columns:
            axes[0, 1].scatter(X.loc[fp_mask, 'longitude'], X.loc[fp_mask, 'latitude'], 
                              c='red', alpha=0.6, s=20, label='False Positives')
            axes[0, 1].scatter(X.loc[fn_mask, 'longitude'], X.loc[fn_mask, 'latitude'], 
                              c='blue', alpha=0.6, s=20, label='False Negatives')
            axes[0, 1].set_xlabel('Longitude')
            axes[0, 1].set_ylabel('Latitude')
            axes[0, 1].set_title('Badcase Spatial Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 速度分布
        if 'speed_kmh' in X.columns:
            fp_speed = X.loc[fp_mask, 'speed_kmh']
            fn_speed = X.loc[fn_mask, 'speed_kmh']
            
            axes[1, 0].hist(fp_speed, bins=30, alpha=0.7, label='False Positives', color='red')
            axes[1, 0].hist(fn_speed, bins=30, alpha=0.7, label='False Negatives', color='blue')
            axes[1, 0].set_xlabel('Speed (km/h)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Badcase Speed Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 距离分布
        if 'distance' in X.columns:
            fp_distance = X.loc[fp_mask, 'distance']
            fn_distance = X.loc[fn_mask, 'distance']
            
            axes[1, 1].hist(fp_distance, bins=30, alpha=0.7, label='False Positives', color='red')
            axes[1, 1].hist(fn_distance, bins=30, alpha=0.7, label='False Negatives', color='blue')
            axes[1, 1].set_xlabel('Distance (m)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Badcase Distance Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig