"""
生成示例数据
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.generator import TrajectoryGenerator
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("开始生成示例数据...")
    
    # 配置数据生成器
    config = {
        'center_lat': 39.9042,  # 北京
        'center_lon': 116.4074,
        'area_radius': 10000,   # 10km
        'day_interval': 5,      # 白天5分钟
        'night_interval': 30,   # 夜间30分钟
        'anomaly_rate': 0.0017  # 0.17%异常率
    }
    
    generator = TrajectoryGenerator(config)
    
    # 生成数据集
    logger.info("生成轨迹数据...")
    dataset = generator.generate_dataset(
        num_users=20,           # 20个用户
        days_per_user=7,        # 每个用户7天数据
        user_types=['commuter', 'resident', 'tourist']
    )
    
    # 数据统计
    total_points = len(dataset)
    anomaly_points = (dataset['label'] == 1).sum()
    anomaly_rate = anomaly_points / total_points
    
    logger.info(f"数据生成完成:")
    logger.info(f"  总数据点: {total_points}")
    logger.info(f"  异常点数: {anomaly_points}")
    logger.info(f"  异常率: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")
    logger.info(f"  用户数: {dataset['user_id'].nunique()}")
    logger.info(f"  时间跨度: {dataset['timestamp'].min()} 到 {dataset['timestamp'].max()}")
    
    # 保存数据
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'sample_trajectory_data.csv')
    dataset.to_csv(output_path, index=False)
    logger.info(f"数据已保存到: {output_path}")
    
    # 生成数据摘要
    summary = generate_data_summary(dataset)
    summary_path = os.path.join(output_dir, 'data_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    logger.info(f"数据摘要已保存到: {summary_path}")
    
    return dataset


def generate_data_summary(dataset: pd.DataFrame) -> str:
    """生成数据摘要"""
    summary = []
    summary.append("=" * 50)
    summary.append("基站坐标异常检测 - 示例数据摘要")
    summary.append("=" * 50)
    summary.append("")
    
    # 基本统计
    summary.append("基本统计:")
    summary.append(f"  总数据点: {len(dataset):,}")
    summary.append(f"  用户数量: {dataset['user_id'].nunique()}")
    summary.append(f"  时间跨度: {dataset['timestamp'].min()} 到 {dataset['timestamp'].max()}")
    summary.append("")
    
    # 异常统计
    anomaly_count = (dataset['label'] == 1).sum()
    normal_count = (dataset['label'] == 0).sum()
    anomaly_rate = anomaly_count / len(dataset)
    
    summary.append("异常统计:")
    summary.append(f"  正常点: {normal_count:,} ({normal_count/len(dataset)*100:.2f}%)")
    summary.append(f"  异常点: {anomaly_count:,} ({anomaly_rate*100:.4f}%)")
    summary.append(f"  异常率: {anomaly_rate:.6f}")
    summary.append("")
    
    # 用户类型分布
    summary.append("用户类型分布:")
    user_type_counts = dataset['user_type'].value_counts()
    for user_type, count in user_type_counts.items():
        summary.append(f"  {user_type}: {count:,}")
    summary.append("")
    
    # 时间分布
    summary.append("时间分布:")
    dataset['hour'] = dataset['timestamp'].dt.hour
    day_points = dataset[(dataset['hour'] >= 7) & (dataset['hour'] < 22)]
    night_points = dataset[(dataset['hour'] < 7) | (dataset['hour'] >= 22)]
    
    summary.append(f"  白天数据点 (7:00-22:00): {len(day_points):,}")
    summary.append(f"  夜间数据点 (22:00-7:00): {len(night_points):,}")
    summary.append("")
    
    # 地理范围
    summary.append("地理范围:")
    summary.append(f"  纬度范围: {dataset['latitude'].min():.6f} 到 {dataset['latitude'].max():.6f}")
    summary.append(f"  经度范围: {dataset['longitude'].min():.6f} 到 {dataset['longitude'].max():.6f}")
    summary.append("")
    
    # 异常类型分析 (如果有相关信息)
    if anomaly_count > 0:
        summary.append("异常点分析:")
        anomaly_data = dataset[dataset['label'] == 1]
        
        # 按用户类型分析异常
        anomaly_by_type = anomaly_data['user_type'].value_counts()
        summary.append("  按用户类型:")
        for user_type, count in anomaly_by_type.items():
            total_type = (dataset['user_type'] == user_type).sum()
            rate = count / total_type if total_type > 0 else 0
            summary.append(f"    {user_type}: {count} / {total_type} ({rate*100:.2f}%)")
        
        # 按时间段分析异常
        anomaly_day = anomaly_data[(anomaly_data['hour'] >= 7) & (anomaly_data['hour'] < 22)]
        anomaly_night = anomaly_data[(anomaly_data['hour'] < 7) | (anomaly_data['hour'] >= 22)]
        
        summary.append("  按时间段:")
        summary.append(f"    白天异常: {len(anomaly_day)} / {len(day_points)} ({len(anomaly_day)/len(day_points)*100:.2f}%)")
        summary.append(f"    夜间异常: {len(anomaly_night)} / {len(night_points)} ({len(anomaly_night)/len(night_points)*100:.2f}%)")
    
    summary.append("")
    summary.append("=" * 50)
    summary.append("数据生成完成，可用于模型训练和测试")
    summary.append("=" * 50)
    
    return "\n".join(summary)


if __name__ == "__main__":
    dataset = main()
    print("\n数据生成完成！")
    print(f"数据形状: {dataset.shape}")
    print(f"列名: {list(dataset.columns)}")
    print("\n前5行数据:")
    print(dataset.head())