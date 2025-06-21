"""
简化版训练模型示例 - 用于快速测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import logging

from src.data.preprocessor import TrajectoryPreprocessor
from src.models.statistical_detector import StatisticalAnomalyDetector
from src.evaluation.metrics import AnomalyDetectionMetrics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(sample_size=5000):
    """加载数据 (采样以加快训练)"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'sample_trajectory_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请先运行 generate_data.py 生成示例数据")
        return None
    
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 采样以加快训练
    if len(df) > sample_size:
        # 保持异常样本的比例
        normal_samples = df[df['label'] == 0].sample(n=int(sample_size * 0.99), random_state=42)
        anomaly_samples = df[df['label'] == 1].sample(n=min(len(df[df['label'] == 1]), int(sample_size * 0.01)), random_state=42)
        df = pd.concat([normal_samples, anomaly_samples]).sort_values('timestamp').reset_index(drop=True)
        logger.info(f"采样后数据大小: {len(df)}")
    
    logger.info(f"数据加载完成: {df.shape}")
    return df


def main():
    """主函数"""
    logger.info("开始训练异常检测模型 (简化版)...")
    
    # 加载配置和数据
    config = load_config()
    df = load_data(sample_size=5000)  # 使用较小的数据集
    
    if df is None:
        return
    
    # 数据预处理 (简化版)
    logger.info("数据预处理...")
    
    # 手动计算基础特征，避免复杂的预处理
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算基础轨迹特征
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 计算距离和速度
    from src.utils.geo_utils import calculate_distance
    df['distance'] = df.apply(
        lambda row: calculate_distance(
            row['prev_lat'], row['prev_lon'],
            row['latitude'], row['longitude']
        ) if not pd.isna(row['prev_lat']) else 0,
        axis=1
    )
    
    df['speed'] = df['distance'] / df['time_diff'].replace(0, np.nan)
    df['speed'] = df['speed'].fillna(0)
    df['speed_kmh'] = df['speed'] * 3.6
    
    # 计算加速度
    df['prev_speed'] = df['speed'].shift(1)
    df['acceleration'] = (df['speed'] - df['prev_speed']) / df['time_diff'].replace(0, np.nan)
    df['acceleration'] = df['acceleration'].fillna(0)
    
    # 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['is_daytime'] = (df['hour'] >= 7) & (df['hour'] < 22)
    
    # 准备特征和标签
    feature_cols = ['distance', 'speed_kmh', 'acceleration', 'hour', 'is_daytime']
    
    # 添加坐标特征
    feature_cols.extend(['latitude', 'longitude'])
    
    X = df[['timestamp'] + feature_cols]
    y = df['label']
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"数据点数量: {len(X)}")
    logger.info(f"异常点数量: {y.sum()} ({y.mean()*100:.2f}%)")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 初始化评估器
    evaluator = AnomalyDetectionMetrics()
    
    # 训练统计检测器
    logger.info("\n训练统计检测器...")
    statistical_detector = StatisticalAnomalyDetector(config['statistical_detector'])
    statistical_detector.fit(X_train, y_train)
    
    # 评估模型
    logger.info("评估模型...")
    y_pred = statistical_detector.predict(X_test)
    y_scores = statistical_detector.predict_proba(X_test)
    
    # 计算评估指标
    results = evaluator.evaluate_model_comprehensive(
        y_test, y_scores, y_pred, 'Statistical Detector'
    )
    
    # 打印结果
    print_results(results)
    
    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'statistical_detector_simple.joblib')
    statistical_detector.save_model(model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    return statistical_detector, results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    
    metrics = results['optimal_metrics']
    print(f"\n统计检测器:")
    print("-" * 30)
    print(f"  最优阈值: {results['optimal_threshold']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1_score']:.4f}")
    print(f"  F2分数: {metrics['f2_score']:.4f}")
    print(f"  ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"  PR-AUC: {results['metrics']['pr_auc']:.4f}")
    
    # 混淆矩阵
    print(f"\n混淆矩阵:")
    print(f"  真阳性: {metrics['true_positives']}")
    print(f"  假阳性: {metrics['false_positives']}")
    print(f"  真阴性: {metrics['true_negatives']}")
    print(f"  假阴性: {metrics['false_negatives']}")
    
    # 数据信息
    data_info = results['data_info']
    print(f"\n数据信息:")
    print(f"  总样本数: {data_info['total_samples']}")
    print(f"  异常样本数: {data_info['positive_samples']}")
    print(f"  异常率: {data_info['positive_rate']:.4f}")
    print(f"  不平衡比率: {data_info['imbalance_ratio']:.1f}")
    
    print("\n分类报告:")
    print(results['classification_report'])


if __name__ == "__main__":
    model, results = main()
    print("\n简化版模型训练完成！")
    print("这个版本使用了较小的数据集和简化的特征，适合快速测试。")
    print("如需完整功能，请优化计算性能后运行 train_model.py")