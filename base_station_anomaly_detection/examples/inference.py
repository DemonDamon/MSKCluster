"""
推理示例
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

from src.models.statistical_detector import StatisticalAnomalyDetector
from src.utils.geo_utils import calculate_distance

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    """加载训练好的模型"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'statistical_detector_simple.joblib')
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先运行 train_simple.py 训练模型")
        return None
    
    logger.info(f"加载模型: {model_path}")
    detector = StatisticalAnomalyDetector()
    detector.load_model(model_path)
    
    return detector


def create_test_trajectory():
    """创建测试轨迹数据"""
    logger.info("创建测试轨迹数据...")
    
    # 创建一个包含正常和异常点的测试轨迹
    start_time = datetime.now()
    trajectory = []
    
    # 正常轨迹 (在北京市中心附近)
    base_lat, base_lon = 39.9042, 116.4074
    
    for i in range(100):
        timestamp = start_time + timedelta(minutes=i*5)
        
        if i < 50:
            # 前50个点：正常移动
            lat = base_lat + np.random.normal(0, 0.01)
            lon = base_lon + np.random.normal(0, 0.01)
            label = 0  # 正常
        else:
            # 后50个点：包含一些异常
            if i in [55, 60, 75, 85]:  # 注入异常点
                # 大幅跳跃
                lat = base_lat + np.random.uniform(-0.1, 0.1)
                lon = base_lon + np.random.uniform(-0.1, 0.1)
                label = 1  # 异常
            else:
                # 正常移动
                lat = base_lat + np.random.normal(0, 0.01)
                lon = base_lon + np.random.normal(0, 0.01)
                label = 0  # 正常
        
        trajectory.append({
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'label': label
        })
    
    df = pd.DataFrame(trajectory)
    logger.info(f"测试轨迹创建完成: {len(df)} 个点，其中 {df['label'].sum()} 个异常点")
    
    return df


def preprocess_test_data(df):
    """预处理测试数据"""
    logger.info("预处理测试数据...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算基础轨迹特征
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 计算距离和速度
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
    
    return df


def run_inference(detector, test_data):
    """运行推理"""
    logger.info("运行异常检测推理...")
    
    # 准备特征
    feature_cols = ['distance', 'speed_kmh', 'acceleration', 'hour', 'is_daytime', 'latitude', 'longitude']
    X_test = test_data[['timestamp'] + feature_cols]
    
    # 预测
    y_pred = detector.predict(X_test)
    y_scores = detector.predict_proba(X_test)
    
    # 添加预测结果到数据中
    test_data['predicted'] = y_pred
    test_data['anomaly_score'] = y_scores
    
    logger.info(f"推理完成: 检测到 {y_pred.sum()} 个异常点")
    
    return test_data


def analyze_results(results):
    """分析结果"""
    logger.info("分析检测结果...")
    
    # 基本统计
    total_points = len(results)
    true_anomalies = results['label'].sum()
    predicted_anomalies = results['predicted'].sum()
    
    # 计算准确性指标
    tp = ((results['label'] == 1) & (results['predicted'] == 1)).sum()
    fp = ((results['label'] == 0) & (results['predicted'] == 1)).sum()
    tn = ((results['label'] == 0) & (results['predicted'] == 0)).sum()
    fn = ((results['label'] == 1) & (results['predicted'] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("异常检测结果分析")
    print("="*50)
    print(f"总数据点: {total_points}")
    print(f"真实异常点: {true_anomalies}")
    print(f"预测异常点: {predicted_anomalies}")
    print(f"正确检测: {tp}")
    print(f"误报: {fp}")
    print(f"漏报: {fn}")
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1:.3f}")
    
    # 显示检测到的异常点详情
    detected_anomalies = results[results['predicted'] == 1]
    if len(detected_anomalies) > 0:
        print(f"\n检测到的异常点详情:")
        print("-" * 30)
        for idx, row in detected_anomalies.iterrows():
            status = "✓" if row['label'] == 1 else "✗"
            print(f"  点 {idx}: {status} 分数={row['anomaly_score']:.3f}, "
                  f"速度={row['speed_kmh']:.1f}km/h, "
                  f"距离={row['distance']:.0f}m")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def visualize_results(results):
    """可视化结果"""
    logger.info("生成可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 轨迹图
    normal_points = results[results['label'] == 0]
    true_anomalies = results[results['label'] == 1]
    predicted_anomalies = results[results['predicted'] == 1]
    
    axes[0, 0].scatter(normal_points['longitude'], normal_points['latitude'], 
                      c='blue', alpha=0.6, s=20, label='Normal')
    axes[0, 0].scatter(true_anomalies['longitude'], true_anomalies['latitude'], 
                      c='red', s=100, marker='x', label='True Anomalies')
    axes[0, 0].scatter(predicted_anomalies['longitude'], predicted_anomalies['latitude'], 
                      c='orange', s=50, marker='o', alpha=0.7, label='Predicted Anomalies')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('Trajectory with Anomalies')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 异常分数时间序列
    axes[0, 1].plot(results.index, results['anomaly_score'], 'b-', alpha=0.7, label='Anomaly Score')
    axes[0, 1].scatter(results[results['label'] == 1].index, 
                      results[results['label'] == 1]['anomaly_score'], 
                      c='red', s=50, marker='x', label='True Anomalies')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Anomaly Score')
    axes[0, 1].set_title('Anomaly Scores Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 速度时间序列
    axes[1, 0].plot(results.index, results['speed_kmh'], 'g-', alpha=0.7, label='Speed')
    axes[1, 0].scatter(results[results['label'] == 1].index, 
                      results[results['label'] == 1]['speed_kmh'], 
                      c='red', s=50, marker='x', label='True Anomalies')
    axes[1, 0].set_xlabel('Time Index')
    axes[1, 0].set_ylabel('Speed (km/h)')
    axes[1, 0].set_title('Speed Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 异常分数分布
    normal_scores = results[results['label'] == 0]['anomaly_score']
    anomaly_scores = results[results['label'] == 1]['anomaly_score']
    
    axes[1, 1].hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue')
    axes[1, 1].hist(anomaly_scores, bins=20, alpha=0.7, label='Anomalies', color='red')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Anomaly Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'inference_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果已保存到: {output_path}")
    
    return fig


def main():
    """主函数"""
    logger.info("开始异常检测推理...")
    
    # 加载模型
    detector = load_model()
    if detector is None:
        return
    
    # 创建测试数据
    test_data = create_test_trajectory()
    
    # 预处理数据
    test_data = preprocess_test_data(test_data)
    
    # 运行推理
    results = run_inference(detector, test_data)
    
    # 分析结果
    metrics = analyze_results(results)
    
    # 可视化结果
    fig = visualize_results(results)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'inference_results.csv')
    results.to_csv(results_path, index=False)
    logger.info(f"推理结果已保存到: {results_path}")
    
    return results, metrics


if __name__ == "__main__":
    results, metrics = main()
    print("\n推理完成！")
    print("可以查看生成的图表和结果文件。")