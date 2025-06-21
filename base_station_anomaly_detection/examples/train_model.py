"""
训练模型示例
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
from src.models.trajectory_detector import TrajectoryAnomalyDetector
from src.models.clustering_detector import ClusteringAnomalyDetector
from src.models.ensemble_detector import EnsembleAnomalyDetector
from src.evaluation.metrics import AnomalyDetectionMetrics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data():
    """加载数据"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'sample_trajectory_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请先运行 generate_data.py 生成示例数据")
        return None
    
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"数据加载完成: {df.shape}")
    return df


def main():
    """主函数"""
    logger.info("开始训练异常检测模型...")
    
    # 加载配置和数据
    config = load_config()
    df = load_data()
    
    if df is None:
        return
    
    # 数据预处理
    logger.info("数据预处理...")
    preprocessor = TrajectoryPreprocessor(config['data'])
    df_processed = preprocessor.process(df, normalize=True)
    
    # 准备特征和标签
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['timestamp', 'latitude', 'longitude', 'label', 'user_id', 'user_type']]
    
    X = df_processed[['timestamp', 'latitude', 'longitude'] + feature_cols]
    y = df_processed['label']
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"数据点数量: {len(X)}")
    logger.info(f"异常点数量: {y.sum()} ({y.mean()*100:.2f}%)")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['evaluation']['test_size'], 
        random_state=42, stratify=y
    )
    
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 初始化评估器
    evaluator = AnomalyDetectionMetrics(config['evaluation'])
    
    # 训练和评估各个模型
    models = {}
    results = {}
    
    # 1. 统计检测器
    logger.info("\n训练统计检测器...")
    statistical_detector = StatisticalAnomalyDetector(config['statistical_detector'])
    statistical_detector.fit(X_train, y_train)
    models['Statistical'] = statistical_detector
    
    # 评估统计检测器
    y_pred_stat = statistical_detector.predict(X_test)
    y_scores_stat = statistical_detector.predict_proba(X_test)
    results['Statistical'] = evaluator.evaluate_model_comprehensive(
        y_test, y_scores_stat, y_pred_stat, 'Statistical Detector'
    )
    
    # 2. 轨迹检测器
    logger.info("\n训练轨迹检测器...")
    trajectory_detector = TrajectoryAnomalyDetector(config['trajectory_detector'])
    trajectory_detector.fit(X_train, y_train)
    models['Trajectory'] = trajectory_detector
    
    # 评估轨迹检测器
    y_pred_traj = trajectory_detector.predict(X_test)
    y_scores_traj = trajectory_detector.predict_proba(X_test)
    results['Trajectory'] = evaluator.evaluate_model_comprehensive(
        y_test, y_scores_traj, y_pred_traj, 'Trajectory Detector'
    )
    
    # 3. 聚类检测器
    logger.info("\n训练聚类检测器...")
    clustering_detector = ClusteringAnomalyDetector(config['clustering_detector'])
    clustering_detector.fit(X_train, y_train)
    models['Clustering'] = clustering_detector
    
    # 评估聚类检测器
    y_pred_clust = clustering_detector.predict(X_test)
    y_scores_clust = clustering_detector.predict_proba(X_test)
    results['Clustering'] = evaluator.evaluate_model_comprehensive(
        y_test, y_scores_clust, y_pred_clust, 'Clustering Detector'
    )
    
    # 4. 集成检测器
    logger.info("\n训练集成检测器...")
    ensemble_detector = EnsembleAnomalyDetector(config['ensemble_detector'])
    ensemble_detector.fit(X_train, y_train)
    models['Ensemble'] = ensemble_detector
    
    # 评估集成检测器
    y_pred_ens = ensemble_detector.predict(X_test)
    y_scores_ens = ensemble_detector.predict_proba(X_test)
    results['Ensemble'] = evaluator.evaluate_model_comprehensive(
        y_test, y_scores_ens, y_pred_ens, 'Ensemble Detector'
    )
    
    # 打印结果
    print_results(results)
    
    # 保存模型
    save_models(models, config)
    
    # 保存结果
    save_results(results)
    
    return models, results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*80)
    print("模型评估结果")
    print("="*80)
    
    # 创建比较表格
    from src.evaluation.metrics import compare_models
    comparison_df = compare_models(results)
    
    print("\n模型性能比较:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    print("\n详细结果:")
    for model_name, result in results.items():
        print(f"\n{model_name} 检测器:")
        print("-" * 40)
        
        metrics = result['optimal_metrics']
        print(f"  最优阈值: {result['optimal_threshold']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  F2分数: {metrics['f2_score']:.4f}")
        print(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"  PR-AUC: {result['metrics']['pr_auc']:.4f}")
        
        # 混淆矩阵
        print(f"  真阳性: {metrics['true_positives']}")
        print(f"  假阳性: {metrics['false_positives']}")
        print(f"  真阴性: {metrics['true_negatives']}")
        print(f"  假阴性: {metrics['false_negatives']}")


def save_models(models, config):
    """保存训练好的模型"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    for name, model in models.items():
        model_path = os.path.join(model_dir, f'{name.lower()}_detector.joblib')
        model.save_model(model_path)
        logger.info(f"{name} 模型已保存到: {model_path}")


def save_results(results):
    """保存评估结果"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存比较结果
    from src.evaluation.metrics import compare_models
    comparison_df = compare_models(results)
    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"模型比较结果已保存到: {comparison_path}")
    
    # 保存详细结果
    import json
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 递归转换字典中的numpy类型
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(v) for v in d]
        else:
            return convert_numpy(d)
    
    results_converted = convert_dict(results)
    
    results_path = os.path.join(results_dir, 'detailed_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存到: {results_path}")


if __name__ == "__main__":
    models, results = main()
    print("\n模型训练完成！")
    print("可以运行 inference.py 进行推理测试")