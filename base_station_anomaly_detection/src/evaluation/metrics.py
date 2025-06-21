"""
评估指标模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyDetectionMetrics:
    """异常检测评估指标类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化评估指标
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.false_positive_cost = self.config.get('false_positive_cost', 1)
        self.false_negative_cost = self.config.get('false_negative_cost', 10)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算基础评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
        
        Returns:
            基础指标字典
        """
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # F2分数 (更重视召回率)
        metrics['f2_score'] = self._calculate_f2_score(metrics['precision'], metrics['recall'])
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn
        
        # 特异度
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 假阳性率和假阴性率
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, 
                                 y_scores: np.ndarray,
                                 y_pred: np.ndarray = None) -> Dict[str, float]:
        """
        计算高级评估指标
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            y_pred: 预测标签 (可选)
        
        Returns:
            高级指标字典
        """
        metrics = {}
        
        # 如果没有预测标签，使用0.5作为阈值
        if y_pred is None:
            y_pred = (y_scores > 0.5).astype(int)
        
        # 基础指标
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        metrics.update(basic_metrics)
        
        # AUC指标
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.5
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)
        
        # 成本敏感指标
        metrics['total_cost'] = self._calculate_total_cost(
            basic_metrics['false_positives'],
            basic_metrics['false_negatives']
        )
        
        # 平衡准确率
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Matthews相关系数
        metrics['mcc'] = self._calculate_mcc(
            basic_metrics['true_positives'],
            basic_metrics['true_negatives'],
            basic_metrics['false_positives'],
            basic_metrics['false_negatives']
        )
        
        return metrics
    
    def _calculate_f2_score(self, precision: float, recall: float) -> float:
        """计算F2分数"""
        if precision + recall == 0:
            return 0
        return 5 * precision * recall / (4 * precision + recall)
    
    def _calculate_total_cost(self, fp: int, fn: int) -> float:
        """计算总成本"""
        return self.false_positive_cost * fp + self.false_negative_cost * fn
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """计算Matthews相关系数"""
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0
        return numerator / denominator
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, 
                                  y_scores: np.ndarray,
                                  thresholds: np.ndarray = None) -> pd.DataFrame:
        """
        计算不同阈值下的指标
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            thresholds: 阈值数组
        
        Returns:
            阈值指标DataFrame
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            metrics = self.calculate_basic_metrics(y_true, y_pred)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true: np.ndarray, 
                             y_scores: np.ndarray,
                             metric: str = 'f2_score') -> Tuple[float, float]:
        """
        寻找最优阈值
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            metric: 优化指标
        
        Returns:
            (最优阈值, 最优指标值)
        """
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_scores)
        
        if metric == 'cost':
            # 最小化成本
            best_idx = threshold_metrics['total_cost'].idxmin()
        else:
            # 最大化指标
            best_idx = threshold_metrics[metric].idxmax()
        
        optimal_threshold = threshold_metrics.loc[best_idx, 'threshold']
        optimal_value = threshold_metrics.loc[best_idx, metric]
        
        return optimal_threshold, optimal_value
    
    def calculate_class_imbalance_metrics(self, y_true: np.ndarray) -> Dict[str, float]:
        """
        计算类别不平衡相关指标
        
        Args:
            y_true: 真实标签
        
        Returns:
            不平衡指标字典
        """
        metrics = {}
        
        # 类别分布
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        
        metrics['total_samples'] = total
        metrics['positive_samples'] = counts[1] if len(counts) > 1 else 0
        metrics['negative_samples'] = counts[0] if len(counts) > 0 else total
        
        # 不平衡比率
        if metrics['positive_samples'] > 0:
            metrics['imbalance_ratio'] = metrics['negative_samples'] / metrics['positive_samples']
            metrics['positive_rate'] = metrics['positive_samples'] / total
        else:
            metrics['imbalance_ratio'] = float('inf')
            metrics['positive_rate'] = 0
        
        metrics['negative_rate'] = metrics['negative_samples'] / total
        
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     target_names: List[str] = None) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称
        
        Returns:
            分类报告字符串
        """
        if target_names is None:
            target_names = ['Normal', 'Anomaly']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            normalize: bool = False,
                            title: str = 'Confusion Matrix') -> plt.Figure:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 是否标准化
            title: 图表标题
        
        Returns:
            matplotlib图形对象
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_scores: np.ndarray,
                      title: str = 'ROC Curve') -> plt.Figure:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            title: 图表标题
        
        Returns:
            matplotlib图形对象
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                   y_scores: np.ndarray,
                                   title: str = 'Precision-Recall Curve') -> plt.Figure:
        """
        绘制PR曲线
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            title: 图表标题
        
        Returns:
            matplotlib图形对象
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = np.trapz(precision, recall)
        
        # 基线 (随机分类器)
        baseline = np.sum(y_true) / len(y_true)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Baseline (Random = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_threshold_metrics(self, y_true: np.ndarray, 
                             y_scores: np.ndarray,
                             metrics: List[str] = None,
                             title: str = 'Threshold vs Metrics') -> plt.Figure:
        """
        绘制阈值与指标的关系
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            metrics: 要绘制的指标列表
            title: 图表标题
        
        Returns:
            matplotlib图形对象
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1_score', 'f2_score']
        
        threshold_df = self.calculate_threshold_metrics(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics:
            if metric in threshold_df.columns:
                ax.plot(threshold_df['threshold'], threshold_df[metric], 
                       linewidth=2, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        return fig
    
    def evaluate_model_comprehensive(self, y_true: np.ndarray, 
                                   y_scores: np.ndarray,
                                   y_pred: np.ndarray = None,
                                   model_name: str = 'Model') -> Dict:
        """
        综合评估模型
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            y_pred: 预测标签 (可选)
            model_name: 模型名称
        
        Returns:
            综合评估结果
        """
        results = {
            'model_name': model_name,
            'data_info': self.calculate_class_imbalance_metrics(y_true)
        }
        
        # 基础和高级指标
        results['metrics'] = self.calculate_advanced_metrics(y_true, y_scores, y_pred)
        
        # 最优阈值
        optimal_threshold, optimal_f2 = self.find_optimal_threshold(y_true, y_scores, 'f2_score')
        results['optimal_threshold'] = optimal_threshold
        results['optimal_f2_score'] = optimal_f2
        
        # 在最优阈值下的预测
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        results['optimal_metrics'] = self.calculate_basic_metrics(y_true, y_pred_optimal)
        
        # 分类报告
        results['classification_report'] = self.generate_classification_report(y_true, y_pred_optimal)
        
        return results


def compare_models(models_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    比较多个模型的性能
    
    Args:
        models_results: 模型结果字典 {model_name: evaluation_results}
    
    Returns:
        比较结果DataFrame
    """
    comparison_data = []
    
    for model_name, results in models_results.items():
        metrics = results.get('optimal_metrics', results.get('metrics', {}))
        
        row = {
            'Model': model_name,
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'F2-Score': metrics.get('f2_score', 0),
            'ROC-AUC': results.get('metrics', {}).get('roc_auc', 0),
            'PR-AUC': results.get('metrics', {}).get('pr_auc', 0),
            'Total Cost': metrics.get('total_cost', 0),
            'Optimal Threshold': results.get('optimal_threshold', 0.5)
        }
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def plot_model_comparison(comparison_df: pd.DataFrame, 
                         metrics: List[str] = None,
                         title: str = 'Model Performance Comparison') -> plt.Figure:
    """
    绘制模型比较图
    
    Args:
        comparison_df: 比较结果DataFrame
        metrics: 要比较的指标
        title: 图表标题
    
    Returns:
        matplotlib图形对象
    """
    if metrics is None:
        metrics = ['Precision', 'Recall', 'F1-Score', 'F2-Score']
    
    # 过滤存在的指标
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not available_metrics:
        raise ValueError("No valid metrics found in comparison DataFrame")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.8 / len(available_metrics)
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2) * width + width / 2
        ax.bar(x + offset, comparison_df[metric], width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig