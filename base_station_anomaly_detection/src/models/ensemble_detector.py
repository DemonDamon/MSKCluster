"""
集成异常检测器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base_detector import BaseAnomalyDetector
from .statistical_detector import StatisticalAnomalyDetector
from .trajectory_detector import TrajectoryAnomalyDetector
from .clustering_detector import ClusteringAnomalyDetector


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """集成异常检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化集成检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 集成配置
        self.statistical_weight = self.config.get('statistical_weight', 0.4)
        self.trajectory_weight = self.config.get('trajectory_weight', 0.4)
        self.clustering_weight = self.config.get('clustering_weight', 0.2)
        self.voting_strategy = self.config.get('voting_strategy', 'weighted')  # majority, weighted, soft
        
        # 子检测器
        self.detectors = {}
        self._initialize_detectors()
        
        # 集成参数
        self.ensemble_threshold = self.config.get('ensemble_threshold', 0.5)
        self.detector_weights = {}
        self.detector_performance = {}
    
    def _initialize_detectors(self) -> None:
        """初始化子检测器"""
        # 统计检测器
        statistical_config = self.config.get('statistical_detector', {})
        self.detectors['statistical'] = StatisticalAnomalyDetector(statistical_config)
        
        # 轨迹检测器
        trajectory_config = self.config.get('trajectory_detector', {})
        self.detectors['trajectory'] = TrajectoryAnomalyDetector(trajectory_config)
        
        # 聚类检测器
        clustering_config = self.config.get('clustering_detector', {})
        self.detectors['clustering'] = ClusteringAnomalyDetector(clustering_config)
        
        # 初始化权重
        self.detector_weights = {
            'statistical': self.statistical_weight,
            'trajectory': self.trajectory_weight,
            'clustering': self.clustering_weight
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'EnsembleAnomalyDetector':
        """
        训练集成检测器
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        
        Returns:
            训练后的检测器
        """
        X = self._validate_input(X)
        
        # 训练各个子检测器
        for name, detector in self.detectors.items():
            self.logger.info(f"Training {name} detector...")
            detector.fit(X, y)
        
        # 如果有标签，评估各检测器性能并调整权重
        if y is not None:
            self._evaluate_detectors(X, y)
            self._adjust_weights()
        
        self.is_fitted = True
        return self
    
    def _evaluate_detectors(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        评估各检测器性能
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        
        for name, detector in self.detectors.items():
            try:
                # 预测
                y_pred = detector.predict(X)
                y_proba = detector.predict_proba(X)
                
                # 计算指标
                f1 = f1_score(y, y_pred, zero_division=0)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
                
                # F2分数 (更重视召回率)
                f2 = self._calculate_f2_score(precision, recall)
                
                # AUC (如果可能)
                try:
                    auc = roc_auc_score(y, y_proba)
                except:
                    auc = 0.5
                
                self.detector_performance[name] = {
                    'f1_score': f1,
                    'f2_score': f2,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc
                }
                
                self.logger.info(f"{name} detector - F1: {f1:.3f}, F2: {f2:.3f}, Recall: {recall:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {name} detector: {e}")
                self.detector_performance[name] = {
                    'f1_score': 0, 'f2_score': 0, 'precision': 0, 'recall': 0, 'auc': 0.5
                }
    
    def _calculate_f2_score(self, precision: float, recall: float) -> float:
        """计算F2分数"""
        if precision + recall == 0:
            return 0
        return 5 * precision * recall / (4 * precision + recall)
    
    def _adjust_weights(self) -> None:
        """基于性能调整检测器权重"""
        if not self.detector_performance:
            return
        
        # 基于F2分数调整权重 (更重视召回率)
        total_f2 = sum(perf['f2_score'] for perf in self.detector_performance.values())
        
        if total_f2 > 0:
            for name in self.detector_weights:
                if name in self.detector_performance:
                    # 基于F2分数的权重
                    f2_weight = self.detector_performance[name]['f2_score'] / total_f2
                    
                    # 与原始权重的加权平均
                    original_weight = self.detector_weights[name]
                    self.detector_weights[name] = 0.7 * original_weight + 0.3 * f2_weight
        
        # 归一化权重
        total_weight = sum(self.detector_weights.values())
        if total_weight > 0:
            for name in self.detector_weights:
                self.detector_weights[name] /= total_weight
        
        self.logger.info(f"Adjusted weights: {self.detector_weights}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X: 特征数据
        
        Returns:
            异常概率
        """
        self._check_fitted()
        X = self._validate_input(X)
        
        # 获取各检测器的预测
        detector_scores = {}
        for name, detector in self.detectors.items():
            try:
                scores = detector.predict_proba(X)
                detector_scores[name] = scores
            except Exception as e:
                self.logger.warning(f"Failed to get scores from {name} detector: {e}")
                detector_scores[name] = np.zeros(len(X))
        
        # 集成预测
        if self.voting_strategy == 'majority':
            ensemble_scores = self._majority_voting(detector_scores, X)
        elif self.voting_strategy == 'weighted':
            ensemble_scores = self._weighted_voting(detector_scores)
        elif self.voting_strategy == 'soft':
            ensemble_scores = self._soft_voting(detector_scores, X)
        else:
            ensemble_scores = self._weighted_voting(detector_scores)
        
        return ensemble_scores
    
    def _majority_voting(self, detector_scores: Dict[str, np.ndarray], X: pd.DataFrame) -> np.ndarray:
        """多数投票"""
        ensemble_scores = np.zeros(len(X))
        
        for i in range(len(X)):
            votes = []
            for name, scores in detector_scores.items():
                # 转换为二值预测
                threshold = 0.5
                vote = 1 if scores[i] > threshold else 0
                votes.append(vote)
            
            # 多数投票
            ensemble_scores[i] = 1 if sum(votes) > len(votes) / 2 else 0
        
        return ensemble_scores
    
    def _weighted_voting(self, detector_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """加权投票"""
        ensemble_scores = np.zeros(len(list(detector_scores.values())[0]))
        
        for name, scores in detector_scores.items():
            weight = self.detector_weights.get(name, 0)
            ensemble_scores += weight * scores
        
        return ensemble_scores
    
    def _soft_voting(self, detector_scores: Dict[str, np.ndarray], X: pd.DataFrame) -> np.ndarray:
        """软投票 (考虑置信度)"""
        ensemble_scores = np.zeros(len(X))
        
        for i in range(len(X)):
            weighted_sum = 0
            total_weight = 0
            
            for name, scores in detector_scores.items():
                score = scores[i]
                weight = self.detector_weights.get(name, 0)
                
                # 置信度调整 (分数越接近0.5置信度越低)
                confidence = 2 * abs(score - 0.5)
                adjusted_weight = weight * confidence
                
                weighted_sum += adjusted_weight * score
                total_weight += adjusted_weight
            
            if total_weight > 0:
                ensemble_scores[i] = weighted_sum / total_weight
            else:
                ensemble_scores[i] = 0.5  # 默认中性分数
        
        return ensemble_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        scores = self.predict_proba(X)
        return (scores > self.ensemble_threshold).astype(int)
    
    def get_detector_predictions(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        获取各检测器的详细预测
        
        Args:
            X: 特征数据
        
        Returns:
            各检测器的预测结果
        """
        self._check_fitted()
        X = self._validate_input(X)
        
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                scores = detector.predict_proba(X)
                predictions = detector.predict(X)
                
                results[name] = {
                    'scores': scores,
                    'predictions': predictions,
                    'weight': self.detector_weights.get(name, 0)
                }
                
                if hasattr(detector, 'get_feature_importance'):
                    results[name]['feature_importance'] = detector.get_feature_importance()
                
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {name} detector: {e}")
                results[name] = {
                    'scores': np.zeros(len(X)),
                    'predictions': np.zeros(len(X)),
                    'weight': 0
                }
        
        return results
    
    def analyze_disagreement(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        分析检测器间的分歧
        
        Args:
            X: 特征数据
        
        Returns:
            分歧分析结果
        """
        detector_predictions = self.get_detector_predictions(X)
        
        # 计算分歧指标
        disagreement_metrics = {}
        
        # 预测方差
        all_scores = np.array([pred['scores'] for pred in detector_predictions.values()])
        disagreement_metrics['score_variance'] = np.var(all_scores, axis=0)
        
        # 预测分歧度
        all_predictions = np.array([pred['predictions'] for pred in detector_predictions.values()])
        disagreement_metrics['prediction_disagreement'] = np.std(all_predictions, axis=0)
        
        # 最大最小分数差
        disagreement_metrics['score_range'] = np.max(all_scores, axis=0) - np.min(all_scores, axis=0)
        
        return disagreement_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取集成特征重要性
        
        Returns:
            特征重要性字典
        """
        combined_importance = {}
        
        for name, detector in self.detectors.items():
            if hasattr(detector, 'get_feature_importance'):
                detector_importance = detector.get_feature_importance()
                weight = self.detector_weights.get(name, 0)
                
                for feature, importance in detector_importance.items():
                    if feature not in combined_importance:
                        combined_importance[feature] = 0
                    combined_importance[feature] += weight * importance
        
        return combined_importance
    
    def set_detector_weights(self, weights: Dict[str, float]) -> None:
        """
        设置检测器权重
        
        Args:
            weights: 权重字典
        """
        # 更新权重
        for name, weight in weights.items():
            if name in self.detector_weights:
                self.detector_weights[name] = weight
        
        # 归一化
        total_weight = sum(self.detector_weights.values())
        if total_weight > 0:
            for name in self.detector_weights:
                self.detector_weights[name] /= total_weight
    
    def get_detector_weights(self) -> Dict[str, float]:
        """获取检测器权重"""
        return self.detector_weights.copy()
    
    def get_detector_performance(self) -> Dict[str, Dict[str, float]]:
        """获取检测器性能"""
        return self.detector_performance.copy()
    
    def _get_model_data(self) -> Dict:
        """获取模型特定数据"""
        return {
            'detector_weights': self.detector_weights,
            'detector_performance': self.detector_performance,
            'ensemble_threshold': self.ensemble_threshold,
            'voting_strategy': self.voting_strategy,
            'detectors': {name: detector for name, detector in self.detectors.items()}
        }
    
    def _set_model_data(self, model_data: Dict) -> None:
        """设置模型特定数据"""
        self.detector_weights = model_data.get('detector_weights', {})
        self.detector_performance = model_data.get('detector_performance', {})
        self.ensemble_threshold = model_data.get('ensemble_threshold', 0.5)
        self.voting_strategy = model_data.get('voting_strategy', 'weighted')
        
        if 'detectors' in model_data:
            self.detectors = model_data['detectors']


class AdaptiveEnsembleDetector(EnsembleAnomalyDetector):
    """自适应集成检测器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化自适应集成检测器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 自适应参数
        self.adaptation_window = self.config.get('adaptation_window', 100)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.performance_history = {name: [] for name in self.detector_weights.keys()}
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        自适应预测异常概率
        
        Args:
            X: 特征数据
        
        Returns:
            异常概率
        """
        # 更新性能历史
        self._update_performance_history(X)
        
        # 自适应调整权重
        self._adaptive_weight_adjustment()
        
        # 调用父类方法
        return super().predict_proba(X)
    
    def _update_performance_history(self, X: pd.DataFrame) -> None:
        """更新性能历史"""
        # 简化实现：基于预测一致性更新性能
        detector_predictions = self.get_detector_predictions(X)
        
        # 计算各检测器与集成结果的一致性
        ensemble_scores = super().predict_proba(X)
        
        for name, pred_data in detector_predictions.items():
            detector_scores = pred_data['scores']
            
            # 计算相关性作为性能指标
            if len(detector_scores) > 1:
                correlation = np.corrcoef(detector_scores, ensemble_scores)[0, 1]
                if not np.isnan(correlation):
                    self.performance_history[name].append(abs(correlation))
                    
                    # 保持历史窗口大小
                    if len(self.performance_history[name]) > self.adaptation_window:
                        self.performance_history[name] = self.performance_history[name][-self.adaptation_window:]
    
    def _adaptive_weight_adjustment(self) -> None:
        """自适应权重调整"""
        if all(len(history) < 10 for history in self.performance_history.values()):
            return
        
        # 基于近期性能调整权重
        recent_performance = {}
        for name, history in self.performance_history.items():
            if len(history) >= 10:
                recent_performance[name] = np.mean(history[-10:])
            else:
                recent_performance[name] = self.detector_weights.get(name, 0)
        
        # 计算新权重
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            new_weights = {}
            for name in self.detector_weights:
                if name in recent_performance:
                    new_weight = recent_performance[name] / total_performance
                    
                    # 平滑更新
                    current_weight = self.detector_weights[name]
                    new_weights[name] = (1 - self.adaptation_rate) * current_weight + \
                                       self.adaptation_rate * new_weight
                else:
                    new_weights[name] = self.detector_weights[name]
            
            # 归一化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for name in new_weights:
                    new_weights[name] /= total_weight
                
                self.detector_weights = new_weights