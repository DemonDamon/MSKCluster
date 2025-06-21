"""
基础检测器抽象类
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import joblib
import os


class BaseAnomalyDetector(ABC):
    """异常检测器基类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 模型参数
        self.model_params = {}
        self.feature_columns = []
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'BaseAnomalyDetector':
        """
        训练检测器
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        
        Returns:
            训练后的检测器
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果 (0: 正常, 1: 异常)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X: 特征数据
        
        Returns:
            异常概率
        """
        pass
    
    def fit_predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        训练并预测
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        
        Returns:
            预测结果
        """
        self.fit(X, y)
        return self.predict(X)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算异常分数
        
        Args:
            X: 特征数据
        
        Returns:
            异常分数 (分数越高越异常)
        """
        return self.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'config': self.config,
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'class_name': self.__class__.__name__
        }
        
        # 保存模型特定的数据
        model_data.update(self._get_model_data())
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseAnomalyDetector':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            加载的检测器
        """
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.model_params = model_data['model_params']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']
        
        # 加载模型特定的数据
        self._set_model_data(model_data)
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def _get_model_data(self) -> Dict:
        """
        获取模型特定的数据 (子类重写)
        
        Returns:
            模型数据字典
        """
        return {}
    
    def _set_model_data(self, model_data: Dict) -> None:
        """
        设置模型特定的数据 (子类重写)
        
        Args:
            model_data: 模型数据字典
        """
        pass
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        验证输入数据
        
        Args:
            X: 输入数据
        
        Returns:
            验证后的数据
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if len(X) == 0:
            raise ValueError("Input DataFrame is empty")
        
        # 检查必要的列
        if self.is_fitted and self.feature_columns:
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns: {missing_cols}")
        
        return X
    
    def _check_fitted(self) -> None:
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性 (如果支持)
        
        Returns:
            特征重要性字典
        """
        return {}
    
    def get_params(self) -> Dict:
        """
        获取模型参数
        
        Returns:
            参数字典
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseAnomalyDetector':
        """
        设置模型参数
        
        Args:
            **params: 参数
        
        Returns:
            检测器实例
        """
        self.config.update(params)
        return self
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(config={self.config}, fitted={self.is_fitted})"


class SupervisedDetector(BaseAnomalyDetector):
    """有监督检测器基类"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SupervisedDetector':
        """
        训练有监督检测器
        
        Args:
            X: 特征数据
            y: 标签数据
        
        Returns:
            训练后的检测器
        """
        if y is None:
            raise ValueError("Supervised detector requires labels (y)")
        
        X = self._validate_input(X)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        self._fit_model(X, y)
        self.is_fitted = True
        
        return self
    
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        实际的模型训练逻辑 (子类实现)
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        pass


class UnsupervisedDetector(BaseAnomalyDetector):
    """无监督检测器基类"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'UnsupervisedDetector':
        """
        训练无监督检测器
        
        Args:
            X: 特征数据
            y: 标签数据 (忽略)
        
        Returns:
            训练后的检测器
        """
        X = self._validate_input(X)
        
        self._fit_model(X)
        self.is_fitted = True
        
        return self
    
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame) -> None:
        """
        实际的模型训练逻辑 (子类实现)
        
        Args:
            X: 特征数据
        """
        pass


class RuleBasedDetector(BaseAnomalyDetector):
    """基于规则的检测器基类"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.rules = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RuleBasedDetector':
        """
        训练基于规则的检测器 (通常不需要训练)
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        
        Returns:
            检测器实例
        """
        X = self._validate_input(X)
        
        # 基于规则的检测器可能需要从数据中学习阈值
        self._learn_thresholds(X, y)
        self.is_fitted = True
        
        return self
    
    def _learn_thresholds(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        从数据中学习阈值 (子类可重写)
        
        Args:
            X: 特征数据
            y: 标签数据 (可选)
        """
        pass
    
    @abstractmethod
    def _apply_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        应用规则 (子类实现)
        
        Args:
            X: 特征数据
        
        Returns:
            异常分数
        """
        pass
    
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
        
        return self._apply_rules(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        scores = self.predict_proba(X)
        threshold = self.config.get('threshold', 0.5)
        return (scores > threshold).astype(int)