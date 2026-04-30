"""
鲁棒性与不确定性处理
基于工程控制论第9章设计

功能：
- 不确定性建模
- 鲁棒性检查
- 噪声过滤
"""

import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """不确定性估计"""
    value: float
    confidence: float  # 0-1
    variance: float
    distribution: str  # normal, uniform, etc.


class UncertaintyModel:
    """
    不确定性模型
    
    基于工程控制论第9章（随机输入系统）
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        
    def estimate(self, data: Any, context: Dict = None) -> UncertaintyEstimate:
        """
        估计不确定性
        
        Args:
            data: 输入数据
            context: 上下文信息
            
        Returns:
            UncertaintyEstimate: 不确定性估计
        """
        # 基于数据类型的不确定性估计
        if isinstance(data, (int, float)):
            value = float(data)
            variance = abs(value) * 0.1  # 假设10%方差
            confidence = 0.8
        elif isinstance(data, str):
            value = len(data)
            variance = value * 0.2
            confidence = 0.6
        else:
            value = 0.0
            variance = 1.0
            confidence = 0.5
        
        estimate = UncertaintyEstimate(
            value=value,
            confidence=confidence,
            variance=variance,
            distribution="normal",
        )
        
        # 记录历史
        self.history.append({
            "data_type": type(data).__name__,
            "estimate": estimate,
        })
        
        return estimate
    
    def propagate_uncertainty(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """
        传播不确定性（多源融合）
        
        Args:
            estimates: 多个不确定性估计
            
        Returns:
            UncertaintyEstimate: 融合后的估计
        """
        if not estimates:
            return UncertaintyEstimate(0.0, 0.0, float('inf'), "uniform")
        
        # 加权平均
        total_weight = sum(e.confidence for e in estimates)
        if total_weight == 0:
            total_weight = 1.0
        
        weighted_value = sum(e.value * e.confidence for e in estimates) / total_weight
        weighted_variance = sum(e.variance * e.confidence for e in estimates) / total_weight
        avg_confidence = sum(e.confidence for e in estimates) / len(estimates)
        
        return UncertaintyEstimate(
            value=weighted_value,
            confidence=avg_confidence,
            variance=weighted_variance,
            distribution="normal",
        )


class RobustnessChecker:
    """
    鲁棒性检查器
    
    基于工程控制论的稳定性理论
    """
    
    def __init__(self, stability_threshold: float = 0.7):
        self.stability_threshold = stability_threshold
        self.check_history: List[Dict] = []
        
    def check_robustness(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查系统鲁棒性
        
        Args:
            system_state: 系统状态
            
        Returns:
            Dict: 鲁棒性评估结果
        """
        # 计算各项指标
        metrics = {
            "stability_score": self._calculate_stability(system_state),
            "perturbation_resistance": self._calculate_perturbation_resistance(system_state),
            "error_tolerance": self._calculate_error_tolerance(system_state),
        }
        
        # 综合评分
        overall_score = sum(metrics.values()) / len(metrics)
        
        result = {
            "overall_score": overall_score,
            "is_robust": overall_score >= self.stability_threshold,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        
        self.check_history.append(result)
        if len(self.check_history) > 1000:
            self.check_history = self.check_history[-1000:]
        
        return result
    
    def _calculate_stability(self, state: Dict) -> float:
        """计算稳定性得分"""
        # 简化实现
        error_rate = state.get("error_rate", 0.0)
        return max(0, 1.0 - error_rate)
    
    def _calculate_perturbation_resistance(self, state: Dict) -> float:
        """计算扰动抵抗能力"""
        # 简化实现
        return 0.8
    
    def _calculate_error_tolerance(self, state: Dict) -> float:
        """计算错误容忍度"""
        # 简化实现
        return 0.7


class NoiseFilter:
    """
    噪声过滤器
    
    基于工程控制论第16章（噪声过滤）
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer: List[float] = []
        
    def filter(self, value: float) -> float:
        """
        过滤噪声
        
        Args:
            value: 输入值
            
        Returns:
            float: 过滤后的值
        """
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        
        # 中值滤波
        sorted_buffer = sorted(self.buffer)
        median = sorted_buffer[len(sorted_buffer) // 2]
        
        return median
    
    def reset(self):
        """重置过滤器"""
        self.buffer.clear()


# 导入time模块
import time
