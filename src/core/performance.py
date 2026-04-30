"""
性能优化器
基于工程控制论第8章（频率响应）和第13章（最优控制）设计

核心功能：
- 实时性能监控
- 瓶颈自动识别
- 参数自动调优
- 性能报告生成
"""

import time
import logging
import statistics
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """性能指标点"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """性能阈值"""
    name: str
    warning: float
    critical: float
    direction: str = "upper"  # upper=越小越好, lower=越大越好


class PerformanceMonitor:
    """
    性能监控器
    
    实现滑动窗口统计、阈值告警、趋势分析
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alerts: List[Dict] = []
        
        # 默认阈值
        self._register_default_thresholds()
    
    def _register_default_thresholds(self):
        """注册默认阈值"""
        defaults = [
            PerformanceThreshold("response_time", warning=2.0, critical=5.0, direction="upper"),
            PerformanceThreshold("success_rate", warning=0.8, critical=0.6, direction="lower"),
            PerformanceThreshold("error_rate", warning=0.1, critical=0.3, direction="upper"),
            PerformanceThreshold("memory_usage", warning=0.8, critical=0.95, direction="upper"),
            PerformanceThreshold("cpu_usage", warning=0.7, critical=0.9, direction="upper"),
        ]
        for t in defaults:
            self.thresholds[t.name] = t
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        
        point = MetricPoint(name=name, value=value, tags=tags or {})
        self.metrics[name].append(point)
        
        # 检查阈值
        self._check_threshold(name, value)
    
    def _check_threshold(self, name: str, value: float):
        """检查阈值"""
        if name not in self.thresholds:
            return
        
        threshold = self.thresholds[name]
        alert_level = None
        
        if threshold.direction == "upper":
            if value >= threshold.critical:
                alert_level = "CRITICAL"
            elif value >= threshold.warning:
                alert_level = "WARNING"
        else:
            if value <= threshold.critical:
                alert_level = "CRITICAL"
            elif value <= threshold.warning:
                alert_level = "WARNING"
        
        if alert_level:
            alert = {
                "metric": name,
                "level": alert_level,
                "value": value,
                "threshold": threshold.critical if alert_level == "CRITICAL" else threshold.warning,
                "timestamp": time.time(),
            }
            self.alerts.append(alert)
            logger.warning(f"性能告警 [{alert_level}] {name}={value}")
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取指标统计"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = [p.value for p in self.metrics[name]]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[-1],
            "latest": values[-1],
        }
    
    def get_trend(self, name: str, window: int = 10) -> str:
        """获取趋势 (上升/下降/稳定)"""
        if name not in self.metrics:
            return "unknown"
        
        points = list(self.metrics[name])
        if len(points) < window:
            return "insufficient_data"
        
        recent = [p.value for p in points[-window:]]
        older = [p.value for p in points[-window*2:-window]]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        change = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        
        if change > 0.1:
            return "rising"
        elif change < -0.1:
            return "falling"
        return "stable"
    
    def get_dashboard(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        dashboard = {}
        
        for name in self.metrics:
            stats = self.get_stats(name)
            trend = self.get_trend(name)
            threshold = self.thresholds.get(name)
            
            dashboard[name] = {
                **stats,
                "trend": trend,
                "threshold": {
                    "warning": threshold.warning if threshold else None,
                    "critical": threshold.critical if threshold else None,
                } if threshold else None,
            }
        
        return {
            "metrics": dashboard,
            "recent_alerts": self.alerts[-10:],
            "total_alerts": len(self.alerts),
        }


class PerformanceOptimizer:
    """
    性能优化器
    
    实现自动调参、瓶颈识别、优化建议
    """
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimization_history: List[Dict] = []
        self.param_space: Dict[str, Dict] = {}
    
    def register_param(self, name: str, min_val: float, max_val: float, 
                       current: float, step: float = None):
        """注册可调参数"""
        self.param_space[name] = {
            "min": min_val,
            "max": max_val,
            "current": current,
            "step": step or (max_val - min_val) / 10,
        }
    
    def auto_tune(self, param_name: str, metric_name: str, 
                  direction: str = "minimize") -> float:
        """
        自动调参 (爬山法)
        
        Args:
            param_name: 参数名
            metric_name: 优化目标指标
            direction: minimize 或 maximize
            
        Returns:
            建议的参数值
        """
        if param_name not in self.param_space:
            logger.warning(f"未知参数: {param_name}")
            return None
        
        param = self.param_space[param_name]
        stats = self.monitor.get_stats(metric_name)
        
        if not stats:
            return param["current"]
        
        current_metric = stats["mean"]
        
        # 简单爬山: 根据趋势调整
        trend = self.monitor.get_trend(metric_name)
        
        if direction == "minimize":
            if trend == "rising":  # 指标上升(变差), 需要调整
                new_val = param["current"] - param["step"]
            elif trend == "falling":  # 指标下降(变好), 继续同方向
                new_val = param["current"] - param["step"]
            else:
                new_val = param["current"]
        else:
            if trend == "rising":
                new_val = param["current"] + param["step"]
            elif trend == "falling":
                new_val = param["current"] + param["step"]
            else:
                new_val = param["current"]
        
        # 限制范围
        new_val = max(param["min"], min(param["max"], new_val))
        
        # 记录优化
        self.optimization_history.append({
            "param": param_name,
            "old_value": param["current"],
            "new_value": new_val,
            "metric": metric_name,
            "metric_value": current_metric,
            "timestamp": time.time(),
        })
        
        param["current"] = new_val
        logger.info(f"自动调参: {param_name} {param['current']:.3f} -> {new_val:.3f}")
        
        return new_val
    
    def identify_bottlenecks(self) -> List[Dict]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        for name in self.monitor.metrics:
            stats = self.monitor.get_stats(name)
            trend = self.monitor.get_trend(name)
            threshold = self.monitor.thresholds.get(name)
            
            score = 0
            reasons = []
            
            # 高P95
            if stats.get("p95", 0) > stats.get("mean", 0) * 2:
                score += 3
                reasons.append("P95远高于均值,存在长尾延迟")
            
            # 持续上升趋势
            if trend == "rising":
                score += 2
                reasons.append("持续恶化趋势")
            
            # 接近阈值
            if threshold:
                if threshold.direction == "upper":
                    if stats.get("mean", 0) > threshold.warning * 0.8:
                        score += 2
                        reasons.append("接近警告阈值")
                else:
                    if stats.get("mean", 0) < threshold.warning * 1.2:
                        score += 2
                        reasons.append("接近警告阈值")
            
            if score >= 2:
                bottlenecks.append({
                    "metric": name,
                    "score": score,
                    "reasons": reasons,
                    "stats": stats,
                    "trend": trend,
                })
        
        return sorted(bottlenecks, key=lambda x: x["score"], reverse=True)
    
    def get_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        bottlenecks = self.identify_bottlenecks()
        
        for b in bottlenecks:
            metric = b["metric"]
            
            if "response_time" in metric:
                recommendations.append(f"[{metric}] 响应时间偏高: 考虑增加缓存、异步处理、或减少调用频率")
            elif "error_rate" in metric:
                recommendations.append(f"[{metric}] 错误率偏高: 检查输入验证、增加重试机制、添加熔断器")
            elif "memory" in metric:
                recommendations.append(f"[{metric}] 内存使用偏高: 检查内存泄漏、优化数据结构、增加GC频率")
            elif "cpu" in metric:
                recommendations.append(f"[{metric}] CPU使用偏高: 考虑算法优化、并行化、或负载分摊")
            elif "success_rate" in metric:
                recommendations.append(f"[{metric}] 成功率偏低: 分析失败原因、增加容错机制")
            else:
                recommendations.append(f"[{metric}] 性能指标异常: {b['reasons']}")
        
        if not recommendations:
            recommendations.append("系统运行良好，暂无优化建议")
        
        return recommendations


def track_performance(monitor: PerformanceMonitor = None):
    """性能追踪装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mon = monitor or PerformanceMonitor()
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                mon.record(f"{func.__name__}_time", elapsed)
                mon.record(f"{func.__name__}_success", 1.0)
                return result
            except Exception as e:
                elapsed = time.time() - start
                mon.record(f"{func.__name__}_time", elapsed)
                mon.record(f"{func.__name__}_success", 0.0)
                raise
        
        return wrapper
    return decorator
