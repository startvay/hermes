"""
Hermes Agent 进化框架 - 核心模块
基于工程控制论设计的多Agent协作系统

作者: Hermes Agent
日期: 2026-04-21
版本: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "Hermes Agent"

from .base_agent import BaseAgent
from .task import Task, TaskResult, TaskPriority, TaskStatus
from .message_bus import MessageBus, get_message_bus
from .agent_registry import AgentRegistry, get_registry
from .signal_processor import SignalProcessor, Signal
from .robustness import UncertaintyModel, RobustnessChecker, NoiseFilter
from .adaptive import EnvironmentSensor, AdaptiveController, SelfOptimizer
from .coordinator import AgentCoordinator
from .learning_engine import LearningEngine, Experience
from .decision_engine import DecisionEngine, DecisionType, DecisionOption
from .performance import PerformanceMonitor, PerformanceOptimizer, track_performance

__all__ = [
    # 基础类
    "BaseAgent",
    "Task",
    "TaskResult",
    "TaskPriority",
    "TaskStatus",
    
    # 通信
    "MessageBus",
    "get_message_bus",
    
    # 注册表
    "AgentRegistry",
    "get_registry",
    
    # 信号处理
    "SignalProcessor",
    "Signal",
    
    # 鲁棒性
    "UncertaintyModel",
    "RobustnessChecker",
    "NoiseFilter",
    
    # 自适应
    "EnvironmentSensor",
    "AdaptiveController",
    "SelfOptimizer",
    
    # 协调器
    "AgentCoordinator",
    
    # 学习引擎
    "LearningEngine",
    "Experience",
    
    # 决策引擎
    "DecisionEngine",
    "DecisionType",
    "DecisionOption",
    
    # 性能监控
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "track_performance",
]
