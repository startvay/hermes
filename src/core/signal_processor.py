"""
信号处理与感知系统
基于工程控制论第6-8章设计

功能：
- 多模态信号输入处理
- 离散采样与连续信号统一
- 时间延迟补偿
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """信号数据结构"""
    signal_id: str
    signal_type: str  # text, numeric, binary, event
    data: Any
    timestamp: datetime
    source: str
    quality: float = 1.0  # 信号质量 0-1


class SignalProcessor:
    """
    信号处理器
    
    基于工程控制论第6-8章：
    - 交流系统 → 多模态信号处理
    - 采样系统 → 离散采样
    - 时滞系统 → 延迟补偿
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.signal_buffer: List[Signal] = []
        self.sampling_rate = 10  # Hz
        self.delay_compensation = 0.5  # 秒
        
    def process_signal(self, signal: Signal) -> Signal:
        """
        处理信号
        
        Args:
            signal: 输入信号
            
        Returns:
            Signal: 处理后的信号
        """
        # 1. 质量检查
        if signal.quality < 0.3:
            logger.warning(f"信号质量过低: {signal.quality}")
            signal.quality = 0.3
        
        # 2. 延迟补偿
        compensated_signal = self._compensate_delay(signal)
        
        # 3. 存入缓冲区
        self.signal_buffer.append(compensated_signal)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer = self.signal_buffer[-self.buffer_size:]
        
        return compensated_signal
    
    def _compensate_delay(self, signal: Signal) -> Signal:
        """延迟补偿"""
        # 简化实现：调整时间戳
        from datetime import timedelta
        signal.timestamp = signal.timestamp + timedelta(seconds=self.delay_compensation)
        return signal
    
    def get_recent_signals(self, count: int = 10, 
                           signal_type: Optional[str] = None) -> List[Signal]:
        """获取最近的信号"""
        signals = self.signal_buffer[-count:]
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return signals
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """计算信号统计"""
        if not self.signal_buffer:
            return {"count": 0}
        
        type_counts = {}
        for signal in self.signal_buffer:
            type_counts[signal.signal_type] = type_counts.get(signal.signal_type, 0) + 1
        
        avg_quality = sum(s.quality for s in self.signal_buffer) / len(self.signal_buffer)
        
        return {
            "total_signals": len(self.signal_buffer),
            "type_distribution": type_counts,
            "average_quality": avg_quality,
            "buffer_usage": len(self.signal_buffer) / self.buffer_size * 100,
        }


class SamplingController:
    """
    采样控制器
    
    基于工程控制论第7章（采样控制系统）
    """
    
    def __init__(self, sampling_rate: int = 10):
        self.sampling_rate = sampling_rate  # Hz
        self.last_sample_time = None
        
    def should_sample(self) -> bool:
        """判断是否应该采样"""
        if self.last_sample_time is None:
            return True
        
        elapsed = time.time() - self.last_sample_time
        return elapsed >= (1.0 / self.sampling_rate)
    
    def sample(self, data: Any) -> Dict[str, Any]:
        """执行采样"""
        self.last_sample_time = time.time()
        
        return {
            "data": data,
            "timestamp": self.last_sample_time,
            "sampling_rate": self.sampling_rate,
        }
