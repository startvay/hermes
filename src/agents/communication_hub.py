"""
Agent间通信中心 (Agent Communication Hub)
=========================================
基于MessageBus实现Agent间实时情报推送与协作

核心功能:
1. 情报发布/订阅 — Scout发现的情报实时推送给Builder
2. 任务状态广播 — 各Agent执行状态实时同步
3. 跨Agent消息路由 — 支持定向发送和广播
4. 消息持久化 — 关键消息可查询历史

通信协议:
- intel.discovered   — 侦察连发现新情报
- intel.updated      — 情报更新
- task.assigned      — 任务分配
- task.completed     — 任务完成
- task.failed        — 任务失败
- agent.status       — Agent状态报告
- builder.request    — 工兵连请求信息
- commander.order    — 司令部命令
"""

import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from core.message_bus import MessageBus, Message, get_message_bus

logger = logging.getLogger(__name__)


@dataclass
class IntelPayload:
    """情报载荷"""
    intel_id: str
    intel_type: str          # web_search, file_scout, env_sense, data_analysis
    source_agent: str        # 来源Agent
    raw_data: Any            # 原始数据
    summary: str             # 摘要
    confidence: float        # 置信度 0-1
    timestamp: float
    metadata: Dict           # 附加元数据
    
    def to_dict(self) -> Dict:
        return {
            "intel_id": self.intel_id,
            "intel_type": self.intel_type,
            "source_agent": self.source_agent,
            "raw_data": self.raw_data,
            "summary": self.summary,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class TaskStatusPayload:
    """任务状态载荷"""
    task_id: str
    task_type: str
    agent_name: str
    status: str              # assigned, running, completed, failed
    progress: float          # 0-100
    result: Optional[Any]
    error: Optional[str]
    timestamp: float


class AgentCommunicationHub:
    """
    Agent通信中心
    
    为所有Agent提供统一的通信接口，封装MessageBus的复杂度。
    """
    
    # 标准事件类型
    EVENT_INTEL_DISCOVERED = "intel.discovered"
    EVENT_INTEL_UPDATED = "intel.updated"
    EVENT_TASK_ASSIGNED = "task.assigned"
    EVENT_TASK_COMPLETED = "task.completed"
    EVENT_TASK_FAILED = "task.failed"
    EVENT_AGENT_STATUS = "agent.status"
    EVENT_BUILDER_REQUEST = "builder.request"
    EVENT_COMMANDER_ORDER = "commander.order"
    
    def __init__(self, bus: Optional[MessageBus] = None):
        self.bus = bus or get_message_bus()
        self.intel_store: Dict[str, IntelPayload] = {}
        self.task_status_store: Dict[str, TaskStatusPayload] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.agent_presence: Dict[str, float] = {}  # agent_name -> last_seen
        
        # 启动消息处理
        self._setup_default_handlers()
        self.bus.start_processing(interval=0.05)
        
        logger.info("[通信中心] 初始化完成，消息处理已启动")
    
    def _setup_default_handlers(self):
        """设置默认消息处理器"""
        # 情报发现 -> 存入情报库
        self.bus.subscribe(self.EVENT_INTEL_DISCOVERED, self._on_intel_discovered)
        self.bus.subscribe(self.EVENT_INTEL_UPDATED, self._on_intel_updated)
        # 任务完成/失败 -> 存入状态库
        self.bus.subscribe(self.EVENT_TASK_COMPLETED, self._on_task_completed)
        self.bus.subscribe(self.EVENT_TASK_FAILED, self._on_task_failed)
        # Agent状态 -> 更新在线状态
        self.bus.subscribe(self.EVENT_AGENT_STATUS, self._on_agent_status)
    
    def _on_intel_discovered(self, data: Dict):
        """处理新情报"""
        intel = IntelPayload(**data)
        self.intel_store[intel.intel_id] = intel
        logger.info(f"[通信中心] 新情报入库: {intel.intel_type} from {intel.source_agent} "
                   f"(置信度: {intel.confidence:.2f})")
        
        # 触发注册的处理器
        for handler in self.message_handlers.get("intel", []):
            try:
                handler(intel)
            except Exception as e:
                logger.error(f"[通信中心] 情报处理器异常: {e}")
    
    def _on_intel_updated(self, data: Dict):
        """处理情报更新"""
        intel_id = data.get("intel_id")
        if intel_id in self.intel_store:
            self.intel_store[intel_id].raw_data = data.get("raw_data")
            self.intel_store[intel_id].confidence = data.get("confidence", 1.0)
            logger.info(f"[通信中心] 情报更新: {intel_id}")
    
    def _on_task_completed(self, data: Dict):
        """处理任务完成"""
        status = TaskStatusPayload(**data)
        self.task_status_store[status.task_id] = status
        logger.info(f"[通信中心] 任务完成: {status.task_id} by {status.agent_name}")
    
    def _on_task_failed(self, data: Dict):
        """处理任务失败"""
        status = TaskStatusPayload(**data)
        self.task_status_store[status.task_id] = status
        logger.warning(f"[通信中心] 任务失败: {status.task_id} by {status.agent_name} "
                      f"错误: {status.error}")
    
    def _on_agent_status(self, data: Dict):
        """处理Agent状态"""
        agent_name = data.get("agent_name", "unknown")
        self.agent_presence[agent_name] = time.time()
    
    # ===== 公开API =====
    
    def publish_intel(self, intel: IntelPayload):
        """发布情报"""
        self.bus.publish(self.EVENT_INTEL_DISCOVERED, intel.to_dict(), source=intel.source_agent)
    
    def subscribe_intel(self, handler: Callable[[IntelPayload], None]):
        """订阅情报"""
        if "intel" not in self.message_handlers:
            self.message_handlers["intel"] = []
        self.message_handlers["intel"].append(handler)
        logger.info(f"[通信中心] 新增情报订阅者: {handler.__name__ if hasattr(handler, '__name__') else handler}")
    
    def broadcast_task_status(self, status: TaskStatusPayload):
        """广播任务状态"""
        event = self.EVENT_TASK_COMPLETED if status.status == "completed" else self.EVENT_TASK_FAILED
        self.bus.publish(event, {
            "task_id": status.task_id,
            "task_type": status.task_type,
            "agent_name": status.agent_name,
            "status": status.status,
            "progress": status.progress,
            "result": status.result,
            "error": status.error,
            "timestamp": status.timestamp,
        }, source=status.agent_name)
    
    def send_directive(self, target_agent: str, order: str, payload: Dict):
        """向指定Agent发送指令"""
        self.bus.publish(
            self.EVENT_COMMANDER_ORDER,
            {"target": target_agent, "order": order, "payload": payload, "timestamp": time.time()},
            source="commander",
        )
    
    def request_intel(self, requester: str, intel_type: str, query: str):
        """请求情报"""
        self.bus.publish(
            self.EVENT_BUILDER_REQUEST,
            {"requester": requester, "intel_type": intel_type, "query": query, "timestamp": time.time()},
            source=requester,
        )
    
    def get_intel_by_type(self, intel_type: str, limit: int = 10) -> List[IntelPayload]:
        """按类型查询情报"""
        intel_list = [
            intel for intel in self.intel_store.values()
            if intel.intel_type == intel_type
        ]
        intel_list.sort(key=lambda x: x.timestamp, reverse=True)
        return intel_list[:limit]
    
    def get_latest_intel(self, intel_type: Optional[str] = None) -> Optional[IntelPayload]:
        """获取最新情报"""
        if not self.intel_store:
            return None
        
        intel_list = list(self.intel_store.values())
        if intel_type:
            intel_list = [i for i in intel_list if i.intel_type == intel_type]
        
        if not intel_list:
            return None
        
        return max(intel_list, key=lambda x: x.timestamp)
    
    def get_agent_presence(self) -> Dict[str, float]:
        """获取Agent在线状态"""
        now = time.time()
        # 过滤掉超过60秒未报告的Agent
        return {
            name: now - last_seen
            for name, last_seen in self.agent_presence.items()
            if now - last_seen < 60
        }
    
    def report_status(self, agent_name: str, status: str, details: Dict):
        """报告Agent状态"""
        self.bus.publish(
            self.EVENT_AGENT_STATUS,
            {"agent_name": agent_name, "status": status, "details": details, "timestamp": time.time()},
            source=agent_name,
        )
    
    def get_stats(self) -> Dict:
        """获取通信统计"""
        bus_stats = self.bus.get_stats()
        return {
            **bus_stats,
            "intel_count": len(self.intel_store),
            "task_status_count": len(self.task_status_store),
            "online_agents": list(self.get_agent_presence().keys()),
        }
    
    def shutdown(self):
        """关闭通信中心"""
        self.bus.stop_processing()
        logger.info("[通信中心] 已关闭")


# 全局通信中心实例
_global_hub = None

def get_communication_hub() -> AgentCommunicationHub:
    """获取全局通信中心"""
    global _global_hub
    if _global_hub is None:
        _global_hub = AgentCommunicationHub()
    return _global_hub
