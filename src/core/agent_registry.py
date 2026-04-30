"""
Agent注册表
管理所有Agent实例，提供查找和调度功能
"""

from typing import Any, Dict, List, Optional, Type
import logging

from .base_agent import BaseAgent
from .task import Task

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Agent注册表
    
    职责：
    - 注册/注销Agent
    - 查找适合处理任务的Agent
    - 提供Agent状态查询
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.capability_index: Dict[str, List[str]] = {}
        self._lock = None  # 可以添加线程锁
        
        logger.info("Agent注册表已初始化")
    
    def register(self, agent: BaseAgent) -> bool:
        """
        注册Agent
        
        Args:
            agent: Agent实例
            
        Returns:
            bool: 是否注册成功
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} 已存在，跳过注册")
            return False
        
        self.agents[agent.agent_id] = agent
        
        # 更新能力索引
        for capability in agent.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(agent.agent_id)
        
        logger.info(f"Agent [{agent.name}] 已注册，能力: {agent.capabilities}")
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """
        注销Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: 是否注销成功
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} 不存在")
            return False
        
        agent = self.agents[agent_id]
        
        # 清理能力索引
        for capability in agent.capabilities:
            if capability in self.capability_index:
                if agent_id in self.capability_index[capability]:
                    self.capability_index[capability].remove(agent_id)
        
        del self.agents[agent_id]
        logger.info(f"Agent [{agent.name}] 已注销")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        获取Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            BaseAgent: Agent实例
        """
        return self.agents.get(agent_id)
    
    def find_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        """
        查找适合处理任务的Agent
        
        基于工程控制论的解耦思想：
        - 根据任务类型匹配Agent能力
        - 选择成功率最高的空闲Agent
        
        Args:
            task: 任务对象
            
        Returns:
            BaseAgent: 适合的Agent，如果没有则返回None
        """
        candidates = []
        
        for agent in self.agents.values():
            if agent.can_handle(task) and agent.status == "idle":
                candidates.append(agent)
        
        if not candidates:
            logger.warning(f"没有找到能处理任务 {task.task_id} 的Agent")
            return None
        
        # 选择成功率最高的Agent
        best_agent = max(candidates, key=lambda a: a.success_rate())
        logger.info(f"为任务 {task.task_id} 选择Agent: [{best_agent.name}]")
        return best_agent
    
    def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        根据能力查找Agent
        
        Args:
            capability: 能力名称
            
        Returns:
            List[BaseAgent]: 具有该能力的Agent列表
        """
        agent_ids = self.capability_index.get(capability, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def get_all_status(self) -> List[Dict[str, Any]]:
        """
        获取所有Agent状态
        
        Returns:
            List[Dict]: Agent状态列表
        """
        return [agent.get_status() for agent in self.agents.values()]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取注册表摘要
        
        Returns:
            Dict: 摘要信息
        """
        total_tasks = sum(a.total_tasks for a in self.agents.values())
        successful_tasks = sum(a.successful_tasks for a in self.agents.values())
        
        return {
            "agent_count": len(self.agents),
            "capabilities": list(self.capability_index.keys()),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "overall_success_rate": (successful_tasks / total_tasks * 100) if total_tasks > 0 else 100.0,
            "idle_agents": sum(1 for a in self.agents.values() if a.status == "idle"),
            "busy_agents": sum(1 for a in self.agents.values() if a.status == "busy"),
        }
    
    def clear(self):
        """清空注册表"""
        self.agents.clear()
        self.capability_index.clear()
        logger.info("注册表已清空")


# 全局注册表实例
_global_registry = None

def get_registry() -> AgentRegistry:
    """获取全局注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry
