"""
Agent基类
所有Agent的抽象接口，基于工程控制论设计
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .task import Task, TaskResult, TaskStatus

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Agent基类
    
    基于工程控制论第5章"不互相影响的控制"设计：
    - 单一职责：每个Agent只负责一个功能领域
    - 接口隔离：只暴露必要接口
    - 状态独立：各Agent状态互不影响
    """
    
    def __init__(self, agent_id: str, name: str):
        """
        初始化Agent
        
        Args:
            agent_id: Agent唯一标识
            name: Agent名称
        """
        self.agent_id = agent_id
        self.name = name
        self.status: str = "idle"  # idle, busy, error, offline
        self.capabilities: List[str] = []
        self.current_task: Optional[Task] = None
        self.task_history: List[TaskResult] = []
        self.created_at = datetime.now()
        
        # 性能指标
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Agent [{self.name}] 已初始化")
    
    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """
        判断是否能处理该任务
        
        Args:
            task: 任务对象
            
        Returns:
            bool: 是否能处理
        """
        pass
    
    @abstractmethod
    def execute(self, task: Task) -> TaskResult:
        """
        执行任务
        
        Args:
            task: 任务对象
            
        Returns:
            TaskResult: 执行结果
        """
        pass
    
    def validate_task(self, task: Task) -> bool:
        """
        验证任务有效性
        
        Args:
            task: 任务对象
            
        Returns:
            bool: 任务是否有效
        """
        # 检查是否能处理
        if not self.can_handle(task):
            logger.warning(f"Agent [{self.name}] 无法处理任务 {task.task_id}")
            return False
        
        # 检查Agent状态
        if self.status != "idle":
            logger.warning(f"Agent [{self.name}] 当前状态为 {self.status}，无法接收新任务")
            return False
        
        return True
    
    def run(self, task: Task) -> TaskResult:
        """
        运行任务（带状态管理）
        
        Args:
            task: 任务对象
            
        Returns:
            TaskResult: 执行结果
        """
        # 验证任务
        if not self.validate_task(task):
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=f"任务验证失败：Agent [{self.name}] 无法处理此任务",
            )
        
        # 更新状态
        self.status = "busy"
        self.current_task = task
        task.status = TaskStatus.RUNNING
        
        logger.info(f"Agent [{self.name}] 开始执行任务 {task.task_id}")
        
        try:
            # 执行任务
            result = self.execute(task)
            
            # 更新统计
            self.total_tasks += 1
            self.total_execution_time += result.execution_time
            
            if result.success:
                self.successful_tasks += 1
                task.status = TaskStatus.COMPLETED
                logger.info(f"Agent [{self.name}] 任务 {task.task_id} 执行成功")
            else:
                self.failed_tasks += 1
                task.status = TaskStatus.FAILED
                logger.error(f"Agent [{self.name}] 任务 {task.task_id} 执行失败: {result.error}")
            
            # 记录历史
            self.task_history.append(result)
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]
            
            return result
            
        except Exception as e:
            # 异常处理
            error_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=f"执行异常: {str(e)}",
            )
            
            self.total_tasks += 1
            self.failed_tasks += 1
            task.status = TaskStatus.FAILED
            self.task_history.append(error_result)
            
            logger.error(f"Agent [{self.name}] 任务 {task.task_id} 执行异常: {e}")
            return error_result
            
        finally:
            # 恢复状态
            self.status = "idle"
            self.current_task = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取Agent状态
        
        Returns:
            Dict: 状态信息
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "capabilities": self.capabilities,
            "current_task": self.current_task.task_id if self.current_task else None,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate(),
            "avg_execution_time": self.avg_execution_time(),
            "uptime": (datetime.now() - self.created_at).total_seconds(),
        }
    
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_tasks == 0:
            return 100.0
        return (self.successful_tasks / self.total_tasks) * 100
    
    def avg_execution_time(self) -> float:
        """计算平均执行时间"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_execution_time / self.total_tasks
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        self.task_history.clear()
        logger.info(f"Agent [{self.name}] 统计信息已重置")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name} status={self.status}>"
