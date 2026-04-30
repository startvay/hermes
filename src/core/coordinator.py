"""
多Agent协调器
基于工程控制论第17章（自适应系统）设计

功能：
- 任务分解与分配
- 并行执行管理
- 结果聚合
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from .base_agent import BaseAgent
from .task import Task, TaskResult, TaskPriority, TaskStatus
from .agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Agent协调器
    
    职责：
    - 接收复杂任务
    - 分解为子任务
    - 分配给合适的Agent
    - 监控执行进度
    - 聚合执行结果
    """
    
    def __init__(self, registry: AgentRegistry, max_workers: int = 4):
        self.registry = registry
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}
        self.task_results: Dict[str, TaskResult] = {}
        
        logger.info(f"Agent协调器已初始化，最大并行数: {max_workers}")
    
    def submit_task(self, task: Task) -> str:
        """
        提交任务
        
        Args:
            task: 任务对象
            
        Returns:
            str: 任务ID
        """
        # 查找合适的Agent
        agent = self.registry.find_agent_for_task(task)
        
        if agent is None:
            logger.warning(f"没有找到能处理任务 {task.task_id} 的Agent")
            return task.task_id
        
        # 提交执行
        future = self.executor.submit(self._execute_task, agent, task)
        self.active_tasks[task.task_id] = future
        
        logger.info(f"任务 {task.task_id} 已提交给 [{agent.name}]")
        return task.task_id
    
    def _execute_task(self, agent: BaseAgent, task: Task) -> TaskResult:
        """执行任务（内部方法）"""
        result = agent.run(task)
        self.task_results[task.task_id] = result
        return result
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            TaskResult: 任务结果
        """
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            try:
                result = future.result(timeout=timeout)
                return result
            except Exception as e:
                logger.error(f"获取任务结果失败: {e}")
                return None
        
        return None
    
    def decompose_task(self, complex_task: Task) -> List[Task]:
        """
        分解复杂任务
        
        Args:
            complex_task: 复杂任务
            
        Returns:
            List[Task]: 子任务列表
        """
        subtasks = []
        
        # 根据任务类型分解
        if complex_task.task_type == "research_and_code":
            # 研究 + 编码任务
            research_task = Task(
                task_type="web_search",
                description=f"研究: {complex_task.description}",
                parameters=complex_task.parameters.get("research", {}),
                priority=complex_task.priority,
            )
            
            code_task = Task(
                task_type="code_generation",
                description=f"编码: {complex_task.description}",
                parameters=complex_task.parameters.get("code", {}),
                priority=complex_task.priority,
                dependencies=[research_task.task_id],
            )
            
            subtasks = [research_task, code_task]
        
        elif complex_task.task_type == "multi_step":
            # 多步骤任务
            steps = complex_task.parameters.get("steps", [])
            for i, step in enumerate(steps):
                subtask = Task(
                    task_type=step.get("type", "command_execution"),
                    description=f"步骤{i+1}: {step.get('description', '')}",
                    parameters=step.get("parameters", {}),
                    priority=complex_task.priority,
                    dependencies=[subtasks[-1].task_id] if subtasks else [],
                )
                subtasks.append(subtask)
        
        else:
            # 无法分解，返回原任务
            subtasks = [complex_task]
        
        logger.info(f"任务 {complex_task.task_id} 已分解为 {len(subtasks)} 个子任务")
        return subtasks
    
    def execute_parallel(self, tasks: List[Task]) -> Dict[str, TaskResult]:
        """
        并行执行多个任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            Dict[str, TaskResult]: 任务结果字典
        """
        # 提交所有任务
        task_ids = []
        for task in tasks:
            if not task.dependencies:  # 只提交无依赖的任务
                task_id = self.submit_task(task)
                task_ids.append(task_id)
        
        # 等待所有任务完成
        results = {}
        for task_id in task_ids:
            result = self.get_result(task_id, timeout=60)
            if result:
                results[task_id] = result
        
        return results
    
    def aggregate_results(self, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """
        聚合任务结果
        
        Args:
            results: 任务结果字典
            
        Returns:
            Dict: 聚合后的结果
        """
        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)
        total_time = sum(r.execution_time for r in results.values())
        
        return {
            "total_tasks": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) * 100 if results else 0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(results) if results else 0,
            "results": {tid: r.to_dict() for tid, r in results.items()},
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            "max_workers": self.max_workers,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results),
            "registry_summary": self.registry.get_summary(),
        }
    
    def shutdown(self):
        """关闭协调器"""
        self.executor.shutdown(wait=True)
        logger.info("Agent协调器已关闭")
