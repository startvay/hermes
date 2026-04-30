"""
任务定义模块
定义Task和TaskResult数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务数据结构"""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = TaskPriority.MEDIUM.value
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "parameters": self.parameters,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """从字典创建"""
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())[:8]),
            task_type=data.get("task_type", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", TaskPriority.MEDIUM.value),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            status=TaskStatus(data.get("status", "pending")),
        )


@dataclass
class TaskResult:
    """任务结果数据结构"""
    
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": str(self.result)[:500] if self.result else None,  # 截断长结果
            "error": self.error,
            "execution_time": self.execution_time,
            "metrics": self.metrics,
            "completed_at": self.completed_at.isoformat(),
        }
    
    def summary(self) -> str:
        """生成摘要"""
        status = "✅ 成功" if self.success else "❌ 失败"
        time_str = f"{self.execution_time:.2f}秒"
        
        summary = f"任务 {self.task_id}: {status} ({time_str})"
        if self.error:
            summary += f"\n错误: {self.error}"
        
        return summary
