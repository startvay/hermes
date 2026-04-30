"""
执行Agent
负责命令执行、文件操作等任务
"""

import os
import subprocess
import time
from typing import Any, Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.base_agent import BaseAgent
from core.task import Task, TaskResult


class ExecutorAgent(BaseAgent):
    """
    执行Agent
    
    能力：
    - 终端命令执行
    - 文件读写操作
    - 进程管理
    """
    
    def __init__(self, agent_id: str = "executor_001"):
        super().__init__(agent_id, "Executor Agent")
        self.capabilities = [
            "command_execution",
            "file_read",
            "file_write",
            "file_list",
            "process_management",
        ]
        self.working_dir = os.getcwd()
        self.env = os.environ.copy()
    
    def can_handle(self, task: Task) -> bool:
        """判断是否能处理任务"""
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "command_execution":
                result = self._execute_command(task.parameters)
            elif task.task_type == "file_read":
                result = self._read_file(task.parameters)
            elif task.task_type == "file_write":
                result = self._write_file(task.parameters)
            elif task.task_type == "file_list":
                result = self._list_files(task.parameters)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"不支持的任务类型: {task.task_type}",
                    execution_time=time.time() - start_time,
                )
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
                metrics={"result_length": len(str(result))},
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _execute_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行终端命令
        
        Args:
            params: {"command": "ls -la", "timeout": 30}
            
        Returns:
            Dict: {"exit_code": 0, "stdout": "...", "stderr": "..."}
        """
        command = params.get("command", "")
        timeout = params.get("timeout", 30)
        workdir = params.get("workdir", self.working_dir)
        
        if not command:
            raise ValueError("命令不能为空")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=self.env,
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[:5000],  # 截断长输出
            "stderr": result.stderr[:1000],
        }
    
    def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        读取文件
        
        Args:
            params: {"path": "/path/to/file", "encoding": "utf-8"}
            
        Returns:
            Dict: {"content": "...", "size": 1234, "lines": 50}
        """
        path = params.get("path", "")
        encoding = params.get("encoding", "utf-8")
        
        if not path:
            raise ValueError("文件路径不能为空")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        
        return {
            "content": content[:10000],  # 截断大文件
            "size": len(content),
            "lines": content.count("\n") + 1,
        }
    
    def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        写入文件
        
        Args:
            params: {"path": "/path/to/file", "content": "...", "mode": "w"}
            
        Returns:
            Dict: {"success": True, "bytes_written": 1234}
        """
        path = params.get("path", "")
        content = params.get("content", "")
        mode = params.get("mode", "w")
        encoding = params.get("encoding", "utf-8")
        
        if not path:
            raise ValueError("文件路径不能为空")
        
        # 创建目录（如果不存在）
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        with open(path, mode, encoding=encoding) as f:
            bytes_written = f.write(content)
        
        return {
            "success": True,
            "bytes_written": bytes_written,
            "path": path,
        }
    
    def _list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        列出目录文件
        
        Args:
            params: {"path": "/path/to/dir", "pattern": "*.py"}
            
        Returns:
            Dict: {"files": [...], "directories": [...], "total": 10}
        """
        path = params.get("path", self.working_dir)
        pattern = params.get("pattern", None)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"目录不存在: {path}")
        
        files = []
        directories = []
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                if pattern is None or item.endswith(pattern.replace("*", "")):
                    files.append(item)
            elif os.path.isdir(item_path):
                directories.append(item)
        
        return {
            "files": sorted(files),
            "directories": sorted(directories),
            "total": len(files) + len(directories),
        }
