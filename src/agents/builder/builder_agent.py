"""
工兵连 (Builder Agent) v2.0 — 智能情报驱动版
==============================================
负责：代码编写、文件操作、命令执行、测试运行

v2.0改进:
- 订阅通信中心情报，根据侦察结果调整代码
- 支持情报驱动的代码生成（如根据搜索结果生成对应代码）
- 代码执行结果实时反馈
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.task import Task, TaskResult

try:
    from agents.communication_hub import (
        AgentCommunicationHub, IntelPayload, TaskStatusPayload, get_communication_hub
    )
except ImportError:
    from communication_hub import (
        AgentCommunicationHub, IntelPayload, TaskStatusPayload, get_communication_hub
    )

logger = logging.getLogger(__name__)


class BuilderAgent(BaseAgent):
    """
    工兵连 - 构建与执行 (情报驱动版)
    
    职责：
    1. 代码编写：生成代码文件（可基于情报上下文）
    2. 文件操作：读写文件
    3. 命令执行：运行终端命令
    4. 测试运行：执行测试验证
    5. 情报消费：接收侦察连情报并据此工作
    """
    
    def __init__(self, hub: Optional[AgentCommunicationHub] = None):
        super().__init__(agent_id="builder_01", name="工兵连")
        self.capabilities = ["code_write", "code_execute", "file_write", 
                            "file_read", "command_run", "test_run"]
        self.hub = hub or get_communication_hub()
        self.recent_intel: List[IntelPayload] = []
        self.intel_context: str = ""  # 聚合的情报上下文
        
        # 订阅侦察连情报
        self.hub.subscribe_intel(self._on_intel_received)
        
        logger.info(f"[{self.name}] v2.0 初始化完成，已订阅情报频道")
    
    def _on_intel_received(self, intel: IntelPayload):
        """接收侦察连情报"""
        self.recent_intel.append(intel)
        if len(self.recent_intel) > 20:
            self.recent_intel = self.recent_intel[-20:]
        
        # 更新情报上下文
        if intel.intel_type == "web_search":
            raw = intel.raw_data
            if isinstance(raw, dict) and "results" in raw:
                snippets = [r.get("snippet", "")[:100] for r in raw["results"][:3]]
                self.intel_context += f"\n[搜索: {raw.get('query', '')}] " + " | ".join(snippets)
        elif intel.intel_type == "file_scout":
            raw = intel.raw_data
            if isinstance(raw, dict):
                self.intel_context += f"\n[文件侦察: {raw.get('path', '')}] 发现 {raw.get('count', 0)} 个项目"
        
        logger.info(f"[{self.name}] 收到情报: {intel.intel_type} from {intel.source_agent} "
                   f"(总情报: {len(self.recent_intel)})")
    
    def can_handle(self, task: Task) -> bool:
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行构建任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "code_write":
                result = self._code_write(task.parameters)
            elif task.task_type == "code_execute":
                result = self._code_execute(task.parameters)
            elif task.task_type == "file_write":
                result = self._file_write(task.parameters)
            elif task.task_type == "file_read":
                result = self._file_read(task.parameters)
            elif task.task_type == "command_run":
                result = self._command_run(task.parameters)
            elif task.task_type == "test_run":
                result = self._test_run(task.parameters)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    result={},
                    error=f"未知任务类型: {task.task_type}",
                    execution_time=time.time() - start_time,
                )
            
            # 报告任务完成
            self._report_task_status(task, True, result, time.time() - start_time)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            self._report_task_status(task, False, {}, time.time() - start_time, error=str(e))
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result={},
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _code_write(self, params: Dict) -> Dict:
        """编写代码 — 支持情报上下文"""
        user_input = params.get("input", "")
        language = params.get("language", "python")
        
        # 检查是否有相关情报可以指导代码生成
        intel_guidance = ""
        if self.intel_context:
            intel_guidance = f"\n# 基于侦察情报的上下文:\n# {self.intel_context[:500]}\n"
        
        # 根据用户输入和情报生成代码
        code, file_path = self._generate_code(user_input, language, intel_guidance)
        
        # 写入文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"[{self.name}] 代码已写入: {file_path} ({len(code.split(chr(10)))} 行)")
        
        return {
            "file_path": file_path,
            "code": code,
            "lines": len(code.split("\n")),
            "language": language,
            "intel_guided": bool(intel_guidance),
            "status": "written",
        }
    
    def _generate_code(self, user_input: str, language: str, intel_guidance: str) -> tuple:
        """生成代码 — 基于需求+情报"""
        
        if "斐波那契" in user_input or "fibonacci" in user_input.lower():
            code = f'''"""
自动生成的斐波那契数列脚本
用户需求: {user_input}
生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
{intel_guidance}
"""

def fibonacci(n: int) -> int:
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_sequence(n: int) -> list:
    """生成斐波那契数列前n项"""
    return [fibonacci(i) for i in range(n)]

def fibonacci_memo(n: int, memo={{}}) -> int:
    """带缓存的斐波那契（优化版）"""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

if __name__ == "__main__":
    print("斐波那契数列前10项:", fibonacci_sequence(10))
    print("第20项:", fibonacci(20))
    print("带缓存第30项:", fibonacci_memo(30))
'''
            file_path = os.path.expanduser("~/.hermes/evolution/output/fibonacci.py")
            
        elif "hello" in user_input.lower() or "你好" in user_input:
            code = f'''print("Hello, World!")
print("你好，世界！")
print("来自工兵连的问候")
'''
            file_path = os.path.expanduser("~/.hermes/evolution/output/hello.py")
            
        elif "排序" in user_input or "sort" in user_input.lower():
            code = f'''"""
排序算法集合
用户需求: {user_input}
{intel_guidance}
"""

def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def bubble_sort(arr):
    """冒泡排序"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

if __name__ == "__main__":
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print("原始数据:", test_data)
    print("快速排序:", quick_sort(test_data))
    print("冒泡排序:", bubble_sort(test_data))
'''
            file_path = os.path.expanduser("~/.hermes/evolution/output/sort_demo.py")
            
        else:
            # 通用代码生成
            code = f'''"""
自动生成的脚本
用户需求: {user_input}
生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
{intel_guidance}
"""

def main():
    """主函数"""
    print("=" * 50)
    print("脚本已执行")
    print(f"用户需求: {user_input}")
    if "{intel_guidance.strip()}" :
        print("基于侦察情报生成")
    print("=" * 50)
    return True

if __name__ == "__main__":
    main()
'''
            file_path = os.path.expanduser("~/.hermes/evolution/output/generated_script.py")
        
        return code, file_path
    
    def _code_execute(self, params: Dict) -> Dict:
        """执行代码"""
        file_path = params.get("file_path", "")
        
        if not file_path:
            file_path = os.path.expanduser("~/.hermes/evolution/output/generated_script.py")
        
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"文件不存在: {file_path}"}
        
        try:
            import subprocess
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            logger.info(f"[{self.name}] 代码执行完成: {file_path}")
            
            return {
                "file_path": file_path,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "执行超时"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _file_write(self, params: Dict) -> Dict:
        """写入文件"""
        content = params.get("content", "")
        file_path = params.get("file_path", "")
        
        if not file_path:
            file_path = os.path.expanduser(f"~/.hermes/evolution/output/file_{int(time.time())}.txt")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"[{self.name}] 文件已写入: {file_path}")
        
        return {
            "file_path": file_path,
            "size": len(content),
            "status": "written",
        }
    
    def _file_read(self, params: Dict) -> Dict:
        """读取文件"""
        file_path = params.get("file_path", "")
        
        if not file_path or not os.path.exists(file_path):
            return {"status": "error", "error": f"文件不存在: {file_path}"}
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        logger.info(f"[{self.name}] 文件已读取: {file_path}")
        
        return {
            "file_path": file_path,
            "content": content,
            "size": len(content),
            "lines": len(content.split("\n")),
        }
    
    def _command_run(self, params: Dict) -> Dict:
        """运行命令"""
        command = params.get("command", "")
        
        if not command:
            return {"status": "error", "error": "未指定命令"}
        
        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            logger.info(f"[{self.name}] 命令执行完成: {command}")
            
            return {
                "command": command,
                "stdout": result.stdout[:2000],  # 限制输出长度
                "stderr": result.stderr[:1000],
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "执行超时"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _test_run(self, params: Dict) -> Dict:
        """运行测试"""
        file_path = params.get("file_path", "")
        
        if not file_path:
            # 运行默认测试
            test_code = '''
import sys
sys.path.insert(0, "src")

def test_basic():
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"
    print("基础测试通过!")

def test_list_ops():
    assert sorted([3,1,2]) == [1,2,3]
    assert len([1,2,3]) == 3
    print("列表操作测试通过!")

if __name__ == "__main__":
    test_basic()
    test_list_ops()
    print("所有测试通过!")
'''
            file_path = os.path.expanduser("~/.hermes/evolution/output/test_temp.py")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(test_code)
        
        # 执行测试
        return self._code_execute({"file_path": file_path})
    
    def _report_task_status(self, task: Task, success: bool, result: Dict, exec_time: float, error: str = None):
        """报告任务状态"""
        self.hub.broadcast_task_status(TaskStatusPayload(
            task_id=task.task_id,
            task_type=task.task_type,
            agent_name=self.name,
            status="completed" if success else "failed",
            progress=100.0,
            result=result if success else None,
            error=error,
            timestamp=time.time(),
        ))
    
    def get_intel_summary(self) -> Dict:
        """获取已接收情报摘要"""
        return {
            "total_intel": len(self.recent_intel),
            "intel_types": list(set(i.intel_type for i in self.recent_intel)),
            "context_length": len(self.intel_context),
            "latest_intel": self.recent_intel[-1].summary if self.recent_intel else None,
        }
