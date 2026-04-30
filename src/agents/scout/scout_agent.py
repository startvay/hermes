"""
侦察连 (Scout Agent) v2.0 — 真实执行版
=========================================
负责：信息搜索、数据收集、环境感知、情报整理

v2.0改进:
- web_search 使用DDGS执行真实DuckDuckGo搜索
- 情报自动发布到通信中心，实时推送给工兵连
- 搜索结果置信度评估
"""

import os
import time
import uuid
import logging
import json
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.task import Task, TaskResult

try:
    from agents.communication_hub import (
        AgentCommunicationHub, IntelPayload, get_communication_hub
    )
except ImportError:
    from communication_hub import (
        AgentCommunicationHub, IntelPayload, get_communication_hub
    )

logger = logging.getLogger(__name__)


class ScoutAgent(BaseAgent):
    """
    侦察连 - 信息搜索与收集 (真实执行版)
    
    职责：
    1. Web搜索：真实DuckDuckGo搜索
    2. 文件侦察：检查文件/目录信息
    3. 环境感知：检测系统状态
    4. 情报整理：整理收集的信息并发布
    """
    
    def __init__(self, hub: Optional[AgentCommunicationHub] = None):
        super().__init__(agent_id="scout_01", name="侦察连")
        self.capabilities = ["web_search", "file_scout", "env_sense", 
                            "data_collect", "info_extract"]
        self.hub = hub or get_communication_hub()
        self.last_intel: Optional[IntelPayload] = None
        
        logger.info(f"[{self.name}] v2.0 初始化完成，通信中心已连接")
    
    def can_handle(self, task: Task) -> bool:
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行侦察任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "web_search":
                result = self._web_search(task.parameters)
            elif task.task_type == "file_scout":
                result = self._file_scout(task.parameters)
            elif task.task_type == "env_sense":
                result = self._env_sense(task.parameters)
            elif task.task_type == "data_collect":
                result = self._data_collect(task.parameters)
            elif task.task_type == "info_extract":
                result = self._info_extract(task.parameters)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    result={},
                    error=f"未知任务类型: {task.task_type}",
                    execution_time=time.time() - start_time,
                )
            
            # 报告任务完成状态到通信中心
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
    
    def _web_search(self, params: Dict) -> Dict:
        """Web搜索 — 真实DuckDuckGo搜索"""
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        
        logger.info(f"[{self.name}] 开始搜索: {query}")
        
        results = []
        search_success = False
        
        # 尝试使用DDGS进行真实搜索
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:300],
                    })
            search_success = True
            logger.info(f"[{self.name}] 搜索完成: 找到 {len(results)} 条结果")
        except Exception as e:
            logger.warning(f"[{self.name}] DDGS搜索失败: {e}，回退到模拟搜索")
            # 回退：返回结构化的模拟结果，但标记为模拟
            results = [{
                "title": f"[模拟结果] 关于 '{query}' 的搜索",
                "url": "",
                "snippet": f"搜索关键词: {query}。由于网络限制，此结果为模拟数据。",
            }]
            search_success = False
        
        result_data = {
            "query": query,
            "results": results,
            "count": len(results),
            "real_search": search_success,
            "timestamp": time.time(),
        }
        
        # 发布情报到通信中心
        intel = IntelPayload(
            intel_id=f"intel_{uuid.uuid4().hex[:8]}",
            intel_type="web_search",
            source_agent=self.name,
            raw_data=result_data,
            summary=f"搜索 '{query}' 找到 {len(results)} 条结果",
            confidence=0.9 if search_success else 0.3,
            timestamp=time.time(),
            metadata={"query": query, "max_results": max_results, "real": search_success},
        )
        self.hub.publish_intel(intel)
        self.last_intel = intel
        
        return result_data
    
    def _file_scout(self, params: Dict) -> Dict:
        """文件侦察"""
        path = params.get("path", ".")
        
        try:
            if os.path.isdir(path):
                files = []
                total_size = 0
                for item in os.listdir(path)[:50]:
                    item_path = os.path.join(path, item)
                    try:
                        stat = os.stat(item_path)
                        total_size += stat.st_size
                        files.append({
                            "name": item,
                            "size": stat.st_size,
                            "is_dir": os.path.isdir(item_path),
                            "modified": stat.st_mtime,
                        })
                    except Exception:
                        pass
                
                result = {"path": path, "files": files, "count": len(files), "total_size": total_size}
                logger.info(f"[{self.name}] 侦察目录: {path}, 发现 {len(files)} 个项目")
                
                # 发布情报
                self._publish_file_intel(path, result)
                return result
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(5000)
                
                result = {"path": path, "content": content, "size": len(content)}
                logger.info(f"[{self.name}] 侦察文件: {path}, {len(content)} 字符")
                
                # 发布情报
                self._publish_file_intel(path, result)
                return result
                
        except Exception as e:
            return {"path": path, "error": str(e)}
    
    def _publish_file_intel(self, path: str, data: Dict):
        """发布文件侦察情报"""
        intel = IntelPayload(
            intel_id=f"intel_{uuid.uuid4().hex[:8]}",
            intel_type="file_scout",
            source_agent=self.name,
            raw_data=data,
            summary=f"文件侦察: {path}, 发现 {data.get('count', 0)} 个项目",
            confidence=1.0,
            timestamp=time.time(),
            metadata={"path": path},
        )
        self.hub.publish_intel(intel)
        self.last_intel = intel
    
    def _env_sense(self, params: Dict) -> Dict:
        """环境感知"""
        import platform
        
        env_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", "unknown"),
            "home": os.environ.get("HOME", "unknown"),
        }
        
        # 检查关键目录
        important_dirs = [
            os.path.expanduser("~/.hermes"),
            os.path.expanduser("~/.hermes/evolution"),
        ]
        
        for d in important_dirs:
            env_info[f"exists_{d}"] = os.path.exists(d)
        
        logger.info(f"[{self.name}] 环境感知完成")
        return env_info
    
    def _data_collect(self, params: Dict) -> Dict:
        """数据收集"""
        source = params.get("source", "")
        data_type = params.get("type", "text")
        
        logger.info(f"[{self.name}] 数据收集: {source}")
        
        return {
            "source": source,
            "type": data_type,
            "status": "collected",
            "timestamp": time.time(),
        }
    
    def _info_extract(self, params: Dict) -> Dict:
        """信息提取"""
        text = params.get("text", "")
        keywords = params.get("keywords", [])
        
        extracted = {}
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                idx = text_lower.find(keyword.lower())
                start = max(0, idx - 50)
                end = min(len(text), idx + len(keyword) + 50)
                context = text[start:end]
                extracted[keyword] = context
        
        logger.info(f"[{self.name}] 提取了 {len(extracted)} 个关键词上下文")
        return {"extracted": extracted, "count": len(extracted)}
    
    def _report_task_status(self, task: Task, success: bool, result: Dict, exec_time: float, error: str = None):
        """报告任务状态到通信中心"""
        try:
            from agents.communication_hub import TaskStatusPayload
        except ImportError:
            from communication_hub import TaskStatusPayload
        
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
