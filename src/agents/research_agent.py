"""
研究Agent
负责信息检索、数据分析等任务
"""

import time
from typing import Any, Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.base_agent import BaseAgent
from core.task import Task, TaskResult


class ResearchAgent(BaseAgent):
    """
    研究Agent
    
    能力：
    - 网络搜索
    - 数据分析
    - 知识检索
    - 内容摘要
    """
    
    def __init__(self, agent_id: str = "researcher_001"):
        super().__init__(agent_id, "Research Agent")
        self.capabilities = [
            "web_search",
            "data_analysis",
            "knowledge_retrieval",
            "summarization",
            "text_analysis",
        ]
        self.search_history: List[Dict] = []
    
    def can_handle(self, task: Task) -> bool:
        """判断是否能处理任务"""
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "web_search":
                result = self._web_search(task.parameters)
            elif task.task_type == "data_analysis":
                result = self._analyze_data(task.parameters)
            elif task.task_type == "knowledge_retrieval":
                result = self._retrieve_knowledge(task.parameters)
            elif task.task_type == "summarization":
                result = self._summarize(task.parameters)
            elif task.task_type == "text_analysis":
                result = self._analyze_text(task.parameters)
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
                metrics={"result_count": len(result) if isinstance(result, (list, dict)) else 1},
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _web_search(self, params: Dict[str, Any]) -> List[Dict]:
        """
        网络搜索
        
        Args:
            params: {"query": "搜索关键词", "max_results": 10}
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        query = params.get("query", "")
        max_results = params.get("max_results", 10)
        
        if not query:
            raise ValueError("搜索关键词不能为空")
        
        # 记录搜索历史
        self.search_history.append({
            "query": query,
            "timestamp": time.time(),
        })
        
        # 这里应该调用实际的搜索API
        # 目前返回模拟结果
        return [
            {
                "title": f"搜索结果 {i+1}: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"这是关于'{query}'的搜索结果摘要...",
            }
            for i in range(min(max_results, 3))
        ]
    
    def _analyze_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        数据分析
        
        Args:
            params: {"data": [...], "analysis_type": "statistical"}
            
        Returns:
            Dict: 分析结果
        """
        data = params.get("data", [])
        analysis_type = params.get("analysis_type", "basic")
        
        if not data:
            raise ValueError("数据不能为空")
        
        # 基础统计分析
        if analysis_type == "statistical" or analysis_type == "basic":
            if isinstance(data[0], (int, float)):
                return {
                    "count": len(data),
                    "sum": sum(data),
                    "mean": sum(data) / len(data),
                    "min": min(data),
                    "max": max(data),
                    "range": max(data) - min(data),
                }
        
        return {
            "count": len(data),
            "type": type(data[0]).__name__,
            "analysis_type": analysis_type,
        }
    
    def _retrieve_knowledge(self, params: Dict[str, Any]) -> List[Dict]:
        """
        知识检索
        
        Args:
            params: {"topic": "主题", "depth": "shallow"}
            
        Returns:
            List[Dict]: 知识条目
        """
        topic = params.get("topic", "")
        depth = params.get("depth", "shallow")
        
        if not topic:
            raise ValueError("主题不能为空")
        
        # 模拟知识检索
        return [
            {
                "topic": topic,
                "content": f"关于'{topic}'的知识条目...",
                "source": "knowledge_base",
                "confidence": 0.85,
            }
        ]
    
    def _summarize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        内容摘要
        
        Args:
            params: {"text": "长文本", "max_length": 200}
            
        Returns:
            Dict: 摘要结果
        """
        text = params.get("text", "")
        max_length = params.get("max_length", 200)
        
        if not text:
            raise ValueError("文本不能为空")
        
        # 简单截断摘要
        summary = text[:max_length]
        if len(text) > max_length:
            summary += "..."
        
        return {
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary,
            "compression_ratio": len(summary) / len(text) if text else 0,
        }
    
    def _analyze_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        文本分析
        
        Args:
            params: {"text": "文本内容"}
            
        Returns:
            Dict: 分析结果
        """
        text = params.get("text", "")
        
        if not text:
            raise ValueError("文本不能为空")
        
        words = text.split()
        lines = text.split("\n")
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        }
