"""
通信连 (Comms Agent)
负责：消息传递、记忆存储、知识检索、经验记录

基于工程控制论第6章（信号传递）设计
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from core.base_agent import BaseAgent
from core.task import Task, TaskResult, TaskStatus
from core.message_bus import get_message_bus

logger = logging.getLogger(__name__)


class CommsAgent(BaseAgent):
    """
    通信连 - 知识与记忆管理
    
    职责：
    1. 记忆存储：记录任务经验
    2. 知识检索：查询历史经验
    3. 经验总结：提炼知识规律
    4. 消息中转：Agent间消息路由
    """
    
    def __init__(self):
        super().__init__(agent_id="comms_01", name="通信连")
        self.capabilities = ["memory_store", "memory_retrieve", "knowledge_query", 
                            "experience_summary", "message_relay"]
        
        # 记忆存储路径
        self.storage_path = Path.home() / ".hermes/evolution/data/comms"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的短期记忆
        self.short_term_memory: List[Dict] = []
        self.max_short_term = 50
        
        # 知识库
        self.knowledge_base: Dict[str, Any] = {}
        self._load_knowledge()
        
        logger.info(f"[{self.name}] 初始化完成，存储路径: {self.storage_path}")
    
    def can_handle(self, task: Task) -> bool:
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行通信相关任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "memory_store":
                result = self._store_memory(task.parameters)
            elif task.task_type == "memory_retrieve":
                result = self._retrieve_memory(task.parameters)
            elif task.task_type == "knowledge_query":
                result = self._query_knowledge(task.parameters)
            elif task.task_type == "experience_summary":
                result = self._summarize_experience(task.parameters)
            elif task.task_type == "message_relay":
                result = self._relay_message(task.parameters)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    result={},
                    error=f"未知任务类型: {task.task_type}",
                    execution_time=time.time() - start_time,
                )
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result={},
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _store_memory(self, params: Dict) -> Dict:
        """存储记忆"""
        memory_entry = {
            "content": params.get("content", ""),
            "category": params.get("category", "general"),
            "importance": params.get("importance", 0.5),
            "timestamp": time.time(),
            "source": params.get("source", "unknown"),
        }
        
        # 短期记忆
        self.short_term_memory.append(memory_entry)
        if len(self.short_term_memory) > self.max_short_term:
            self.short_term_memory = self.short_term_memory[-self.max_short_term:]
        
        # 长期存储（JSON文件）
        self._save_memory_entry(memory_entry)
        
        logger.info(f"[{self.name}] 记忆已存储: {memory_entry['content'][:50]}...")
        return {"status": "stored", "entry": memory_entry}
    
    def _retrieve_memory(self, params: Dict) -> Dict:
        """检索记忆"""
        query = params.get("query", "")
        category = params.get("category")
        limit = params.get("limit", 5)
        
        # 从短期记忆中搜索
        results = []
        for entry in reversed(self.short_term_memory):
            if query.lower() in entry["content"].lower():
                if category is None or entry["category"] == category:
                    results.append(entry)
            if len(results) >= limit:
                break
        
        # 如果短期记忆不够，从文件中搜索
        if len(results) < limit:
            file_results = self._search_memory_file(query, limit - len(results))
            results.extend(file_results)
        
        logger.info(f"[{self.name}] 检索到 {len(results)} 条记忆")
        return {"query": query, "results": results, "count": len(results)}
    
    def _query_knowledge(self, params: Dict) -> Dict:
        """查询知识库"""
        topic = params.get("topic", "")
        
        # 简单关键词匹配
        results = {}
        for key, value in self.knowledge_base.items():
            if topic.lower() in key.lower():
                results[key] = value
        
        logger.info(f"[{self.name}] 知识查询: {topic}, 找到 {len(results)} 条")
        return {"topic": topic, "results": results, "count": len(results)}
    
    def _summarize_experience(self, params: Dict) -> Dict:
        """总结经验"""
        task_type = params.get("task_type", "")
        
        # 从记忆中提取相关经验
        related = []
        for entry in self.short_term_memory:
            if task_type.lower() in entry.get("category", "").lower():
                related.append(entry)
        
        # 简单统计
        if related:
            avg_importance = sum(e.get("importance", 0.5) for e in related) / len(related)
            summary = {
                "task_type": task_type,
                "experience_count": len(related),
                "avg_importance": avg_importance,
                "latest": related[-1] if related else None,
            }
        else:
            summary = {
                "task_type": task_type,
                "experience_count": 0,
                "avg_importance": 0,
                "latest": None,
            }
        
        return summary
    
    def _relay_message(self, params: Dict) -> Dict:
        """中转消息"""
        from_agent = params.get("from", "unknown")
        to_agent = params.get("to", "unknown")
        content = params.get("content", "")
        
        # 通过消息总线发送
        bus = get_message_bus()
        msg_id = bus.publish(
            message_type="agent_message",
            data={"from": from_agent, "to": to_agent, "content": content},
            source=from_agent,
        )
        
        logger.info(f"[{self.name}] 消息中转: {from_agent} -> {to_agent}")
        return {"message_id": msg_id, "status": "relayed"}
    
    def _save_memory_entry(self, entry: Dict):
        """保存记忆到文件"""
        memory_file = self.storage_path / "memories.jsonl"
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def _search_memory_file(self, query: str, limit: int) -> List[Dict]:
        """从文件中搜索记忆"""
        memory_file = self.storage_path / "memories.jsonl"
        if not memory_file.exists():
            return []
        
        results = []
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if query.lower() in entry.get("content", "").lower():
                        results.append(entry)
                    if len(results) >= limit:
                        break
        except Exception as e:
            logger.error(f"[{self.name}] 搜索记忆文件失败: {e}")
        
        return results
    
    def _load_knowledge(self):
        """加载知识库"""
        kb_file = self.storage_path / "knowledge.json"
        if kb_file.exists():
            try:
                with open(kb_file, "r", encoding="utf-8") as f:
                    self.knowledge_base = json.load(f)
            except Exception as e:
                logger.warning(f"[{self.name}] 加载知识库失败: {e}")
    
    def save_knowledge(self):
        """保存知识库"""
        kb_file = self.storage_path / "knowledge.json"
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
    
    def add_knowledge(self, topic: str, content: Any):
        """添加知识"""
        self.knowledge_base[topic] = {
            "content": content,
            "timestamp": time.time(),
        }
        self.save_knowledge()
