"""
参谋部 (Staff Agent)
负责：任务分析、方案规划、质量评估

基于工程控制论第10章（继电器伺服系统）和第15章（自寻最优控制）设计
"""

import time
import logging
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.task import Task, TaskResult, TaskPriority

logger = logging.getLogger(__name__)


class StaffAgent(BaseAgent):
    """
    参谋部 - 分析、规划、评估
    
    职责：
    1. 任务分析：理解用户意图，分类任务类型
    2. 方案规划：制定执行方案，选择执行连
    3. 质量评估：评估执行结果，判断是否达标
    4. 改进建议：提出优化建议
    """
    
    def __init__(self):
        super().__init__(agent_id="staff_01", name="参谋部")
        self.capabilities = ["task_analyze", "plan_make", "quality_evaluate", 
                            "suggest_improve", "select_agent"]
        
        # 任务类型到执行连的映射
        self.agent_routing = {
            "web_search": "scout",
            "file_scout": "scout",
            "data_collect": "scout",
            "env_sense": "scout",
            "info_extract": "scout",
            "code_write": "builder",
            "code_execute": "builder",
            "file_write": "builder",
            "file_read": "builder",
            "command_run": "builder",
            "test_run": "builder",
            "memory_store": "comms",
            "memory_retrieve": "comms",
            "knowledge_query": "comms",
            "experience_summary": "comms",
        }
        
        # 任务类型关键词映射
        self.task_keywords = {
            "搜索": "web_search",
            "查找": "web_search",
            "搜": "web_search",
            "写代码": "code_write",
            "编写": "code_write",
            "代码": "code_write",
            "运行": "code_execute",
            "执行": "command_run",
            "测试": "test_run",
            "读取": "file_read",
            "读文件": "file_read",
            "写入": "file_write",
            "写文件": "file_write",
            "创建": "file_write",
            "查看": "file_scout",
            "列出": "file_scout",
            "目录": "file_scout",
            "记住": "memory_store",
            "记录": "memory_store",
            "回忆": "memory_retrieve",
            "查找记忆": "memory_retrieve",
            "知识": "knowledge_query",
        }
        
        logger.info(f"[{self.name}] 初始化完成")
    
    def can_handle(self, task: Task) -> bool:
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行参谋任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "task_analyze":
                result = self._analyze_task(task.parameters)
            elif task.task_type == "plan_make":
                result = self._make_plan(task.parameters)
            elif task.task_type == "quality_evaluate":
                result = self._evaluate_quality(task.parameters)
            elif task.task_type == "suggest_improve":
                result = self._suggest_improve(task.parameters)
            elif task.task_type == "select_agent":
                result = self._select_agent(task.parameters)
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
    
    def _analyze_task(self, params: Dict) -> Dict:
        """分析用户指令"""
        user_input = params.get("input", "")
        
        # 识别任务类型
        detected_types = []
        for keyword, task_type in self.task_keywords.items():
            if keyword in user_input:
                detected_types.append(task_type)
        
        # 去重
        detected_types = list(set(detected_types))
        
        # 如果没识别出来，用默认类型
        if not detected_types:
            detected_types = ["code_write"]  # 默认为编码任务
        
        # 判断优先级
        priority = TaskPriority.MEDIUM
        if any(word in user_input for word in ["紧急", "马上", "立刻", "立即"]):
            priority = TaskPriority.HIGH
        elif any(word in user_input for word in ["不急", "有空", "方便时"]):
            priority = TaskPriority.LOW
        
        analysis = {
            "original_input": user_input,
            "detected_types": detected_types,
            "primary_type": detected_types[0],
            "priority": priority,
            "requires_research": "search" in str(detected_types) or "搜索" in user_input,
            "requires_coding": "code" in str(detected_types) or "代码" in user_input or "写" in user_input,
            "requires_memory": "memory" in str(detected_types) or "记住" in user_input,
        }
        
        logger.info(f"[{self.name}] 任务分析: {analysis['primary_type']}")
        return analysis
    
    def _make_plan(self, params: Dict) -> Dict:
        """制定执行方案"""
        analysis = params.get("analysis", {})
        task_type = analysis.get("primary_type", "code_write")
        
        # 根据任务类型制定步骤
        steps = []
        
        if analysis.get("requires_research"):
            steps.append({
                "step": 1,
                "agent": "scout",
                "task_type": "web_search",
                "description": "搜索相关信息",
            })
        
        if analysis.get("requires_coding"):
            step_num = len(steps) + 1
            steps.append({
                "step": step_num,
                "agent": "builder",
                "task_type": "code_write",
                "description": "编写代码",
            })
            
            # 如果是编码任务，加测试步骤
            steps.append({
                "step": step_num + 1,
                "agent": "builder",
                "task_type": "test_run",
                "description": "运行测试验证",
            })
        
        if analysis.get("requires_memory"):
            steps.append({
                "step": len(steps) + 1,
                "agent": "comms",
                "task_type": "memory_store",
                "description": "记录执行经验",
            })
        
        # 如果没有明确步骤，添加默认执行步骤
        if not steps:
            steps.append({
                "step": 1,
                "agent": "builder",
                "task_type": task_type,
                "description": f"执行{task_type}任务",
            })
        
        plan = {
            "analysis": analysis,
            "steps": steps,
            "total_steps": len(steps),
            "estimated_complexity": "high" if len(steps) > 3 else "medium" if len(steps) > 1 else "low",
        }
        
        logger.info(f"[{self.name}] 制定方案: {len(steps)} 步")
        return plan
    
    def _evaluate_quality(self, params: Dict) -> Dict:
        """评估执行质量"""
        results = params.get("results", [])
        
        if not results:
            return {"score": 0, "verdict": "no_results", "suggestions": ["无执行结果"]}
        
        # 统计成功率
        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)
        success_rate = successful / total if total > 0 else 0
        
        # 评分
        score = success_rate * 10
        
        # 判断
        if score >= 8:
            verdict = "excellent"
        elif score >= 6:
            verdict = "good"
        elif score >= 4:
            verdict = "acceptable"
        else:
            verdict = "needs_improvement"
        
        # 建议
        suggestions = []
        if success_rate < 1.0:
            failed = [r for r in results if not r.get("success", False)]
            for f in failed:
                suggestions.append(f"步骤失败: {f.get('error', 'unknown')}")
        
        evaluation = {
            "score": score,
            "success_rate": success_rate,
            "verdict": verdict,
            "suggestions": suggestions,
            "total_steps": total,
            "successful_steps": successful,
        }
        
        logger.info(f"[{self.name}] 质量评估: {score:.1f}/10 ({verdict})")
        return evaluation
    
    def _suggest_improvements(self, params: Dict) -> Dict:
        """提出改进建议"""
        evaluation = params.get("evaluation", {})
        score = evaluation.get("score", 0)
        
        suggestions = []
        
        if score < 6:
            suggestions.append("执行质量较低，建议检查任务分解是否合理")
            suggestions.append("建议增加预处理步骤（如搜索相关资料）")
        
        if score < 8:
            suggestions.append("可以增加测试验证步骤提高质量")
            suggestions.append("建议记录执行经验，避免重复错误")
        
        return {"suggestions": suggestions, "priority": "high" if score < 5 else "medium"}
    
    def _select_agent(self, params: Dict) -> Dict:
        """选择合适的执行连"""
        task_type = params.get("task_type", "")
        
        agent = self.agent_routing.get(task_type, "builder")
        
        logger.info(f"[{self.name}] 选择执行连: {agent} (任务类型: {task_type})")
        return {"selected_agent": agent, "task_type": task_type}
