"""
司令部 (Commander Agent)
统一指挥、用户接口、最终决策

基于工程控制论第17章（自适应系统）设计
"""

import time
import logging
from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.task import Task, TaskResult, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class CommanderAgent(BaseAgent):
    """
    司令部 - 统一指挥中心
    
    职责：
    1. 接收用户指令
    2. 下达给参谋部分析
    3. 审批执行方案
    4. 监控执行进度
    5. 向用户汇报结果
    """
    
    def __init__(self, staff_agent=None, scout_agent=None, 
                 builder_agent=None, comms_agent=None):
        super().__init__(agent_id="commander_01", name="司令部")
        self.capabilities = ["command", "orchestrate", "report"]
        
        # 下属部队
        self.staff = staff_agent
        self.scout = scout_agent
        self.builder = builder_agent
        self.comms = comms_agent
        
        # 任务队列
        self.task_queue: List[Dict] = []
        self.completed_tasks: List[Dict] = []
        
        logger.info(f"[{self.name}] 初始化完成，下属部队就绪")
    
    def can_handle(self, task: Task) -> bool:
        return task.task_type in self.capabilities
    
    def execute(self, task: Task) -> TaskResult:
        """执行指挥任务"""
        start_time = time.time()
        
        try:
            if task.task_type == "command":
                result = self._handle_command(task.parameters)
            elif task.task_type == "orchestrate":
                result = self._orchestrate(task.parameters)
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
    
    def process_user_input(self, user_input: str) -> Dict:
        """
        处理用户输入 - 这是主要入口
        
        完整流程：
        用户 → 司令部 → 参谋部分析 → 执行连执行 → 参谋部评估 → 司令部汇报
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.name}] 收到用户指令: {user_input}")
        logger.info(f"{'='*60}")
        
        overall_start = time.time()
        execution_log = []
        
        try:
            # Step 1: 下达给参谋部分析
            logger.info(f"\n[{self.name}] Step 1: 参谋部分析...")
            analysis_task = Task(
                task_type="task_analyze",
                description="分析用户指令",
                parameters={"input": user_input},
            )
            analysis_result = self.staff.execute(analysis_task)
            execution_log.append({"step": "analysis", "result": analysis_result.to_dict()})
            
            if not analysis_result.success:
                return self._build_report(False, "参谋部分析失败", execution_log, overall_start)
            
            analysis = analysis_result.result
            logger.info(f"[{self.name}] 分析结果: {analysis.get('primary_type')}")
            
            # Step 2: 参谋部制定方案
            logger.info(f"\n[{self.name}] Step 2: 制定执行方案...")
            plan_task = Task(
                task_type="plan_make",
                description="制定执行方案",
                parameters={"analysis": analysis},
            )
            plan_result = self.staff.execute(plan_task)
            execution_log.append({"step": "planning", "result": plan_result.to_dict()})
            
            if not plan_result.success:
                return self._build_report(False, "方案制定失败", execution_log, overall_start)
            
            plan = plan_result.result
            logger.info(f"[{self.name}] 方案: {plan.get('total_steps')} 步")
            
            # Step 3: 执行方案
            logger.info(f"\n[{self.name}] Step 3: 执行方案...")
            step_results = []
            
            for step in plan.get("steps", []):
                agent_name = step.get("agent")
                task_type = step.get("task_type")
                description = step.get("description")
                
                logger.info(f"\n[{self.name}] 执行步骤 {step['step']}: {description} [{agent_name}]")
                
                # 选择执行连
                agent = self._get_agent(agent_name)
                if agent is None:
                    logger.error(f"[{self.name}] 未找到执行连: {agent_name}")
                    step_results.append({
                        "step": step["step"],
                        "success": False,
                        "error": f"未找到执行连: {agent_name}",
                    })
                    continue
                
                # 构建执行参数
                exec_params = self._build_exec_params(task_type, user_input, analysis)
                
                exec_task = Task(
                    task_type=task_type,
                    description=description,
                    parameters=exec_params,
                )
                
                exec_result = agent.execute(exec_task)
                step_results.append({
                    "step": step["step"],
                    "agent": agent_name,
                    "task_type": task_type,
                    "success": exec_result.success,
                    "data": exec_result.result,
                    "error": exec_result.error,
                    "execution_time": exec_result.execution_time,
                })
                
                logger.info(f"[{self.name}] 步骤 {step['step']} 完成: "
                          f"{'✅' if exec_result.success else '❌'}")
            
            # Step 4: 参谋部评估质量
            logger.info(f"\n[{self.name}] Step 4: 质量评估...")
            eval_task = Task(
                task_type="quality_evaluate",
                description="评估执行质量",
                parameters={"results": step_results},
            )
            eval_result = self.staff.execute(eval_task)
            execution_log.append({"step": "evaluation", "result": eval_result.to_dict()})
            
            evaluation = eval_result.result if eval_result.success else {"score": 0, "verdict": "eval_failed"}
            
            # Step 5: 记录经验
            logger.info(f"\n[{self.name}] Step 5: 记录经验...")
            if self.comms:
                memory_task = Task(
                    task_type="memory_store",
                    description="记录执行经验",
                    parameters={
                        "content": f"用户指令: {user_input}\n分析: {analysis.get('primary_type')}\n评分: {evaluation.get('score', 0)}",
                        "category": "task_execution",
                        "importance": evaluation.get("score", 5) / 10,
                        "source": "commander",
                    },
                )
                self.comms.execute(memory_task)
            
            # Step 6: 生成汇报
            logger.info(f"\n[{self.name}] Step 6: 生成汇报...")
            return self._build_report(
                success=True,
                user_input=user_input,
                analysis=analysis,
                plan=plan,
                step_results=step_results,
                evaluation=evaluation,
                execution_log=execution_log,
                start_time=overall_start,
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 指挥失败: {e}")
            return self._build_report(False, str(e), execution_log, overall_start)
    
    def _get_agent(self, agent_name: str):
        """获取执行连"""
        agents = {
            "staff": self.staff,
            "scout": self.scout,
            "builder": self.builder,
            "comms": self.comms,
        }
        return agents.get(agent_name)
    
    def _build_exec_params(self, task_type: str, user_input: str, analysis: Dict) -> Dict:
        """构建执行参数"""
        if task_type == "web_search":
            # 从用户输入中提取搜索关键词
            query = user_input
            for prefix in ["搜索", "查找", "搜", "帮我找", "帮我搜"]:
                if prefix in query:
                    query = query.replace(prefix, "").strip()
            return {"query": query, "max_results": 5}
        
        elif task_type in ["code_write", "code_execute"]:
            return {"input": user_input, "analysis": analysis}
        
        elif task_type == "file_write":
            return {"input": user_input, "analysis": analysis}
        
        elif task_type == "file_read":
            return {"input": user_input}
        
        elif task_type == "memory_store":
            return {"content": user_input, "category": "user_request"}
        
        else:
            return {"input": user_input, "analysis": analysis}
    
    def _build_report(self, success: bool, user_input: str = "", 
                     analysis: Dict = None, plan: Dict = None,
                     step_results: List = None, evaluation: Dict = None,
                     execution_log: List = None, start_time: float = 0,
                     error_msg: str = "") -> Dict:
        """生成汇报"""
        total_time = time.time() - start_time if start_time else 0
        
        if success:
            # 提取关键结果
            results_summary = []
            for sr in (step_results or []):
                agent = sr.get("agent", "?")
                task_type = sr.get("task_type", "?")
                status = "✅" if sr.get("success") else "❌"
                results_summary.append(f"{status} [{agent}] {task_type}")
            
            report = {
                "success": True,
                "user_input": user_input,
                "task_type": analysis.get("primary_type", "unknown") if analysis else "unknown",
                "total_steps": plan.get("total_steps", 0) if plan else 0,
                "step_results": results_summary,
                "quality_score": evaluation.get("score", 0) if evaluation else 0,
                "quality_verdict": evaluation.get("verdict", "unknown") if evaluation else "unknown",
                "total_time": total_time,
                "suggestions": evaluation.get("suggestions", []) if evaluation else [],
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.name}] 任务完成!")
            logger.info(f"  类型: {report['task_type']}")
            logger.info(f"  步骤: {report['total_steps']}")
            logger.info(f"  质量: {report['quality_score']:.1f}/10")
            logger.info(f"  耗时: {total_time:.2f}秒")
            logger.info(f"{'='*60}")
            
            return report
        else:
            return {
                "success": False,
                "error": error_msg or user_input,
                "total_time": total_time,
            }
    
    def _handle_command(self, params: Dict) -> Dict:
        """处理命令"""
        return {"status": "command_received", "params": params}
    
    def _orchestrate(self, params: Dict) -> Dict:
        """编排任务"""
        return {"status": "orchestrated", "params": params}
