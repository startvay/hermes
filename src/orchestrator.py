"""
任务编排器 (Task Orchestrator)
==============================
负责任务分解、依赖管理、容错重试

核心功能:
1. 智能任务分解 — 根据任务类型自动拆分为子任务
2. 依赖调度 — 按依赖关系串行/并行执行
3. 容错重试 — Agent失败时自动切换备用Agent
4. 动态Agent选择 — 根据经验池为每个子任务选最佳Agent

基于工程控制论: 多回路协调控制(Ch.16) + 可靠性理论
"""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

try:
    from core.task import Task, TaskResult, TaskPriority
    from core.base_agent import BaseAgent
    from agents.multi_agent_evolution import AgentExperiencePool
    from agents.communication_hub import AgentCommunicationHub, get_communication_hub
except ImportError:
    from src.core.task import Task, TaskResult, TaskPriority
    from src.core.base_agent import BaseAgent
    from src.agents.multi_agent_evolution import AgentExperiencePool
    from src.agents.communication_hub import AgentCommunicationHub, get_communication_hub

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class SubTask:
    """子任务"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    backup_agents: List[str] = field(default_factory=list)
    max_retries: int = 2
    
    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    retry_count: int = 0
    execution_time: float = 0.0
    error_log: List[str] = field(default_factory=list)


@dataclass
class DecompositionPlan:
    """任务分解方案"""
    original_task: str
    subtasks: List[SubTask]
    parallel_groups: List[List[str]]  # 可并行执行的task_id组
    estimated_total_time: float


class TaskDecomposer:
    """任务分解器 — 将复杂任务拆分为可执行的子任务"""
    
    # 任务分解模板
    DECOMPOSITION_TEMPLATES = {
        "research_and_code": {
            "description": "研究+编码任务",
            "steps": [
                {"type": "web_search", "desc_prefix": "研究: ", "params_key": "research"},
                {"type": "code_write", "desc_prefix": "编码: ", "params_key": "code"},
                {"type": "test_run", "desc_prefix": "验证: ", "params_key": "test"},
            ],
        },
        "analyze_and_fix": {
            "description": "分析+修复任务",
            "steps": [
                {"type": "file_scout", "desc_prefix": "侦察: ", "params_key": "scout"},
                {"type": "code_write", "desc_prefix": "修复: ", "params_key": "fix"},
                {"type": "test_run", "desc_prefix": "验证: ", "params_key": "test"},
            ],
        },
        "data_pipeline": {
            "description": "数据流水线",
            "steps": [
                {"type": "data_collect", "desc_prefix": "采集: ", "params_key": "collect"},
                {"type": "info_extract", "desc_prefix": "提取: ", "params_key": "extract"},
                {"type": "file_write", "desc_prefix": "输出: ", "params_key": "output"},
            ],
        },
        "multi_step_command": {
            "description": "多步骤命令",
            "steps": [
                {"type": "command_run", "desc_prefix": "步骤1: ", "params_key": "step1"},
                {"type": "command_run", "desc_prefix": "步骤2: ", "params_key": "step2"},
                {"type": "command_run", "desc_prefix": "步骤3: ", "params_key": "step3"},
            ],
        },
    }
    
    def __init__(self, experience_pool: Optional[AgentExperiencePool] = None):
        self.experience_pool = experience_pool
    
    def decompose(self, user_input: str, parameters: Dict = None) -> DecompositionPlan:
        """
        智能分解用户请求
        
        分析用户输入，匹配分解模板，生成子任务计划
        """
        parameters = parameters or {}
        
        # 1. 关键词分析，确定任务类型
        task_category = self._analyze_task_category(user_input)
        
        # 2. 获取分解模板
        template = self.DECOMPOSITION_TEMPLATES.get(task_category)
        
        if template:
            subtasks = self._build_from_template(user_input, template, parameters)
        else:
            # 无法匹配模板，生成通用单任务
            subtasks = [self._create_single_subtask(user_input, parameters)]
        
        # 3. 计算并行组
        parallel_groups = self._compute_parallel_groups(subtasks)
        
        # 4. 估算总时间
        est_time = self._estimate_time(subtasks)
        
        plan = DecompositionPlan(
            original_task=user_input,
            subtasks=subtasks,
            parallel_groups=parallel_groups,
            estimated_total_time=est_time,
        )
        
        logger.info(f"[任务分解] '{user_input[:50]}...' → {len(subtasks)}个子任务, "
                   f"并行组: {len(parallel_groups)}")
        return plan
    
    def _analyze_task_category(self, user_input: str) -> str:
        """分析用户输入，确定任务类型"""
        text = user_input.lower()
        
        # 研究+编码
        if any(k in text for k in ["研究", "搜索", "调研", "查找"]) and \
           any(k in text for k in ["写", "代码", "程序", "实现", "生成"]):
            return "research_and_code"
        
        # 分析+修复
        if any(k in text for k in ["分析", "检查", "调试", "修复", "fix", "debug"]):
            return "analyze_and_fix"
        
        # 数据流水线
        if any(k in text for k in ["数据", "采集", "处理", "pipeline", "etl"]):
            return "data_pipeline"
        
        # 多步骤命令
        if any(k in text for k in ["多步", "流水线", "依次", "然后", "再"]) or \
           (";" in user_input and user_input.count(";") >= 2):
            return "multi_step_command"
        
        # 默认：尝试单任务处理
        return "single"
    
    def _build_from_template(self, user_input: str, template: Dict, params: Dict) -> List[SubTask]:
        """基于模板构建子任务"""
        subtasks = []
        prev_id = None
        
        for i, step in enumerate(template["steps"]):
            task_id = f"subtask_{uuid.uuid4().hex[:8]}"
            
            subtask = SubTask(
                task_id=task_id,
                task_type=step["type"],
                description=f"{step['desc_prefix']}{user_input[:40]}",
                parameters=params.get(step.get("params_key", ""), {}),
                dependencies=[prev_id] if prev_id else [],
                max_retries=2,
            )
            
            subtasks.append(subtask)
            prev_id = task_id
        
        return subtasks
    
    def _create_single_subtask(self, user_input: str, params: Dict) -> SubTask:
        """创建单个子任务"""
        # 根据关键词判断最合适的task_type
        task_type = self._infer_task_type(user_input)
        
        return SubTask(
            task_id=f"subtask_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=user_input[:80],
            parameters={"input": user_input, **params},
            max_retries=2,
        )
    
    def _infer_task_type(self, user_input: str) -> str:
        """从用户输入推断任务类型"""
        text = user_input.lower()
        
        if any(k in text for k in ["搜索", "查找", "search", "find"]):
            return "web_search"
        if any(k in text for k in ["写代码", "生成代码", "code", "program"]):
            return "code_write"
        if any(k in text for k in ["执行", "运行", "run", "execute"]):
            return "code_execute"
        if any(k in text for k in ["文件", "读取", "read", "write"]):
            return "file_read" if "read" in text else "file_write"
        if any(k in text for k in ["命令", "cmd", "command", "shell"]):
            return "command_run"
        if any(k in text for k in ["测试", "test"]):
            return "test_run"
        
        return "code_write"  # 默认
    
    def _compute_parallel_groups(self, subtasks: List[SubTask]) -> List[List[str]]:
        """计算可并行执行的task_id组"""
        groups = []
        completed: Set[str] = set()
        remaining = {st.task_id for st in subtasks}
        
        while remaining:
            # 找当前没有未满足依赖的任务
            runnable = []
            for st in subtasks:
                if st.task_id in remaining:
                    deps_satisfied = all(d in completed for d in st.dependencies)
                    if deps_satisfied:
                        runnable.append(st.task_id)
            
            if not runnable:
                # 有循环依赖，break
                break
            
            groups.append(runnable)
            completed.update(runnable)
            remaining.difference_update(runnable)
        
        return groups
    
    def _estimate_time(self, subtasks: List[SubTask]) -> float:
        """估算总执行时间"""
        if not self.experience_pool:
            return len(subtasks) * 2.0
        
        total = 0.0
        for st in subtasks:
            # 查经验池中的平均执行时间
            key = f"{st.assigned_agent or 'unknown'}:{st.task_type}"
            stat = self.experience_pool.agent_stats.get(key, {})
            avg_time = stat.get("avg_time", 2.0)
            total += avg_time
        
        return total


class FaultTolerantExecutor:
    """容错执行器 — 带重试和Agent切换的执行引擎"""
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        experience_pool: Optional[AgentExperiencePool] = None,
        hub: Optional[AgentCommunicationHub] = None,
    ):
        self.agents = agents
        self.experience_pool = experience_pool
        self.hub = hub
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 任务类型到Agent的默认映射
        self.default_agent_map = {
            "web_search": "scout",
            "file_scout": "scout",
            "env_sense": "scout",
            "data_collect": "scout",
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
        }
        
        logger.info(f"[容错执行器] 初始化完成，可用Agent: {list(agents.keys())}")
    
    def execute_plan(self, plan: DecompositionPlan) -> Dict:
        """
        执行分解计划
        
        按并行组调度，组内并行，组间串行
        """
        logger.info(f"[容错执行器] 开始执行计划: {len(plan.subtasks)} 个子任务")
        
        results = {}
        subtask_map = {st.task_id: st for st in plan.subtasks}
        
        for group_idx, group in enumerate(plan.parallel_groups):
            logger.info(f"[容错执行器] 执行并行组 {group_idx+1}/{len(plan.parallel_groups)}: {group}")
            
            # 并行执行组内任务
            futures = {}
            for task_id in group:
                subtask = subtask_map[task_id]
                future = self.executor.submit(self._execute_with_retry, subtask)
                futures[future] = task_id
            
            # 收集结果
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result(timeout=60)
                    results[task_id] = result
                    subtask_map[task_id].result = result
                    subtask_map[task_id].status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                except Exception as e:
                    logger.error(f"[容错执行器] 任务 {task_id} 执行异常: {e}")
                    subtask_map[task_id].status = TaskStatus.FAILED
                    subtask_map[task_id].error_log.append(str(e))
        
        # 汇总
        summary = self._summarize_results(plan, subtask_map)
        logger.info(f"[容错执行器] 计划执行完成: {summary['success_count']}/{summary['total']} 成功")
        
        return summary
    
    def _execute_with_retry(self, subtask: SubTask) -> TaskResult:
        """执行任务，支持重试和Agent切换"""
        
        # 1. 选择最佳Agent
        primary_agent = self._select_agent(subtask)
        backup_agents = self._get_backup_agents(subtask, primary_agent)
        
        subtask.assigned_agent = primary_agent
        subtask.backup_agents = backup_agents
        
        last_error = None
        
        # 2. 尝试执行（主Agent + 备用Agent）
        for attempt in range(subtask.max_retries + 1):
            agent_name = primary_agent if attempt == 0 else (backup_agents[attempt - 1] if attempt - 1 < len(backup_agents) else primary_agent)
            
            agent = self.agents.get(agent_name)
            if not agent:
                last_error = f"Agent {agent_name} 不存在"
                continue
            
            subtask.status = TaskStatus.RUNNING
            subtask.retry_count = attempt
            
            start_time = time.time()
            logger.info(f"[容错执行器] 执行任务 {subtask.task_id} ({subtask.task_type}) "
                       f"→ Agent: {agent_name} (尝试 {attempt + 1}/{subtask.max_retries + 1})")
            
            try:
                # 构建Task对象
                task = Task(
                    task_type=subtask.task_type,
                    description=subtask.description,
                    parameters=subtask.parameters,
                    priority=subtask.priority,
                )
                
                result = agent.execute(task)
                subtask.execution_time = time.time() - start_time
                
                if result.success:
                    logger.info(f"[容错执行器] 任务 {subtask.task_id} 成功 "
                               f"(Agent: {agent_name}, 耗时: {subtask.execution_time:.2f}s)")
                    
                    # 记录成功经验
                    self._record_experience(subtask, agent_name, True, result)
                    return result
                else:
                    last_error = result.error
                    subtask.error_log.append(f"[{agent_name}] {result.error}")
                    logger.warning(f"[容错执行器] 任务 {subtask.task_id} 失败 "
                                  f"(Agent: {agent_name}): {result.error}")
                    
                    # 记录失败经验
                    self._record_experience(subtask, agent_name, False, result)
                    
                    # 继续重试
                    subtask.status = TaskStatus.RETRYING
                    if attempt < subtask.max_retries:
                        logger.info(f"[容错执行器] 准备重试，切换到备用Agent...")
                    
            except Exception as e:
                last_error = str(e)
                subtask.error_log.append(f"[{agent_name}] {e}")
                subtask.execution_time = time.time() - start_time
                logger.error(f"[容错执行器] 任务 {subtask.task_id} 异常: {e}")
        
        # 所有重试都失败
        logger.error(f"[容错执行器] 任务 {subtask.task_id} 全部重试失败")
        subtask.status = TaskStatus.FAILED
        
        return TaskResult(
            task_id=subtask.task_id,
            success=False,
            result={},
            error=f"全部重试失败: {last_error}",
            execution_time=subtask.execution_time,
        )
    
    def _select_agent(self, subtask: SubTask) -> str:
        """为子任务选择最佳Agent"""
        # 1. 查经验池
        if self.experience_pool:
            best, _ = self.experience_pool.get_best_agent_for_task(subtask.task_type)
            if best:
                return best
        
        # 2. 默认映射
        return self.default_agent_map.get(subtask.task_type, "builder")
    
    def _get_backup_agents(self, subtask: SubTask, primary: str) -> List[str]:
        """获取备用Agent列表"""
        task_type = subtask.task_type
        
        # 根据任务类型确定候选Agent
        candidates = []
        if task_type in ["web_search", "file_scout", "env_sense", "data_collect"]:
            candidates = ["scout", "builder", "comms"]
        elif task_type in ["code_write", "code_execute", "file_write", "file_read", "command_run", "test_run"]:
            candidates = ["builder", "scout", "comms"]
        elif task_type in ["memory_store", "memory_retrieve", "knowledge_query"]:
            candidates = ["comms", "builder", "scout"]
        else:
            candidates = ["builder", "scout", "comms"]
        
        # 排除主Agent，按经验池成功率排序
        candidates = [c for c in candidates if c != primary and c in self.agents]
        
        if self.experience_pool:
            scored = []
            for c in candidates:
                rate = self.experience_pool.get_agent_success_rate(c, task_type)
                scored.append((c, rate))
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in scored]
        
        return candidates[:2]  # 最多2个备用
    
    def _record_experience(self, subtask: SubTask, agent_name: str, success: bool, result: TaskResult):
        """记录执行经验"""
        if not self.experience_pool:
            return
        
        from agents.multi_agent_evolution import AgentExecutionRecord
        
        self.experience_pool.record(AgentExecutionRecord(
            agent_name=agent_name,
            task_type=subtask.task_type,
            parameters=subtask.parameters,
            success=success,
            execution_time=result.execution_time if result else subtask.execution_time,
            score=8.0 if success else 2.0,
            error=result.error if result and not success else None,
        ))
    
    def _summarize_results(self, plan: DecompositionPlan, subtask_map: Dict[str, SubTask]) -> Dict:
        """汇总执行结果"""
        total = len(plan.subtasks)
        success_count = sum(1 for st in plan.subtasks if st.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for st in plan.subtasks if st.status == TaskStatus.FAILED)
        total_time = sum(st.execution_time for st in plan.subtasks)
        
        return {
            "total": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": success_count / total if total > 0 else 0,
            "total_time": total_time,
            "subtask_results": [
                {
                    "task_id": st.task_id,
                    "task_type": st.task_type,
                    "status": st.status.value,
                    "agent": st.assigned_agent,
                    "retries": st.retry_count,
                    "time": st.execution_time,
                    "error": st.error_log[-1] if st.error_log else None,
                }
                for st in plan.subtasks
            ],
            "original_task": plan.original_task,
        }
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("[容错执行器] 已关闭")


class Orchestrator:
    """
    编排器 — 整合分解器+执行器的统一接口
    """
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        experience_pool: Optional[AgentExperiencePool] = None,
    ):
        self.decomposer = TaskDecomposer(experience_pool)
        self.executor = FaultTolerantExecutor(agents, experience_pool)
        
        logger.info("[编排器] 初始化完成")
    
    def execute(self, user_input: str, parameters: Dict = None) -> Dict:
        """
        一站式执行：分解→调度→容错执行→汇总
        """
        # 1. 分解
        plan = self.decomposer.decompose(user_input, parameters)
        
        # 2. 执行
        result = self.executor.execute_plan(plan)
        
        # 3. 附加分解元信息
        result["decomposition"] = {
            "subtask_count": len(plan.subtasks),
            "parallel_groups": len(plan.parallel_groups),
            "estimated_time": plan.estimated_total_time,
        }
        
        return result
    
    def get_stats(self) -> Dict:
        """获取编排器统计"""
        return {
            "agents": list(self.executor.agents.keys()),
            "has_experience_pool": self.experience_pool is not None,
        }
    
    def shutdown(self):
        """关闭"""
        self.executor.shutdown()


# 兼容属性
Orchestrator.experience_pool = property(lambda self: self.executor.experience_pool)
