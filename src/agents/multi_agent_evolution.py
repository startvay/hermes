"""
多Agent体系与进化系统打通层
============================
让进化系统成为军师旅的智慧大脑

打通点:
1. Agent经验共享池 — 所有Agent执行结果统一收集
2. 进化系统调度多Agent — 用级联决策选择最优Agent组合
3. 执行结果反馈学习 — Agent经验自动回流进化系统
4. 动态Agent路由 — 参谋部根据历史成功率选择Agent

基于工程控制论: 多回路协调控制(Ch.16) + 自适应系统(Ch.17)
"""

import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from ..core.learning_engine import LearningEngine, Experience
    from ..core.decision_engine import DecisionEngine, DecisionType, DecisionOption
    from ..core.performance import PerformanceMonitor
    from ..core.task import Task, TaskResult
    from .commander.commander_agent import CommanderAgent
    from .staff.staff_agent import StaffAgent
    from .scout.scout_agent import ScoutAgent
    from .builder.builder_agent import BuilderAgent
    from .comms.comms_agent import CommsAgent
    from .communication_hub import AgentCommunicationHub, get_communication_hub
except ImportError:
    from core.learning_engine import LearningEngine, Experience
    from core.decision_engine import DecisionEngine, DecisionType, DecisionOption
    from core.performance import PerformanceMonitor
    from core.task import Task, TaskResult
    from agents.commander.commander_agent import CommanderAgent
    from agents.staff.staff_agent import StaffAgent
    from agents.scout.scout_agent import ScoutAgent
    from agents.builder.builder_agent import BuilderAgent
    from agents.comms.comms_agent import CommsAgent
    from agents.communication_hub import AgentCommunicationHub, get_communication_hub

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionRecord:
    """Agent执行记录 — 经验共享池的基础单元"""
    agent_name: str           # scout/builder/comms/staff
    task_type: str            # web_search/code_write/...
    parameters: Dict[str, Any]
    success: bool
    execution_time: float
    score: float              # 质量评分 0-10
    error: Optional[str] = None
    timestamp: float = 0.0
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.context is None:
            self.context = {}


class AgentExperiencePool:
    """
    Agent经验共享池
    
    所有Agent的执行经验统一存入此处，供进化系统学习。
    相当于多Agent之间的"集体记忆"。
    """
    
    def __init__(self):
        self.records: List[AgentExecutionRecord] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        
    def record(self, record: AgentExecutionRecord):
        """记录一次Agent执行"""
        self.records.append(record)
        
        # 更新Agent统计
        key = f"{record.agent_name}:{record.task_type}"
        if key not in self.agent_stats:
            self.agent_stats[key] = {
                "total": 0, "success": 0, "avg_time": 0.0,
                "avg_score": 0.0, "history": []
            }
        
        stat = self.agent_stats[key]
        stat["total"] += 1
        if record.success:
            stat["success"] += 1
        stat["history"].append(record.score)
        stat["avg_score"] = sum(stat["history"][-10:]) / min(len(stat["history"]), 10)
        
        # 更新时间(EMA)
        stat["avg_time"] = 0.7 * stat["avg_time"] + 0.3 * record.execution_time if stat["avg_time"] > 0 else record.execution_time
        
    def get_agent_success_rate(self, agent_name: str, task_type: str = None) -> float:
        """获取Agent成功率"""
        if task_type:
            key = f"{agent_name}:{task_type}"
            stat = self.agent_stats.get(key, {})
            total = stat.get("total", 0)
            return stat.get("success", 0) / total if total > 0 else 0.5
        else:
            # 统计该Agent所有task_type
            totals = []
            successes = []
            for key, stat in self.agent_stats.items():
                if key.startswith(f"{agent_name}:"):
                    totals.append(stat.get("total", 0))
                    successes.append(stat.get("success", 0))
            total = sum(totals)
            return sum(successes) / total if total > 0 else 0.5
    
    def get_best_agent_for_task(self, task_type: str) -> Tuple[str, float]:
        """
        获取执行某任务的最佳Agent
        
        Returns:
            (agent_name, success_rate)
        """
        candidates = []
        for key, stat in self.agent_stats.items():
            if key.endswith(f":{task_type}"):
                agent_name = key.split(":")[0]
                rate = stat.get("success", 0) / stat.get("total", 1) if stat.get("total", 0) > 0 else 0.5
                avg_score = stat.get("avg_score", 0)
                # 综合评分 = 成功率*0.6 + 平均质量*0.4
                combined = rate * 0.6 + (avg_score / 10.0) * 0.4
                candidates.append((agent_name, combined, rate, avg_score))
        
        if not candidates:
            # 无历史数据，返回默认映射
            default_map = {
                "web_search": "scout", "file_scout": "scout", "env_sense": "scout",
                "code_write": "builder", "code_execute": "builder", "file_write": "builder",
                "file_read": "builder", "command_run": "builder", "test_run": "builder",
                "memory_store": "comms", "memory_retrieve": "comms", "knowledge_query": "comms",
            }
            return default_map.get(task_type, "builder"), 0.5
        
        # 按综合评分排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        return best[0], best[2]
    
    def to_experiences(self) -> List[Experience]:
        """将共享池记录转换为进化系统的Experience"""
        experiences = []
        for r in self.records:
            exp = Experience(
                task_type=f"{r.agent_name}:{r.task_type}",
                action=r.agent_name,
                context={
                    "task_type": r.task_type,
                    "params": str(r.parameters),
                    "execution_time": r.execution_time,
                },
                result=r.success,
                score=r.score,
                feedback=r.error,
            )
            experiences.append(exp)
        return experiences
    
    def get_summary(self) -> Dict[str, Any]:
        """获取共享池摘要"""
        total_records = len(self.records)
        total_success = sum(1 for r in self.records if r.success)
        agent_names = set(r.agent_name for r in self.records)
        task_types = set(r.task_type for r in self.records)
        
        return {
            "total_records": total_records,
            "success_rate": total_success / total_records if total_records > 0 else 0,
            "unique_agents": len(agent_names),
            "unique_tasks": len(task_types),
            "agent_breakdown": {
                name: {
                    "total": sum(1 for r in self.records if r.agent_name == name),
                    "success": sum(1 for r in self.records if r.agent_name == name and r.success),
                }
                for name in agent_names
            },
        }


class EvolvedStaffAgent(StaffAgent):
    """
    进化版参谋部 — 用进化系统的策略推荐做Agent路由
    
    不再硬编码agent_routing，而是根据经验共享池动态选择最优Agent。
    """
    
    def __init__(self, experience_pool: AgentExperiencePool = None, learning_engine: LearningEngine = None):
        super().__init__()
        self.experience_pool = experience_pool or AgentExperiencePool()
        self.learning_engine = learning_engine
        
        logger.info(f"[{self.name}] 进化版参谋部初始化完成")
    
    def _select_agent(self, params: Dict) -> Dict:
        """进化版Agent选择 — 基于历史成功率"""
        task_type = params.get("task_type", "")
        
        # 1. 先查经验共享池：谁干这个任务最靠谱？
        best_agent, success_rate = self.experience_pool.get_best_agent_for_task(task_type)
        
        # 2. 再查学习引擎：有没有更高级的策略推荐？
        if self.learning_engine:
            rec = self.learning_engine.recommend_strategy(f"agent:{task_type}")
            if rec and "unknown" not in rec:
                # 从推荐中提取Agent名
                for agent_name in ["scout", "builder", "comms", "staff"]:
                    if agent_name in rec:
                        best_agent = agent_name
                        break
        
        # 3. 如果成功率太低(<0.3)，尝试其他候选Agent
        candidates = []
        if success_rate < 0.3 and self.experience_pool.agent_stats:
            for key, stat in self.experience_pool.agent_stats.items():
                if key.endswith(f":{task_type}"):
                    agent = key.split(":")[0]
                    rate = stat.get("success", 0) / max(stat.get("total", 1), 1)
                    candidates.append((agent, rate))
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                best_agent = candidates[0][0]
        
        logger.info(f"[{self.name}] 进化路由: {task_type} -> {best_agent} (历史成功率: {success_rate:.1%})")
        return {"selected_agent": best_agent, "task_type": task_type, "estimated_success_rate": success_rate}
    
    def _evaluate_quality(self, params: Dict) -> Dict:
        """进化版质量评估 — 结合进化系统的feedback学习"""
        base_eval = super()._evaluate_quality(params)
        
        # 如果有学习引擎，调整评分
        if self.learning_engine and self.experience_pool.records:
            recent = self.experience_pool.records[-5:]
            avg_score = sum(r.score for r in recent) / len(recent)
            # 用最近平均分和基础评估的加权
            base_eval["score"] = base_eval["score"] * 0.7 + avg_score * 0.3
            base_eval["evolution_adjusted"] = True
        
        return base_eval


class EvolvedCommander(CommanderAgent):
    """
    进化版司令部 — 集成进化系统的级联决策 + Agent间通信中心
    
    特点:
    1. 用决策引擎选择执行策略(级联决策)
    2. Agent执行经验自动回流进化系统
    3. 支持进化系统指导任务分配
    4. 集成通信中心，实现侦察连→工兵连情报实时推送
    """
    
    def __init__(self, evolution_system=None, hub: Optional[AgentCommunicationHub] = None):
        # 通信中心（所有Agent共享）
        self.hub = hub or get_communication_hub()
        
        # 先创建下属部队（传入hub实现共享）
        self._pool = AgentExperiencePool()
        self._learning = LearningEngine()
        self._decision = DecisionEngine(risk_tolerance=0.4)
        self._monitor = PerformanceMonitor()
        
        staff = EvolvedStaffAgent(experience_pool=self._pool, learning_engine=self._learning)
        scout = ScoutAgent(hub=self.hub)
        builder = BuilderAgent(hub=self.hub)
        comms = CommsAgent()
        
        super().__init__(
            staff_agent=staff,
            scout_agent=scout,
            builder_agent=builder,
            comms_agent=comms,
        )
        
        self.evolution_system = evolution_system
        self.staff = staff
        
        logger.info(f"[{self.name}] 进化版司令部v2.0初始化完成 (通信中心已连接)")
    
    def execute_with_evolution(self, user_input: str) -> Dict:
        """
        进化版任务执行 — 完整打通流程
        
        流程:
        1. 进化系统做级联决策(选策略→选强度→选学习深度)
        2. 参谋部分析 + 动态Agent路由
        3. 选中的Agent执行
        4. 执行结果记录到共享池
        5. 共享池经验反馈给进化系统
        6. 进化系统学习并更新策略
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.name}] 进化任务启动: {user_input}")
        logger.info(f"{'='*60}")
        
        overall_start = time.time()
        
        # === Step 1: 进化系统级联决策 ===
        logger.info(f"[{self.name}] [进化决策] 级联决策制定...")
        
        # 根据经验池状态调整选项
        pool_summary = self._pool.get_summary()
        history_count = pool_summary["total_records"]
        base_uncertainty = min(0.5, max(0, 1 - history_count / 30) * 0.4)
        
        decision_chain = [
            (DecisionType.RESOURCE, [
                DecisionOption("direct_execute", "直接执行", 6.0, 0.2, 1.0, 0.8, uncertainty=base_uncertainty * 0.5),
                DecisionOption("analyze_then_execute", "先分析再执行", 7.5, 0.15, 2.0, 0.75, uncertainty=base_uncertainty * 0.6),
                DecisionOption("multi_agent", "多Agent协作", 8.0, 0.3, 3.0, 0.65, uncertainty=base_uncertainty + 0.15),
            ], "选择执行策略"),
            (DecisionType.PRIORITY, [
                DecisionOption("quick", "快速执行", 5.0, 0.3, 1.0, 0.8, uncertainty=0.1),
                DecisionOption("standard", "标准执行", 7.0, 0.2, 2.0, 0.85, uncertainty=0.15),
                DecisionOption("deep", "深入执行", 8.5, 0.4, 4.0, 0.7, uncertainty=0.25),
            ], "选择执行强度"),
        ]
        
        cascade = self._decision.cascade_decide(decision_chain, strategy="balanced")
        strategy = cascade[0].chosen
        intensity = cascade[1].chosen
        
        logger.info(f"[{self.name}] [进化决策] 策略={strategy}, 强度={intensity}")
        
        # === Step 2~6: 根据策略选择执行路径 ===
        if strategy == "direct_execute":
            result = self._direct_execute(user_input, intensity)
        elif strategy == "multi_agent":
            result = self._multi_agent_execute(user_input, intensity)
        else:
            # analyze_then_execute 或默认: 走完整参谋流程
            result = self._full_staff_execute(user_input, intensity)
        
        total_time = time.time() - overall_start
        result["evolution_meta"] = {
            "strategy": strategy,
            "intensity": intensity,
            "total_time": total_time,
            "pool_summary": self._pool.get_summary(),
        }
        
        # === Step 7: 进化系统学习 ===
        self._feedback_to_evolution(result, user_input)
        
        logger.info(f"[{self.name}] 进化任务完成! 策略={strategy}, 耗时={total_time:.2f}s")
        return result
    
    def _direct_execute(self, user_input: str, intensity: str) -> Dict:
        """直接执行模式 — 跳过参谋部分析，快速执行"""
        logger.info(f"[{self.name}] [直接执行] 快速响应...")
        
        # 简化为builder直接处理
        task = Task(
            task_type="code_write",
            description="直接执行用户请求",
            parameters={"input": user_input},
        )
        
        start = time.time()
        result = self.builder.execute(task)
        exec_time = time.time() - start
        
        # 记录到共享池
        self._pool.record(AgentExecutionRecord(
            agent_name="builder",
            task_type="code_write",
            parameters={"input": user_input},
            success=result.success,
            execution_time=exec_time,
            score=8.0 if result.success else 3.0,
            error=result.error,
        ))
        
        return {
            "success": result.success,
            "strategy": "direct_execute",
            "result": result.result,
            "error": result.error,
        }
    
    def _full_staff_execute(self, user_input: str, intensity: str) -> Dict:
        """完整参谋流程 — 分析→规划→执行→评估"""
        # 复用原CommanderAgent的process_user_input逻辑
        report = self.process_user_input(user_input)
        
        # 将执行结果记录到共享池
        if report.get("success"):
            for step_result in report.get("step_results", []):
                # 解析step_result字符串: "✅ [scout] web_search"
                agent_name = "unknown"
                task_type = "unknown"
                success = False
                
                if isinstance(step_result, str):
                    success = "✅" in step_result
                    parts = step_result.strip("✅❌ ").strip("[]").split("]")
                    if len(parts) >= 2:
                        agent_name = parts[0].strip("[] ")
                        task_type = parts[1].strip() if len(parts) > 1 else "unknown"
                
                self._pool.record(AgentExecutionRecord(
                    agent_name=agent_name,
                    task_type=task_type,
                    parameters={},
                    success=success,
                    execution_time=0.1,
                    score=8.0 if success else 2.0,
                ))
        
        return report
    
    def _multi_agent_execute(self, user_input: str, intensity: str) -> Dict:
        """
        多Agent协作模式 — 情报驱动执行
        
        流程:
        1. 侦察连执行搜索/侦察任务
        2. 情报通过通信中心实时推送给工兵连
        3. 工兵连根据情报上下文生成/调整代码
        4. 执行并反馈结果
        """
        logger.info(f"[{self.name}] [多Agent协作] 启动情报驱动执行...")
        
        # 给通信中心一点时间让消息处理线程运转
        import time as _time
        _time.sleep(0.1)
        
        results = {}
        
        # === Step 1: 侦察连执行搜索（真实搜索）===
        logger.info(f"[{self.name}] [多Agent协作] Step 1: 侦察连搜索...")
        search_query = user_input if len(user_input) < 50 else user_input[:50]
        scout_task = Task(
            task_type="web_search",
            description=f"搜索: {search_query}",
            parameters={"query": search_query, "max_results": 3},
        )
        scout_result = self.scout.execute(scout_task)
        
        # 记录侦察经验
        self._pool.record(AgentExecutionRecord(
            agent_name="scout", task_type="web_search",
            parameters={"query": search_query},
            success=scout_result.success,
            execution_time=scout_result.execution_time,
            score=8.0 if scout_result.success else 3.0,
            error=scout_result.error,
        ))
        
        # 给消息处理一点时间
        _time.sleep(0.1)
        
        # 检查builder是否收到了情报
        builder_intel = self.builder.get_intel_summary()
        logger.info(f"[{self.name}] [多Agent协作] 工兵连已接收 {builder_intel['total_intel']} 条情报")
        
        # === Step 2: 工兵连根据情报生成代码 ===
        logger.info(f"[{self.name}] [多Agent协作] Step 2: 工兵连情报驱动编码...")
        code_task = Task(
            task_type="code_write",
            description="编写代码",
            parameters={"input": user_input},
        )
        code_result = self.builder.execute(code_task)
        
        self._pool.record(AgentExecutionRecord(
            agent_name="builder", task_type="code_write",
            parameters={"input": user_input},
            success=code_result.success,
            execution_time=code_result.execution_time,
            score=8.0 if code_result.success else 2.0,
            error=code_result.error,
        ))
        
        # === Step 3: 如果生成成功，尝试执行 ===
        exec_result = None
        if code_result.success and code_result.result.get("file_path"):
            exec_task = Task(
                task_type="code_execute",
                description="执行生成的代码",
                parameters={"file_path": code_result.result["file_path"]},
            )
            exec_result = self.builder.execute(exec_task)
            
            self._pool.record(AgentExecutionRecord(
                agent_name="builder", task_type="code_execute",
                parameters={"file_path": code_result.result["file_path"]},
                success=exec_result.success,
                execution_time=exec_result.execution_time,
                score=9.0 if exec_result.success else 3.0,
                error=exec_result.error,
            ))
        
        # 汇总结果
        scout_data = scout_result.result if scout_result.success else {}
        code_data = code_result.result if code_result.success else {}
        
        results = {
            "success": code_result.success,
            "strategy": "multi_agent",
            "scout": {
                "query": scout_data.get("query", search_query),
                "results_count": scout_data.get("count", 0),
                "real_search": scout_data.get("real_search", False),
                "sample_results": [r.get("title", "") for r in scout_data.get("results", [])[:2]],
            },
            "builder": {
                "file_path": code_data.get("file_path", ""),
                "lines": code_data.get("lines", 0),
                "intel_guided": code_data.get("intel_guided", False),
                "execution": {
                    "stdout": exec_result.result.get("stdout", "")[:300] if exec_result and exec_result.success else None,
                    "success": exec_result.success if exec_result else False,
                } if exec_result else None,
            },
            "communication": {
                "hub_stats": self.hub.get_stats(),
                "builder_intel": builder_intel,
            },
        }
        
        logger.info(f"[{self.name}] [多Agent协作] 完成! 搜索={scout_data.get('count', 0)}条, "
                   f"代码={code_data.get('lines', 0)}行, 执行={'成功' if exec_result and exec_result.success else '失败/未执行'}")
        
        return results
    
    def _feedback_to_evolution(self, result: Dict, user_input: str):
        """将多Agent执行结果反馈给进化系统学习"""
        # 计算综合评分
        if result.get("success"):
            score = result.get("quality_score", 7.0)
        else:
            score = 2.0
        
        # 构建Experience
        strategy = result.get("evolution_meta", {}).get("strategy", "unknown")
        exp = Experience(
            task_type="multi_agent_task",
            action=strategy,
            context={
                "user_input": user_input,
                "strategy": strategy,
                "intensity": result.get("evolution_meta", {}).get("intensity", "standard"),
                "pool_records": self._pool.get_summary()["total_records"],
            },
            result=result.get("success", False),
            score=score,
            feedback=f"多Agent执行: {strategy}",
        )
        
        self._learning.record_experience(exp)
        
        # 同时将共享池所有记录也转给学习引擎
        for pool_exp in self._pool.to_experiences():
            self._learning.record_experience(pool_exp)
        
        logger.info(f"[{self.name}] [进化反馈] 经验已记录: score={score:.1f}, 策略={strategy}")
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """获取进化版司令部的完整报告"""
        return {
            "experience_pool": self._pool.get_summary(),
            "learning": self._learning.get_learning_summary(),
            "decisions": self._decision.get_decision_quality(),
            "performance": self._monitor.get_dashboard(),
        }


class MultiAgentEvolutionBridge:
    """
    多Agent-进化系统桥梁 v3.0
    
    提供统一的接口，让外部可以直接调用进化版多Agent系统。
    管理通信中心生命周期，集成任务编排器（分解+容错重试）。
    """
    
    def __init__(self):
        self.hub = get_communication_hub()
        self.commander = EvolvedCommander(hub=self.hub)
        
        # 集成编排器
        try:
            from orchestrator import Orchestrator
        except ImportError:
            from src.orchestrator import Orchestrator
        
        self.orchestrator = Orchestrator(
            agents={
                "scout": self.commander.scout,
                "builder": self.commander.builder,
                "comms": self.commander.comms,
                "staff": self.commander.staff,
            },
            experience_pool=self.commander._pool,
        )
        
        logger.info("多Agent-进化桥梁v3.0初始化完成 (编排器已集成)")
    
    def execute(self, user_input: str, use_orchestrator: bool = False) -> Dict:
        """
        执行用户请求
        
        Args:
            user_input: 用户输入
            use_orchestrator: 是否使用任务编排器（分解+容错重试），False则使用级联决策
        """
        if use_orchestrator:
            return self.orchestrator.execute(user_input)
        return self.commander.execute_with_evolution(user_input)
    
    def execute_complex(self, user_input: str) -> Dict:
        """执行复杂任务（自动分解+容错重试）"""
        return self.orchestrator.execute(user_input)
    
    def get_report(self) -> Dict:
        """获取完整报告"""
        report = self.commander.get_evolution_report()
        report["communication_hub"] = self.hub.get_stats()
        return report
    
    def shutdown(self):
        """关闭桥梁和通信中心"""
        self.orchestrator.shutdown()
        self.hub.shutdown()
        logger.info("多Agent-进化桥梁已关闭")
