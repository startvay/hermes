"""
决策引擎
基于工程控制论第10章（非线性系统）和第12章（最优控制）设计

核心功能：
- 多目标决策
- 风险评估
- 不确定性下的决策
- 决策质量追踪
"""

import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """决策类型"""
    ROUTING = "routing"          # 任务路由决策
    RESOURCE = "resource"        # 资源分配决策
    PRIORITY = "priority"        # 优先级决策
    RETRY = "retry"              # 重试决策
    ESCALATION = "escalation"    # 升级决策


@dataclass
class DecisionOption:
    """决策选项 v3.0+ — 支持不确定性量化"""
    name: str
    description: str
    expected_reward: float  # 预期收益 0-10
    risk: float            # 风险 0-1
    cost: float            # 成本 0-10
    confidence: float      # 置信度 0-1
    uncertainty: float = 0.0  # 不确定性 0-1（越高越不确定）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def score(self) -> float:
        """综合评分（考虑不确定性惩罚）"""
        # 风险调整后的收益
        risk_adjusted = self.expected_reward * (1 - self.risk * 0.5)
        # 成本调整
        cost_adjusted = risk_adjusted - self.cost * 0.3
        # 置信度加权
        confidence_weighted = cost_adjusted * self.confidence
        # 不确定性惩罚：高不确定性会拉低评分
        uncertainty_penalty = 1.0 - self.uncertainty * 0.3
        return confidence_weighted * uncertainty_penalty
    
    @property
    def needs_human_verification(self) -> bool:
        """是否需要人类确认：不确定性极高或置信度极低"""
        return self.uncertainty > 0.7 or self.confidence < 0.3


@dataclass
class Decision:
    """决策记录"""
    decision_type: DecisionType
    options: List[DecisionOption]
    chosen: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    outcome: Optional[bool] = None  # 事后评估
    
    def evaluate(self, actual_reward: float, expected_reward: float):
        """事后评估决策质量"""
        if expected_reward > 0:
            self.outcome = actual_reward >= expected_reward * 0.7


class DecisionEngine:
    """
    决策引擎 v3.0 — 支持级联决策误差补偿
    
    实现多种决策策略：
    1. 最大期望收益
    2. 最小遗憾
    3. 风险厌恶
    4. 探索/利用平衡
    5. 级联决策（误差补偿闭环）
    """
    
    def __init__(self, risk_tolerance: float = 0.5):
        self.risk_tolerance = risk_tolerance  # 0=极度厌恶, 1=极度偏好
        self.decision_history: List[Decision] = []
        self.decision_quality_scores: List[float] = []
        
        # 探索率 (epsilon-greedy)
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        
        # 迟滞带状态（工程控制论Ch.10 继电器伺服系统）
        self._last_risk_averse_choice: Optional[str] = None
        self._hysteresis_band: float = 0.05  # ±5%迟滞带
        
        # 级联误差补偿状态（Ch.4 反馈控制）
        self._cascade_error_memory: float = 0.0  # 级联误差累积
        self._error_compensation_gain: float = 0.3  # 补偿增益
        
        logger.info(f"决策引擎初始化完成, 风险容忍度: {risk_tolerance}")
    
    def cascade_decide(self, decision_chain: List[Tuple[DecisionType, List[DecisionOption], str]],
                       strategy: str = "balanced") -> List[Decision]:
        """
        级联决策（多阶段，带误差补偿）
        
        原理: 工程控制论Ch.4 级联反馈系统
        - 每一级决策后，根据实际结果评估误差
        - 误差通过补偿增益传递到下一级，修正confidence/risk
        - 避免"多米诺骨牌"式误差累积
        
        Args:
            decision_chain: [(DecisionType, options, description), ...]
            strategy: 决策策略
            
        Returns:
            List[Decision]: 每级决策结果
        """
        if not decision_chain:
            return []
        
        results = []
        accumulated_error = 0.0
        
        for i, (dtype, options, desc) in enumerate(decision_chain):
            # 根据累积误差修正当前级的选项参数
            compensated_options = self._apply_error_compensation(options, accumulated_error)
            
            decision = self.decide(dtype, compensated_options, strategy)
            decision.reason += f" | 级联补偿:{accumulated_error:+.2f}"
            
            results.append(decision)
            
            # 模拟评估当前决策的实际效果（简化版：基于chosen option的expected_reward vs 最优option的对比）
            if options:
                best_reward = max(o.expected_reward for o in options)
                chosen_opt = next((o for o in options if o.name == decision.chosen), None)
                if chosen_opt:
                    actual_reward = chosen_opt.expected_reward
                    # 计算决策误差：最优值 - 实际值
                    decision_error = (best_reward - actual_reward) / 10.0  # 归一化到0-1
                    accumulated_error += decision_error * self._error_compensation_gain
                    accumulated_error = max(-0.5, min(0.5, accumulated_error))  # 限制累积范围
                    
                    logger.info(f"  级联级{i+1} 误差:{decision_error:.3f} 累积:{accumulated_error:.3f}")
        
        self._cascade_error_memory = accumulated_error
        return results
    
    def _apply_error_compensation(self, options: List[DecisionOption], 
                                   accumulated_error: float) -> List[DecisionOption]:
        """将累积误差应用到选项参数上"""
        if abs(accumulated_error) < 0.01:
            return options
        
        compensated = []
        for opt in options:
            # 误差为负（前级表现好）→ 提高confidence，降低risk
            # 误差为正（前级表现差）→ 降低confidence，提高risk，增加cost（更谨慎）
            new_confidence = max(0.1, min(1.0, opt.confidence - accumulated_error * 0.5))
            new_risk = max(0.0, min(1.0, opt.risk + accumulated_error * 0.3))
            new_cost = max(0.0, opt.cost + accumulated_error * 2.0)
            
            compensated.append(DecisionOption(
                name=opt.name,
                description=opt.description,
                expected_reward=opt.expected_reward,
                risk=new_risk,
                cost=new_cost,
                confidence=new_confidence,
                metadata=opt.metadata,
            ))
        return compensated
    
    def decide(self, decision_type: DecisionType, 
               options: List[DecisionOption],
               strategy: str = "balanced") -> Decision:
        """
        做出决策
        
        Args:
            decision_type: 决策类型
            options: 可选方案
            strategy: 决策策略
                - "greedy": 纯利用，选最高分
                - "balanced": 平衡收益与风险
                - "exploration": 纯探索，随机选
                - "risk_averse": 风险厌恶
                
        Returns:
            Decision: 决策结果
        """
        if not options:
            raise ValueError("没有可选方案")
        
        # 应用探索策略
        if strategy == "exploration" or (strategy == "balanced" and 
            self._should_explore()):
            chosen = self._explore(options)
            reason = f"探索模式 (ε={self.exploration_rate:.3f})"
        elif strategy == "greedy":
            chosen = self._greedy(options)
            reason = "贪婪策略: 选择最高期望收益"
        elif strategy == "risk_averse":
            chosen = self._risk_averse(options)
            reason = "风险厌恶策略: 最小化下行风险"
        else:  # balanced
            chosen = self._balanced(options)
            reason = "平衡策略: 收益-风险-成本综合评估"
        
        decision = Decision(
            decision_type=decision_type,
            options=options,
            chosen=chosen.name,
            reason=reason,
        )
        
        self.decision_history.append(decision)
        
        # 衰减探索率
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        logger.info(f"决策: [{decision_type.value}] -> {chosen.name} ({reason})")
        return decision
    
    def _greedy(self, options: List[DecisionOption]) -> DecisionOption:
        """贪婪: 选期望收益最高的"""
        return max(options, key=lambda o: o.expected_reward)
    
    def _balanced(self, options: List[DecisionOption]) -> DecisionOption:
        """平衡: 综合评分最高"""
        return max(options, key=lambda o: o.score)
    
    def _risk_averse(self, options: List[DecisionOption]) -> DecisionOption:
        """风险厌恶: 在低风险中选收益最高的，带迟滞带防止继电器抖动"""
        # 动态阈值：基于risk_tolerance调整
        base_threshold = 0.3 + (1 - self.risk_tolerance) * 0.2
        
        # 迟滞带逻辑：如果上次选择的选项仍在扩展安全区内，优先保持
        if self._last_risk_averse_choice:
            last_opt = next((o for o in options if o.name == self._last_risk_averse_choice), None)
            if last_opt:
                # 扩展阈值：上次选择的选项允许在 base_threshold + hysteresis 范围内
                extended_threshold = base_threshold + self._hysteresis_band
                if last_opt.risk < extended_threshold:
                    # 保持上次选择，避免边界抖动
                    return last_opt
        
        # 标准过滤：使用基础阈值
        safe_options = [o for o in options if o.risk < base_threshold]
        chosen = None
        if safe_options:
            chosen = max(safe_options, key=lambda o: o.expected_reward)
        else:
            chosen = min(options, key=lambda o: o.risk)
        
        self._last_risk_averse_choice = chosen.name
        return chosen
    
    def _explore(self, options: List[DecisionOption]) -> DecisionOption:
        """探索: 随机选择"""
        import random
        return random.choice(options)
    
    def _should_explore(self) -> bool:
        """决定是否探索"""
        import random
        return random.random() < self.exploration_rate
    
    def route_task(self, task_type: str, available_agents: List[Dict],
                   history: Dict[str, float] = None) -> Optional[str]:
        """
        任务路由决策
        
        Args:
            task_type: 任务类型
            available_agents: 可用Agent列表 [{name, capabilities, load, success_rate}]
            history: 历史成功率 {agent_name: rate}
            
        Returns:
            推荐的Agent名称
        """
        if not available_agents:
            return None
        
        options = []
        for agent in available_agents:
            # 计算能力匹配度
            capabilities = agent.get("capabilities", [])
            capability_match = 1.0 if task_type in capabilities else 0.3
            
            # 负载调整
            load = agent.get("load", 0.5)
            load_factor = 1.0 - load
            
            # 历史成功率
            hist_rate = history.get(agent["name"], 0.5) if history else 0.5
            
            expected_reward = (capability_match * 0.4 + load_factor * 0.3 + hist_rate * 0.3) * 10
            risk = load * 0.5 + (1 - capability_match) * 0.3
            cost = load * 5
            
            options.append(DecisionOption(
                name=agent["name"],
                description=f"能力匹配:{capability_match:.0%} 负载:{load:.0%}",
                expected_reward=expected_reward,
                risk=risk,
                cost=cost,
                confidence=hist_rate,
            ))
        
        decision = self.decide(DecisionType.ROUTING, options)
        return decision.chosen
    
    def should_retry(self, attempt: int, max_attempts: int, 
                     last_error: str, task_importance: float = 0.5) -> Tuple[bool, str]:
        """
        重试决策
        
        Args:
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数
            last_error: 上次错误信息
            task_importance: 任务重要性 0-1
            
        Returns:
            (是否重试, 原因)
        """
        if attempt >= max_attempts:
            return False, f"已达最大重试次数 ({max_attempts})"
        
        # 分析错误类型
        is_transient = any(keyword in last_error.lower() 
                         for keyword in ["timeout", "connection", "temporary", "busy"])
        
        if is_transient:
            # 瞬时错误，高概率重试
            retry_prob = 0.9 ** attempt
        else:
            # 持久错误，低概率重试
            retry_prob = 0.5 ** attempt
        
        # 重要任务更愿意重试（钳制到[0,1]范围）
        retry_prob = min(1.0, retry_prob * (1 + task_importance * 0.5))
        
        import random
        should = random.random() < retry_prob
        
        reason = f"重试概率:{retry_prob:.1%} (尝试{attempt}/{max_attempts}, 瞬时错误:{is_transient})"
        return should, reason
    
    def evaluate_uncertainty(self, decision_type: DecisionType, 
                             options: List[DecisionOption]) -> Dict[str, Any]:
        """
        评估决策不确定性
        
        Returns:
            {
                "overall_uncertainty": float,  # 整体不确定性 0-1
                "max_option_uncertainty": float,
                "needs_human_input": bool,      # 是否需要人类确认
                "reason": str,
                "suggested_action": str,        # 建议行动
            }
        """
        if not options:
            return {
                "overall_uncertainty": 1.0,
                "max_option_uncertainty": 1.0,
                "needs_human_input": True,
                "reason": "没有可用选项",
                "suggested_action": "等待人类提供选项",
            }
        
        # 计算各指标
        uncertainties = [o.uncertainty for o in options]
        confidences = [o.confidence for o in options]
        scores = [o.score for o in options]
        
        avg_uncertainty = sum(uncertainties) / len(uncertainties)
        max_uncertainty = max(uncertainties)
        avg_confidence = sum(confidences) / len(confidences)
        score_spread = max(scores) - min(scores) if len(scores) > 1 else 0
        
        # 需要人类确认的几种情况：
        # 1. 最高不确定性 > 0.7
        # 2. 平均置信度 < 0.4
        # 3. 选项得分差距很小（<0.5）且不确定性高
        needs_human = False
        reasons = []
        
        if max_uncertainty > 0.7:
            needs_human = True
            reasons.append(f"选项不确定性过高({max_uncertainty:.1f})")
        
        if avg_confidence < 0.4:
            needs_human = True
            reasons.append(f"平均置信度过低({avg_confidence:.1f})")
        
        if score_spread < 0.5 and avg_uncertainty > 0.5:
            needs_human = True
            reasons.append(f"选项难分优劣且不确定性高")
        
        # 整体不确定性 = 平均不确定性的加权
        overall = avg_uncertainty * 0.6 + (1 - avg_confidence) * 0.4
        
        return {
            "overall_uncertainty": round(overall, 2),
            "max_option_uncertainty": round(max_uncertainty, 2),
            "needs_human_input": needs_human,
            "reason": "; ".join(reasons) if reasons else "决策条件充分",
            "suggested_action": "请人类确认" if needs_human else "自动执行",
        }
    
    def get_decision_quality(self) -> Dict[str, Any]:
        """获取决策质量报告"""
        if not self.decision_history:
            return {"message": "暂无决策记录"}
        
        evaluated = [d for d in self.decision_history if d.outcome is not None]
        
        if evaluated:
            success_rate = sum(1 for d in evaluated if d.outcome) / len(evaluated)
        else:
            success_rate = None
        
        type_counts = {}
        for d in self.decision_history:
            t = d.decision_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "evaluated_decisions": len(evaluated),
            "success_rate": f"{success_rate:.1%}" if success_rate is not None else "N/A",
            "exploration_rate": f"{self.exploration_rate:.3f}",
            "decision_types": type_counts,
            "risk_tolerance": self.risk_tolerance,
        }
