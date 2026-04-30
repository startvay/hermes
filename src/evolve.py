"""
Hermes进化运行器
将学习引擎、决策引擎、性能优化器整合为一个可运行的进化系统

基于工程控制论设计：
- 反馈循环（负反馈控制）
- 自适应调节
- 多Agent协调
- 性能优化
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.task import Task, TaskResult, TaskPriority
from core.base_agent import BaseAgent
from core.agent_registry import get_registry
from core.message_bus import get_message_bus
from core.coordinator import AgentCoordinator
from core.signal_processor import SignalProcessor
from core.robustness import UncertaintyModel, RobustnessChecker
from core.adaptive import EnvironmentSensor, AdaptiveController, SelfOptimizer
from core.learning_engine import LearningEngine, Experience
from core.decision_engine import DecisionEngine, DecisionType, DecisionOption
from core.performance import PerformanceMonitor, PerformanceOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Evolution")


class EvolutionSystem:
    """
    进化系统
    
    整合所有子系统，提供统一的进化接口
    """
    
    def __init__(self):
        # 核心子系统
        self.registry = get_registry()
        self.message_bus = get_message_bus()
        self.coordinator = AgentCoordinator(self.registry)
        
        # 新增子系统（本次实现）
        self.learning_engine = LearningEngine()
        self.decision_engine = DecisionEngine(risk_tolerance=0.4)
        self.performance_monitor = PerformanceMonitor()
        self.performance_optimizer = PerformanceOptimizer()
        
        # 控制论子系统
        self.signal_processor = SignalProcessor()
        self.uncertainty_model = UncertaintyModel()
        self.robustness_checker = RobustnessChecker()
        self.env_sensor = EnvironmentSensor()
        self.adaptive_controller = AdaptiveController()
        self.self_optimizer = SelfOptimizer()
        
        # 进化状态
        self.evolution_cycle = 0
        self.start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("   H E R M E S   进 化 系 统   启 动")
        logger.info("   基于钱学森《工程控制论》设计")
        logger.info("=" * 60)
    
    def run_evolution_cycle(self, feedback: str = None, score: float = None):
        """
        执行一个进化周期
        
        包含：
        1. 环境感知
        2. 性能评估
        3. 决策制定
        4. 策略执行
        5. 反馈学习
        6. 参数优化
        """
        self.evolution_cycle += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"进化周期 #{self.evolution_cycle}")
        logger.info(f"{'='*50}")
        
        cycle_start = time.time()
        
        # 1. 环境感知
        logger.info("[1/6] 环境感知...")
        env_data = self.env_sensor.sense()
        env_summary = self.env_sensor.get_environment_summary()
        self.performance_monitor.record("env_complexity", env_summary.get("trend_score", 0.5))
        
        # 2. 性能评估
        logger.info("[2/6] 性能评估...")
        bottlenecks = self.performance_optimizer.identify_bottlenecks()
        recommendations = self.performance_optimizer.get_recommendations()
        
        # 3. 决策制定（级联决策，带误差补偿）
        logger.info("[3/6] 级联决策制定...")
        
        # 根据环境负载动态调整选项参数（控制论Ch.12 变系数系统）
        env_status = env_summary.get("status", "normal")
        trend_score = env_summary.get("trend_score", 0.5)
        load_factor = min(1.0, trend_score * 1.5)
        
        deep_learning_reward = 6.0 * (1 - load_factor * 0.4)
        deep_learning_risk = 0.3 + load_factor * 0.3
        deep_learning_cost = 4.0 + load_factor * 3.0
        expand_reward = 5.0 + load_factor * 1.5
        
        # 计算各选项的不确定性（基于环境负载和历史数据）
        # 负载越高，不确定性越高；历史数据越少，不确定性越高
        history_count = len(self.decision_engine.decision_history)
        base_uncertainty = min(0.5, load_factor * 0.3 + max(0, 1 - history_count / 50) * 0.3)
        
        # 级联决策链（3级）
        # 级1: 选择主策略方向
        # 级2: 选择执行强度（影响结果质量）
        # 级3: 选择学习深度（影响经验提取方式）
        decision_chain = [
            (DecisionType.RESOURCE, [
                DecisionOption(
                    name="optimize_performance",
                    description="优化系统性能",
                    expected_reward=7.0 if bottlenecks else 3.0,
                    risk=0.2,
                    cost=2.0,
                    confidence=0.8,
                    uncertainty=base_uncertainty * 0.8,
                ),
                DecisionOption(
                    name="expand_knowledge",
                    description="扩展知识库",
                    expected_reward=expand_reward,
                    risk=0.1,
                    cost=1.0,
                    confidence=0.7,
                    uncertainty=base_uncertainty * 0.6,
                ),
                DecisionOption(
                    name="deep_learning",
                    description="深度学习",
                    expected_reward=deep_learning_reward,
                    risk=deep_learning_risk,
                    cost=deep_learning_cost,
                    confidence=max(0.3, 0.6 - load_factor * 0.3),
                    uncertainty=base_uncertainty + 0.2,  # 深度学习不确定性更高
                ),
            ], "选择主策略方向"),
            (DecisionType.PRIORITY, [
                DecisionOption(name="quick", description="快速执行", expected_reward=5.0, risk=0.3, cost=1.0, confidence=0.8, uncertainty=base_uncertainty * 0.5),
                DecisionOption(name="standard", description="标准执行", expected_reward=7.0, risk=0.2, cost=2.0, confidence=0.85, uncertainty=base_uncertainty * 0.6),
                DecisionOption(name="deep", description="深入执行", expected_reward=8.5, risk=0.4, cost=4.0, confidence=0.7, uncertainty=base_uncertainty * 0.9),
            ], "选择执行强度"),
            (DecisionType.ESCALATION, [
                DecisionOption(name="shallow_learn", description="浅层学习", expected_reward=4.0, risk=0.05, cost=0.5, confidence=0.9, uncertainty=base_uncertainty * 0.4),
                DecisionOption(name="standard_learn", description="标准学习", expected_reward=6.0, risk=0.1, cost=1.0, confidence=0.85, uncertainty=base_uncertainty * 0.5),
                DecisionOption(name="deep_learn", description="深度学习", expected_reward=7.5, risk=0.2, cost=2.5, confidence=0.75, uncertainty=base_uncertainty * 0.7),
            ], "选择学习深度"),
        ]
        
        logger.info(f"  环境负载: {load_factor:.1%} ({env_status})")
        logger.info(f"  基础不确定性: {base_uncertainty:.2f}")
        
        # 使用级联决策
        cascade_results = self.decision_engine.cascade_decide(decision_chain, strategy="balanced")
        
        # 提取各级决策
        main_decision = cascade_results[0]
        intensity_decision = cascade_results[1]
        learn_decision = cascade_results[2]
        
        # v3.0+: 评估整体决策不确定性，必要时主动求助
        uncertainty_eval = self.decision_engine.evaluate_uncertainty(
            DecisionType.RESOURCE, decision_chain[0][1]
        )
        if uncertainty_eval["needs_human_input"]:
            logger.warning(f"[主动求助] 决策不确定性高，建议人类确认!")
            logger.warning(f"  原因: {uncertainty_eval['reason']}")
            logger.warning(f"  整体不确定性: {uncertainty_eval['overall_uncertainty']}")
            main_decision.reason += f" | [需确认] {uncertainty_eval['reason']}"
        
        logger.info(f"  级联决策结果:")
        logger.info(f"    主策略: {main_decision.chosen} ({main_decision.reason})")
        logger.info(f"    执行强度: {intensity_decision.chosen} ({intensity_decision.reason})")
        logger.info(f"    学习深度: {learn_decision.chosen} ({learn_decision.reason})")
        
        # 4. 策略执行（传入级联决策结果）
        logger.info(f"[4/6] 执行策略: {main_decision.chosen} | 强度: {intensity_decision.chosen}")
        action_result = self._execute_strategy(main_decision.chosen, env_data, intensity_decision.chosen)
        
        # 5. 反馈学习（根据学习深度决策调整）
        logger.info(f"[5/6] 反馈学习 (模式: {learn_decision.chosen})...")
        
        # v3.0+: 计算预期评分（基于决策选项的expected_reward）
        chosen_main_opt = next((o for o in decision_chain[0][1] if o.name == main_decision.chosen), None)
        expected_score = chosen_main_opt.expected_reward if chosen_main_opt else 5.0
        
        exp = Experience(
            task_type="evolution_cycle",
            action=main_decision.chosen,
            context={
                "cycle": self.evolution_cycle,
                "env": str(env_data),
                "intensity": intensity_decision.chosen,
                "learn_depth": learn_decision.chosen,
                "base_uncertainty": base_uncertainty,
            },
            result=action_result.get("success", True),
            score=score if score else action_result.get("score", 5.0),
            expected_score=expected_score,
            feedback=feedback,
        )
        self.learning_engine.record_experience(exp)
        
        if feedback and score:
            # 根据学习深度调整feedback权重
            depth_multiplier = {"shallow_learn": 0.7, "standard_learn": 1.0, "deep_learn": 1.3}.get(learn_decision.chosen, 1.0)
            adjusted_score = min(10.0, score * depth_multiplier)
            self.learning_engine.learn_from_feedback(
                "evolution_cycle", main_decision.chosen, feedback, adjusted_score
            )
        
        # 6. 参数优化
        logger.info("[6/6] 参数优化...")
        self._optimize_parameters()
        
        # 记录周期时间
        cycle_time = time.time() - cycle_start
        self.performance_monitor.record("cycle_time", cycle_time)
        
        # 输出周期报告（使用主决策）
        report = self._generate_cycle_report(main_decision, action_result, cycle_time)
        logger.info(f"\n{report}")
        
        return report
    
    def _execute_strategy(self, strategy: str, env_data: Dict, intensity: str = "standard") -> Dict:
        """执行选定的策略（支持执行强度调节）"""
        
        # 强度系数影响结果质量
        intensity_multiplier = {
            "quick": 0.8,
            "standard": 1.0,
            "deep": 1.3,
        }.get(intensity, 1.0)
        
        if strategy == "optimize_performance":
            result = self._optimize_performance()
        elif strategy == "expand_knowledge":
            result = self._expand_knowledge()
        elif strategy == "deep_learning":
            result = self._deep_learning()
        else:
            result = {"success": True, "score": 5.0, "details": "默认策略"}
        
        # 应用强度系数到分数
        if "score" in result:
            base_score = result["score"]
            # quick: 降低上限但保证下限；deep: 提高上限但也有风险
            if intensity == "quick":
                result["score"] = max(3.0, base_score * intensity_multiplier)
            elif intensity == "deep":
                result["score"] = min(10.0, base_score * intensity_multiplier)
            else:
                result["score"] = base_score
            result["intensity_applied"] = intensity
        
        return result
    
    def _optimize_performance(self) -> Dict:
        """性能优化策略"""
        logger.info("  -> 分析性能瓶颈...")
        
        bottlenecks = self.performance_optimizer.identify_bottlenecks()
        recs = self.performance_optimizer.get_recommendations()
        
        optimizations = 0
        for rec in recs:
            logger.info(f"  -> 建议: {rec}")
            optimizations += 1
        
        # 模拟自动调参
        self.performance_optimizer.register_param(
            "batch_size", min_val=1, max_val=64, current=8, step=2
        )
        self.performance_optimizer.register_param(
            "timeout", min_val=1, max_val=30, current=10, step=2
        )
        
        new_batch = self.performance_optimizer.auto_tune(
            "batch_size", "response_time", "minimize"
        )
        
        return {
            "success": True,
            "score": 7.0 if bottlenecks else 4.0,
            "details": f"识别{len(bottlenecks)}个瓶颈, {optimizations}条建议",
            "optimizations": optimizations,
        }
    
    def _expand_knowledge(self) -> Dict:
        """知识扩展策略"""
        logger.info("  -> 扩展知识库...")
        
        summary = self.learning_engine.get_learning_summary()
        knowledge_count = summary["stats"]["total_knowledge"]
        
        # 模拟学习新知识
        test_experiences = [
            Experience("code_generation", "python", {"lang": "python"}, True, score=8.0),
            Experience("web_search", "duckduckgo", {"engine": "ddg"}, True, score=7.0),
            Experience("file_operation", "read", {"type": "text"}, True, score=9.0),
            Experience("code_generation", "javascript", {"lang": "js"}, False, score=3.0),
        ]
        
        for exp in test_experiences:
            self.learning_engine.record_experience(exp)
        
        return {
            "success": True,
            "score": 6.0,
            "details": f"知识库: {knowledge_count} -> {knowledge_count + 4}条",
            "new_knowledge": 4,
        }
    
    def _deep_learning(self) -> Dict:
        """深度学习策略"""
        logger.info("  -> 深度学习分析...")
        
        # 分析历史决策质量
        decision_quality = self.decision_engine.get_decision_quality()
        
        # 分析学习效果
        learning_summary = self.learning_engine.get_learning_summary()
        
        return {
            "success": True,
            "score": 6.5,
            "details": f"决策数:{decision_quality['total_decisions']}, "
                      f"知识域:{len(learning_summary['knowledge_domains'])}",
            "decision_quality": decision_quality,
        }
    
    def _optimize_parameters(self):
        """系统参数优化（PID自适应版）"""
        # 1. 基于决策成功率调整风险容忍度
        dq = self.decision_engine.get_decision_quality()
        if dq.get("success_rate") and dq["success_rate"] != "N/A":
            rate = float(dq["success_rate"].strip("%")) / 100
            if rate > 0.8:
                self.decision_engine.risk_tolerance = min(0.7, self.decision_engine.risk_tolerance + 0.05)
            elif rate < 0.5:
                self.decision_engine.risk_tolerance = max(0.2, self.decision_engine.risk_tolerance - 0.05)
        
        # 2. PID自适应调节学习率和探索率（新增）
        # 综合性能指标 = 决策成功率 + 学习效率 - 周期耗时惩罚
        learning_summary = self.learning_engine.get_learning_summary()
        perf_score = 0.5  # 默认
        if dq.get("success_rate") and dq["success_rate"] != "N/A":
            perf_score = float(dq["success_rate"].strip("%")) / 100
        
        # 用AdaptiveController的PID来调节参数
        self.adaptive_controller.adapt(performance=perf_score, target=0.8)
        
        # 将PID调节结果同步到决策引擎
        new_params = self.adaptive_controller.parameters
        self.decision_engine.exploration_rate = new_params["exploration_rate"]
        
        logger.info(f"  PID调节: learning_rate={new_params['learning_rate']:.3f}, "
                    f"exploration_rate={new_params['exploration_rate']:.3f}")
    
    def _generate_cycle_report(self, decision, action_result, cycle_time) -> str:
        """生成周期报告"""
        elapsed = time.time() - self.start_time
        
        report = f"""
┌──────────────────────────────────────────┐
│  进化周期 #{self.evolution_cycle} 报告                  │
├──────────────────────────────────────────┤
│  已运行: {elapsed:.0f}秒                        
│  周期耗时: {cycle_time:.2f}秒                      
│                                          
│  决策: {decision.chosen:<30}       
│  原因: {decision.reason[:35]:<35}  
│  结果: {'✅ 成功' if action_result.get('success') else '❌ 失败'}                           
│  评分: {action_result.get('score', 0):.1f}/10                         
│                                          
│  详情: {str(action_result.get('details', ''))[:35]:<35}
├──────────────────────────────────────────┤
│  学习状态                                
│  - 经验数: {self.learning_engine.stats['total_experiences']}                            
│  - 知识数: {self.learning_engine.stats['total_knowledge']}                              
│  - 决策数: {len(self.decision_engine.decision_history)}                              
│  - 告警数: {len(self.performance_monitor.alerts)}                              
├──────────────────────────────────────────┤
│  策略推荐                                
"""
        
        # 添加策略推荐
        for task_type in ["code_generation", "web_search", "file_operation"]:
            rec = self.learning_engine.recommend_strategy(task_type)
            if rec:
                report += f"│  - {task_type}: {rec}\n"
        
        report += "└──────────────────────────────────────────┘"
        return report
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """获取完整进化报告"""
        return {
            "evolution_cycles": self.evolution_cycle,
            "uptime_seconds": time.time() - self.start_time,
            "learning": self.learning_engine.get_learning_summary(),
            "decisions": self.decision_engine.get_decision_quality(),
            "performance": self.performance_monitor.get_dashboard(),
            "bottlenecks": self.performance_optimizer.identify_bottlenecks(),
            "recommendations": self.performance_optimizer.get_recommendations(),
        }
    
    def save_report(self, filepath: str = None):
        """保存进化报告到文件"""
        if filepath is None:
            filepath = str(Path.home() / ".hermes/evolution/evolution_report.json")
        
        report = self.get_evolution_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"进化报告已保存: {filepath}")
        return filepath


def main():
    """主函数：运行进化系统"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     H E R M E S   进 化 系 统                                ║
    ║                                                              ║
    ║     基于钱学森《工程控制论》设计                              ║
    ║     多Agent协作 · 自主学习 · 决策优化 · 性能监控             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建进化系统
    system = EvolutionSystem()
    
    # 运行多个进化周期
    feedbacks = [
        ("继续优化", 7.0),
        ("不错，但响应有点慢", 6.0),
        ("很好！", 9.0),
    ]
    
    for feedback, score in feedbacks:
        system.run_evolution_cycle(feedback=feedback, score=score)
        time.sleep(0.5)  # 模拟间隔
    
    # 运行无反馈的周期
    for _ in range(3):
        system.run_evolution_cycle()
        time.sleep(0.3)
    
    # 生成最终报告
    print("\n" + "=" * 60)
    print("   最终进化报告")
    print("=" * 60)
    
    report = system.get_evolution_report()
    
    print(f"\n📊 进化概况:")
    print(f"   总周期数: {report['evolution_cycles']}")
    print(f"   运行时间: {report['uptime_seconds']:.1f}秒")
    
    print(f"\n🧠 学习状态:")
    learning = report['learning']
    print(f"   经验总数: {learning['stats']['total_experiences']}")
    print(f"   知识条目: {learning['stats']['total_knowledge']}")
    print(f"   知识领域: {', '.join(learning['knowledge_domains'])}")
    
    print(f"\n🎯 决策质量:")
    decisions = report['decisions']
    print(f"   总决策数: {decisions['total_decisions']}")
    print(f"   成功率: {decisions['success_rate']}")
    print(f"   探索率: {decisions['exploration_rate']}")
    
    print(f"\n📈 性能监控:")
    perf = report['performance']
    for metric, data in perf['metrics'].items():
        if 'mean' in data:
            print(f"   {metric}: 均值={data['mean']:.3f}, 趋势={data['trend']}")
    
    print(f"\n⚠️ 瓶颈分析:")
    for b in report['bottlenecks']:
        print(f"   [{b['metric']}] 风险分={b['score']}: {', '.join(b['reasons'])}")
    
    print(f"\n💡 优化建议:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    # 保存报告
    report_path = system.save_report()
    
    # 保存到桌面
    desktop_path = str(Path.home() / ".hermes/evolution/Hermes进化报告_实时版.md")
    with open(desktop_path, "w") as f:
        f.write("# Hermes进化系统 - 实时进化报告\n\n")
        f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 进化概况\n\n")
        f.write(f"- 总进化周期: {report['evolution_cycles']}\n")
        f.write(f"- 运行时间: {report['uptime_seconds']:.1f}秒\n\n")
        f.write(f"## 学习成果\n\n")
        f.write(f"- 累计经验: {learning['stats']['total_experiences']}条\n")
        f.write(f"- 知识条目: {learning['stats']['total_knowledge']}条\n")
        f.write(f"- 知识领域: {', '.join(learning['knowledge_domains'])}\n\n")
        f.write(f"## 决策质量\n\n")
        f.write(f"- 决策总数: {decisions['total_decisions']}\n")
        f.write(f"- 成功率: {decisions['success_rate']}\n\n")
        f.write(f"## 优化建议\n\n")
        for rec in report['recommendations']:
            f.write(f"- {rec}\n")
        f.write(f"\n## 技术架构\n\n")
        f.write(f"基于钱学森《工程控制论》设计，包含以下子系统：\n\n")
        f.write(f"1. **学习引擎** - 从经验、反馈、类比中学习\n")
        f.write(f"2. **决策引擎** - 多策略决策、风险评估\n")
        f.write(f"3. **性能监控** - 实时监控、阈值告警\n")
        f.write(f"4. **性能优化** - 自动调参、瓶颈识别\n")
        f.write(f"5. **信号处理** - 噪声过滤、趋势分析\n")
        f.write(f"6. **鲁棒性检查** - 不确定性建模\n")
        f.write(f"7. **自适应控制** - 环境感知、自动调节\n")
        f.write(f"8. **多Agent协调** - 任务分解、并行执行\n")
    
    print(f"\n✅ 报告已保存:")
    print(f"   - JSON: {report_path}")
    print(f"   - Markdown: {desktop_path}")
    
    print("\n" + "=" * 60)
    print("   进化系统运行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
