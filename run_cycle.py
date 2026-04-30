#!/usr/bin/env python3
"""
Hermes进化系统 - 运行入口
=========================
执行一个完整的进化周期：
1. 环境感知
2. 经验回放学习
3. 策略优化
4. 知识蒸馏
5. 自优化调参
"""

import sys
import time
sys.path.insert(0, "src")

from evolve import HermesEvoSystem

def main():
    print("=" * 60)
    print("Hermes 自进化系统 v3.0++++")
    print("=" * 60)
    
    evo = HermesEvoSystem()
    
    print("\n[1/5] 环境感知...")
    env_state = evo.environment_sensor.sense()
    print(f"    CPU: {env_state.cpu_usage:.1f}% | 内存: {env_state.memory_usage:.1f}%")
    
    print("\n[2/5] 经验回放学习...")
    replay_results = evo.learning_engine.replay_experiences(n=5)
    print(f"    回放 {len(replay_results)} 条经验")
    
    print("\n[3/5] 策略优化...")
    suggestions = evo.decision_engine.suggest_strategies(task_type="code_gen")
    if suggestions:
        best = max(suggestions, key=lambda x: x.score)
        print(f"    最优策略: {best.name} (分数: {best.score:.3f})")
    
    print("\n[4/5] 知识蒸馏...")
    new_knowledge = evo.learning_engine.distill_knowledge()
    print(f"    新增知识: {len(new_knowledge)} 条")
    
    print("\n[5/5] 自优化调参...")
    improved = evo.self_optimizer.optimize()
    print(f"    优化结果: {'✓ 收敛' if improved else '— 保持现状'}")
    
    status = evo.get_status()
    print("\n" + "=" * 60)
    print(f"进化周期完成！")
    print(f"  总经验: {status['total_experiences']}")
    print(f"  总知识: {status['total_knowledge']}")
    print(f"  策略准确率: {status.get('strategy_accuracy', 'N/A')}")
    print("=" * 60)

if __name__ == "__main__":
    main()