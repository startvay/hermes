"""
自适应与优化系统
基于工程控制论第12-15章设计

功能：
- 环境适应
- 参数优化
- 自主学习
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """环境状态"""
    cpu_usage: float
    memory_usage: float
    task_load: int
    error_rate: float
    timestamp: float


class EnvironmentSensor:
    """
    环境传感器
    
    基于工程控制论第12章（变系数线性系统）
    """
    
    def __init__(self):
        self.history: List[EnvironmentState] = []
        
    def sense(self) -> EnvironmentState:
        """感知环境状态"""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
        except ImportError:
            # psutil不可用时使用模拟数据
            import random
            cpu = random.uniform(5, 30)
            mem = random.uniform(30, 60)
        
        state = EnvironmentState(
            cpu_usage=cpu,
            memory_usage=mem,
            task_load=0,
            error_rate=0.0,
            timestamp=time.time(),
        )
        
        self.history.append(state)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return state
    
    def detect_change(self, threshold: float = 0.2) -> bool:
        """检测环境变化"""
        if len(self.history) < 2:
            return False
        
        current = self.history[-1]
        previous = self.history[-2]
        
        cpu_change = abs(current.cpu_usage - previous.cpu_usage) / 100
        memory_change = abs(current.memory_usage - previous.memory_usage) / 100
        
        return cpu_change > threshold or memory_change > threshold
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """获取环境摘要"""
        if not self.history:
            return {"status": "no_data", "trend_score": 0.5}
        
        recent = self.history[-10:] if len(self.history) >= 10 else self.history
        
        avg_cpu = sum(s.cpu_usage for s in recent) / len(recent)
        avg_mem = sum(s.memory_usage for s in recent) / len(recent)
        
        # 趋势评分: 0=轻松, 1=紧张
        trend_score = (avg_cpu / 100 * 0.5 + avg_mem / 100 * 0.5)
        
        return {
            "avg_cpu": avg_cpu,
            "avg_memory": avg_mem,
            "samples": len(recent),
            "trend_score": trend_score,
            "status": "normal" if trend_score < 0.7 else "high_load",
        }


class AdaptiveController:
    """
    自适应控制器 v2.0 — PID闭环调节
    
    基于工程控制论第17章（自适应系统）+ PID反馈控制（Ch.4）
    
    改进: 从bang-bang开关控制升级为连续PID调节，消除稳态误差
    """
    
    def __init__(self):
        self.parameters: Dict[str, float] = {
            "learning_rate": 0.1,
            "exploration_rate": 0.1,
            "discount_factor": 0.9,
        }
        self.adaptation_history: List[Dict] = []
        
        # PID状态
        self._integral_error: float = 0.0
        self._last_error: float = 0.0
        self._kp: float = 0.8   # 比例系数
        self._ki: float = 0.15  # 积分系数
        self._kd: float = 0.3   # 微分系数
        self._integral_limit: float = 3.0  # 抗积分饱和
        
        # 参数锁定状态（持久化，多场景支持）
        self._profiles: Dict[str, Dict[str, float]] = {}  # profile_name -> params
        self._active_profile: Optional[str] = None
        self._lock_timestamps: Dict[str, float] = {}
        self._locked_params_path = Path.home() / ".hermes/evolution/data/locked_params.json"
        self._load_locked_params()
        
    def adapt(self, performance: float, target: float = 0.8) -> Dict[str, float]:
        """
        自适应调整参数（PID版）
        
        Args:
            performance: 当前性能
            target: 目标性能
            
        Returns:
            Dict: 调整后的参数
        """
        error = target - performance
        
        # PID计算
        self._integral_error += error
        self._integral_error = max(-self._integral_limit, 
                                   min(self._integral_limit, self._integral_error))
        derivative = error - self._last_error
        self._last_error = error
        
        pid_signal = self._kp * error + self._ki * self._integral_error + self._kd * derivative
        
        # PID信号映射到参数调整（以1.0为基准，正负微调）
        # learning_rate: 性能差→提高学习速度，性能好→降低避免震荡
        lr_adjustment = 1.0 + pid_signal * 0.4
        new_lr = max(0.01, min(0.5, self.parameters["learning_rate"] * lr_adjustment))
        
        # exploration_rate: 性能差→多探索，性能好→减少探索利用已知
        if error > 0:
            exp_adjustment = 1.0 + abs(pid_signal) * 0.3
        else:
            exp_adjustment = 1.0 - abs(pid_signal) * 0.15
        new_exp = max(0.01, min(0.5, self.parameters["exploration_rate"] * exp_adjustment))
        
        # 只有变化有意义时才更新（避免微幅抖动）
        if abs(new_lr - self.parameters["learning_rate"]) > 0.005:
            self.parameters["learning_rate"] = new_lr
        if abs(new_exp - self.parameters["exploration_rate"]) > 0.005:
            self.parameters["exploration_rate"] = new_exp
        
        # 记录历史
        self.adaptation_history.append({
            "timestamp": time.time(),
            "performance": performance,
            "error": error,
            "pid_signal": pid_signal,
            "parameters": self.parameters.copy(),
        })
        
        return self.parameters.copy()
    
    def get_trend(self) -> Dict[str, Any]:
        """获取调节趋势分析"""
        if len(self.adaptation_history) < 3:
            return {"status": "insufficient_data"}
        
        recent = self.adaptation_history[-10:]
        errors = [h["error"] for h in recent]
        
        # 误差趋势
        error_trend = errors[-1] - errors[0]
        
        # 收敛判断
        converged = abs(errors[-1]) < 0.05 and abs(error_trend) < 0.02
        
        return {
            "current_error": errors[-1],
            "error_trend": error_trend,
            "converged": converged,
            "integral_error": self._integral_error,
            "total_adjustments": len(self.adaptation_history),
        }
    
    def lock_parameters(self, profile_name: str = "default") -> bool:
        """
        锁定当前最优参数到指定场景
        
        Args:
            profile_name: 场景名称，如 'normal', 'high_load'
            
        Returns:
            bool: 是否成功锁定
        """
        trend = self.get_trend()
        if not trend.get("converged", False):
            logger.info("参数未收敛，暂不锁定")
            return False
        
        self._profiles[profile_name] = self.parameters.copy()
        self._lock_timestamps[profile_name] = time.time()
        self._active_profile = profile_name
        self._save_locked_params()
        logger.info(f"参数已锁定到场景 '{profile_name}': {self._profiles[profile_name]}")
        return True
    
    def unlock_parameters(self, profile_name: str = None):
        """解锁参数，允许继续调节"""
        if profile_name:
            self._profiles.pop(profile_name, None)
            self._lock_timestamps.pop(profile_name, None)
            if self._active_profile == profile_name:
                self._active_profile = None
            logger.info(f"场景 '{profile_name}' 参数已解锁")
        else:
            self._profiles.clear()
            self._lock_timestamps.clear()
            self._active_profile = None
            logger.info("所有场景参数已解锁，恢复调节")
        self._save_locked_params()
    
    def switch_profile(self, profile_name: str) -> bool:
        """切换到指定场景的锁定参数"""
        if profile_name not in self._profiles:
            logger.warning(f"场景 '{profile_name}' 没有锁定参数")
            return False
        self._active_profile = profile_name
        self.parameters = self._profiles[profile_name].copy()
        logger.info(f"已切换到场景 '{profile_name}' 的参数: {self.parameters}")
        return True
    
    def auto_switch_profile(self, env_summary: Dict[str, Any]):
        """根据环境摘要自动切换场景"""
        if len(self._profiles) < 2:
            return  # 只有一个场景，无需切换
        
        trend_score = env_summary.get("trend_score", 0.5)
        status = env_summary.get("status", "normal")
        
        # 简单规则：高负载切到 high_load，否则 normal
        target = "high_load" if status == "high_load" or trend_score >= 0.7 else "normal"
        
        if target in self._profiles and self._active_profile != target:
            self.switch_profile(target)
    
    def _save_locked_params(self):
        """将锁定参数持久化到文件（多场景格式）"""
        import json
        try:
            self._locked_params_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 2,
                "active_profile": self._active_profile,
                "profiles": self._profiles,
                "lock_timestamps": self._lock_timestamps,
                "saved_at": time.time(),
            }
            with open(self._locked_params_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"锁定参数保存失败: {e}")
    
    def _load_locked_params(self):
        """从文件加载锁定参数（兼容旧格式）"""
        import json
        if not self._locked_params_path.exists():
            return
        try:
            with open(self._locked_params_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 新格式
            if data.get("version") == 2:
                self._profiles = data.get("profiles", {})
                self._active_profile = data.get("active_profile")
                self._lock_timestamps = data.get("lock_timestamps", {})
                if self._active_profile and self._active_profile in self._profiles:
                    self.parameters = self._profiles[self._active_profile].copy()
                    logger.info(f"已加载场景 '{self._active_profile}' 的锁定参数")
            else:
                # 兼容旧格式 v1
                old_params = data.get("locked_params")
                if old_params:
                    self._profiles["default"] = old_params
                    self._active_profile = "default"
                    self._lock_timestamps["default"] = data.get("lock_timestamp", time.time())
                    self.parameters = old_params.copy()
                    logger.info(f"已从旧格式加载锁定参数到 'default' 场景: {old_params}")
        except Exception as e:
            logger.warning(f"锁定参数加载失败: {e}")
    
    def get_locked_params(self, profile_name: str = None) -> Optional[Dict[str, float]]:
        """获取锁定参数（指定场景或当前激活场景）"""
        if profile_name:
            return self._profiles.get(profile_name)
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None
    
    def apply_locked_params(self, profile_name: str = None) -> bool:
        """应用锁定参数（指定场景或当前激活场景）"""
        params = self.get_locked_params(profile_name)
        if params:
            self.parameters = params.copy()
            logger.info("已应用锁定参数")
            return True
        return False
    
    def should_lock(self) -> bool:
        """判断是否满足锁定条件"""
        trend = self.get_trend()
        return trend.get("converged", False) and len(self.adaptation_history) >= 10
    
    def list_profiles(self) -> List[str]:
        """列出所有已锁定的场景"""
        return list(self._profiles.keys())


class SelfOptimizer:
    """
    自优化器 v3.0 — 修复梯度估计 + 自适应邻域收缩 + 鲁棒评估
    
    基于工程控制论第15章（自寻最优控制）
    
    v3.0改进:
    1. 梯度估计器修复: 每次迭代都估计 + EMA平滑
    2. 邻域收缩自适应: 根据改进率动态调整收缩速度
    3. 扰动稳定性: 多次评估取中位数 + 自适应噪声
    """
    
    def __init__(self):
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score: float = float('-inf')
        self.optimization_history: List[Dict] = []
        
        # 局部搜索状态
        self._current_center: Optional[Dict[str, float]] = None
        self._neighborhood_ratio: float = 1.0  # 邻域比例，逐渐收缩
        self._gradient_estimate: Dict[str, float] = {}
        self._exploration_fraction: float = 0.3  # 前30%迭代全局探索
        
        # v3.0新增: 梯度估计修复 — 保存上一步状态
        self._prev_params: Optional[Dict[str, float]] = None
        self._prev_score: float = float('-inf')
        self._gradient_ema_alpha: float = 0.3  # EMA平滑系数
        
        # v3.0新增: 邻域收缩自适应
        self._min_ratio: float = 0.05  # 最小邻域下限
        self._no_improvement_streak: int = 0  # 连续无改进计数
        
        # v3.0新增: 鲁棒评估
        self._eval_repeats: Dict[str, int] = {
            "explore": 1,   # 探索期快
            "exploit": 2,   # 收缩期稳
            "gradient": 3,  # 精修期准
            "fine_tune": 3,
        }
        
    def optimize(self, objective_func: Callable, 
                 param_ranges: Dict[str, tuple],
                 iterations: int = 10) -> Dict[str, float]:
        """
        自动优化参数 v3.0
        
        策略:
        1. 前30%: 全局随机探索，建立初始最优解（单次评估）
        2. 中40%: 以最优解为中心，自适应收缩邻域（2次评估取中位数）
        3. 后30%: 基于EMA平滑梯度做精细局部搜索（3次评估取中位数）
        """
        import random
        
        # 阶段划分
        explore_end = int(iterations * self._exploration_fraction)
        exploit_end = int(iterations * (1 - 0.3))
        
        for i in range(iterations):
            # 决定采样策略
            if i < explore_end or self._current_center is None:
                # 阶段1: 全局随机探索
                params = self._global_sample(param_ranges)
                strategy = "explore"
            elif i < exploit_end:
                # 阶段2: 邻域收缩采样
                params = self._neighborhood_sample(param_ranges, self._current_center, self._neighborhood_ratio)
                strategy = "exploit"
            else:
                # 阶段3: 梯度引导搜索
                if self._gradient_estimate:
                    params = self._gradient_step(param_ranges, self._current_center, self._gradient_estimate)
                    strategy = "gradient"
                else:
                    params = self._neighborhood_sample(param_ranges, self._current_center, max(self._neighborhood_ratio, 0.1))
                    strategy = "fine_tune"
            
            # 评估（v3.0: 鲁棒评估，多次取中位数）
            score = self._robust_evaluate(objective_func, params, strategy)
            if score is None:
                continue
            
            # v3.0: 每次迭代都更新梯度估计（不只是改进时）
            self._update_gradient_estimate(params, score)
            
            # 更新最优
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.best_params = params.copy()
                self._current_center = params.copy()
                self._no_improvement_streak = 0
            else:
                self._no_improvement_streak += 1
            
            # v3.0: 自适应邻域收缩（只在阶段2）
            if strategy == "exploit":
                self._adapt_neighborhood()
            
            # v3.0: 连续5次无改进，临时扩大邻域跳出局部最优
            if self._no_improvement_streak >= 5 and strategy != "explore":
                self._neighborhood_ratio = min(0.5, self._neighborhood_ratio * 2.0)
                self._no_improvement_streak = 0
                logger.info(f"[SelfOptimizer] 连续5次无改进，扩大邻域到 {self._neighborhood_ratio:.3f} 跳出局部最优")
            
            # 记录历史
            self.optimization_history.append({
                "iteration": i,
                "params": params,
                "score": score,
                "strategy": strategy,
                "improved": improved,
                "neighborhood_ratio": self._neighborhood_ratio,
                "gradient_estimate": self._gradient_estimate.copy(),
            })
            
            logger.debug(f"迭代{i} [{strategy}] 得分={score:.4f} {'↑' if improved else ''} "
                        f"邻域={self._neighborhood_ratio:.3f}")
        
        return self.best_params or {}
    
    def _global_sample(self, param_ranges: Dict[str, tuple]) -> Dict[str, float]:
        """全局随机采样"""
        import random
        return {name: random.uniform(lo, hi) for name, (lo, hi) in param_ranges.items()}
    
    def _neighborhood_sample(self, param_ranges: Dict[str, tuple], 
                             center: Dict[str, float], ratio: float) -> Dict[str, float]:
        """在中心点的邻域内采样（v3.0: 自适应噪声幅度）"""
        import random
        params = {}
        for name, (lo, hi) in param_ranges.items():
            center_val = center.get(name, (lo + hi) / 2)
            span = (hi - lo) * ratio
            # v3.0: 噪声幅度与邻域比例挂钩，避免固定小噪声
            noise_ratio = max(0.02, ratio * 0.1)  # 探索期噪声大，精修期噪声小
            noise = random.uniform(-noise_ratio, noise_ratio) * span
            new_lo = max(lo, center_val - span / 2 + noise)
            new_hi = min(hi, center_val + span / 2 + noise)
            # 确保范围有效
            if new_lo >= new_hi:
                new_lo, new_hi = center_val - span / 2, center_val + span / 2
            params[name] = random.uniform(new_lo, new_hi)
        return params
    
    def _gradient_step(self, param_ranges: Dict[str, tuple],
                       center: Dict[str, float], gradient: Dict[str, float]) -> Dict[str, float]:
        """沿梯度方向走一步（v3.0: 自适应噪声 + 动量感）"""
        import random
        params = {}
        step_size = 0.15 * max(self._neighborhood_ratio, 0.05)  # v3.0: 增大基础步长
        for name, (lo, hi) in param_ranges.items():
            grad = gradient.get(name, 0)
            # v3.0: 噪声随邻域比例自适应
            noise_ratio = max(0.005, self._neighborhood_ratio * 0.05)
            noise = random.uniform(-noise_ratio, noise_ratio) * (hi - lo)
            new_val = center.get(name, 0) + step_size * grad * (hi - lo) + noise
            params[name] = max(lo, min(hi, new_val))
        return params
    
    def _update_gradient_estimate(self, new_params: Dict[str, float], new_score: float):
        """
        v3.0: 每次迭代都更新梯度估计（修复v2.0只在改进时估计的问题）
        
        用EMA平滑，避免单次噪声干扰
        """
        if self._prev_params is None:
            self._prev_params = new_params.copy()
            self._prev_score = new_score
            return
        
        delta_score = new_score - self._prev_score
        
        for name in new_params:
            delta_param = new_params.get(name, 0) - self._prev_params.get(name, 0)
            if abs(delta_param) < 1e-8:
                continue
            
            # 有限差分估计梯度
            raw_gradient = delta_score / delta_param
            
            # EMA平滑
            old_grad = self._gradient_estimate.get(name, 0)
            smoothed = self._gradient_ema_alpha * raw_gradient + (1 - self._gradient_ema_alpha) * old_grad
            self._gradient_estimate[name] = smoothed
        
        # 更新上一步状态
        self._prev_params = new_params.copy()
        self._prev_score = new_score
    
    def _get_improvement_rate(self, window: int = 10) -> float:
        """计算最近window次迭代的改进率"""
        if len(self.optimization_history) < 2:
            return 0.0
        recent = self.optimization_history[-window:]
        improvements = sum(1 for h in recent if h.get("improved", False))
        return improvements / len(recent)
    
    def _adapt_neighborhood(self):
        """
        v3.0: 自适应邻域收缩
        
        根据改进率动态调整收缩速度：
        - 改进频繁(>30%) → 可以快收缩(0.90)
        - 中等改进(10-30%) → 正常收缩(0.95)
        - 改进稀少(<10%) → 慢收缩甚至扩大(1.05)
        """
        rate = self._get_improvement_rate(window=10)
        
        if rate > 0.3:
            shrink = 0.90
        elif rate > 0.1:
            shrink = 0.95
        else:
            shrink = 1.05  # 扩大探索范围
        
        self._neighborhood_ratio *= shrink
        self._neighborhood_ratio = max(self._min_ratio, min(1.0, self._neighborhood_ratio))
    
    def _robust_evaluate(self, objective_func: Callable, params: Dict[str, float], 
                         strategy: str) -> Optional[float]:
        """
        v3.0: 鲁棒评估 — 多次评估取中位数
        
        不同阶段评估次数不同：
        - explore: 1次（快）
        - exploit: 2次（稳）
        - gradient/fine_tune: 3次（准）
        """
        repeats = self._eval_repeats.get(strategy, 2)
        scores = []
        
        for _ in range(repeats):
            try:
                s = objective_func(params)
                scores.append(s)
            except Exception as e:
                logger.error(f"目标函数评估失败: {e}")
                continue
        
        if not scores:
            return None
        
        # 取中位数（比均值抗噪）
        scores.sort()
        median = scores[len(scores) // 2]
        
        if repeats > 1:
            std = (max(scores) - min(scores)) / 2 if len(scores) > 1 else 0
            logger.debug(f"[鲁棒评估] {repeats}次评估: {scores} → 中位数={median:.4f}, 半极差={std:.4f}")
        
        return median
    
    def get_optimization_report(self) -> Dict:
        """v3.0: 获取优化过程报告"""
        if not self.optimization_history:
            return {"status": "no_data"}
        
        strategies = {}
        for h in self.optimization_history:
            s = h["strategy"]
            if s not in strategies:
                strategies[s] = {"count": 0, "improvements": 0, "avg_score": 0.0}
            strategies[s]["count"] += 1
            if h["improved"]:
                strategies[s]["improvements"] += 1
            strategies[s]["avg_score"] += h["score"]
        
        for s in strategies:
            strategies[s]["avg_score"] /= strategies[s]["count"]
            strategies[s]["improvement_rate"] = strategies[s]["improvements"] / strategies[s]["count"]
        
        return {
            "total_iterations": len(self.optimization_history),
            "best_score": self.best_score,
            "final_neighborhood_ratio": self._neighborhood_ratio,
            "gradient_estimate": self._gradient_estimate,
            "strategy_breakdown": strategies,
            "no_improvement_streak": self._no_improvement_streak,
        }
