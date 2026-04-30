"""
鲁棒API客户端 - 基于工程控制论Ch.11(变参数系统) + Ch.13(最优控制)设计

核心策略：
1. 速率限制(令牌桶) - 符合NVIDIA 40次/分钟限制
2. 指数退避重试 - 网络抖动自适应
3. 熔断降级 - 连续失败自动切换备用模型
4. 动态超时 - 根据历史延迟自适应调整
"""

import time
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置"""
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0
    max_retries: int = 3
    enabled: bool = True


@dataclass
class CallMetrics:
    """单次调用指标"""
    start_time: float
    end_time: float = 0.0
    success: bool = False
    error_type: Optional[str] = None
    tokens_used: int = 0

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0


class TokenBucketRateLimiter:
    """
    令牌桶速率限制器
    
    NVIDIA限制: 40次/分钟 → 每秒0.666个令牌
    桶容量40，允许突发但长期不超过40/min
    """
    
    def __init__(self, capacity: int = 40, refill_per_minute: float = 40.0):
        self.capacity = capacity
        self.tokens = float(capacity)  # 当前令牌数
        self.refill_rate = refill_per_minute / 60.0  # 每秒补充数
        self.last_refill = time.time()
        self._lock = Lock()
        
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """获取令牌，blocking=True时会等待"""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            if not blocking:
                return False
                
            # 需要等待多久
            need = tokens - self.tokens
            wait_time = need / self.refill_rate
            if timeout is not None and wait_time > timeout:
                return False
                
        if blocking:
            time.sleep(wait_time)
            return self.acquire(tokens, blocking=False)
        return False
    
    def get_stats(self) -> Dict[str, float]:
        """获取当前状态"""
        with self._lock:
            self._refill()
            return {
                "tokens_available": round(self.tokens, 2),
                "capacity": self.capacity,
                "refill_rate_per_sec": round(self.refill_rate, 3),
                "utilization": round((self.capacity - self.tokens) / self.capacity * 100, 1),
            }


class CircuitBreaker:
    """
    熔断器 - 控制论Ch.12 继电系统类比
    
    状态机:
    CLOSED(正常) → OPEN(熔断) → HALF_OPEN(半开试探) → CLOSED/OPEN
    
    触发条件: 连续失败5次进入OPEN
    恢复条件: OPEN后30秒进入HALF_OPEN，试一次成功则CLOSED
    """
    
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = Lock()
        
    def record_success(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_CLOSED
                self.failure_count = 0
                logger.info("[熔断器] 半开状态恢复成功 → CLOSED")
            elif self.state == self.STATE_CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                
    def record_failure(self) -> bool:
        """记录失败，返回是否触发熔断"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                logger.warning(f"[熔断器] 半开状态再次失败 → OPEN (连续失败{self.failure_count}次)")
                return True
                
            if self.failure_count >= self.failure_threshold:
                self.state = self.STATE_OPEN
                logger.warning(f"[熔断器] 连续失败{self.failure_count}次 → OPEN (熔断{self.recovery_timeout}s)")
                return True
            return False
            
    def can_execute(self) -> bool:
        """检查是否允许执行"""
        with self._lock:
            if self.state == self.STATE_CLOSED:
                return True
            if self.state == self.STATE_OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    logger.info("[熔断器] 冷却完成 → HALF_OPEN (试探一次)")
                    return True
                return False
            if self.state == self.STATE_HALF_OPEN:
                return True
            return False
            
    def get_state(self) -> str:
        with self._lock:
            return self.state


class AdaptiveTimeout:
    """
    自适应超时 - 基于历史延迟动态调整(控制论Ch.11 变参数)
    
    超时 = 历史P90延迟 * 安全系数(默认2.0)
    上限120秒，下限10秒
    """
    
    def __init__(self, safety_factor: float = 2.0, min_timeout: float = 10.0, max_timeout: float = 120.0, window_size: int = 20):
        self.safety_factor = safety_factor
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.latencies: deque = deque(maxlen=window_size)
        
    def record(self, latency: float, success: bool):
        if success and latency > 0:
            self.latencies.append(latency)
            
    def get_timeout(self) -> float:
        if len(self.latencies) < 3:
            return 45.0  # 初始保守值
        
        sorted_latencies = sorted(self.latencies)
        p90_idx = int(len(sorted_latencies) * 0.9)
        p90 = sorted_latencies[min(p90_idx, len(sorted_latencies) - 1)]
        
        timeout = p90 * self.safety_factor
        return max(self.min_timeout, min(self.max_timeout, timeout))
    
    def get_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {"count": 0, "current_timeout": 45.0}
        return {
            "count": len(self.latencies),
            "p50": round(sorted(self.latencies)[len(self.latencies)//2], 2),
            "p90": round(sorted(self.latencies)[int(len(self.latencies)*0.9)], 2),
            "current_timeout": round(self.get_timeout(), 2),
        }


class ResilientAPIClient:
    """
    鲁棒API客户端 - 整合速率限制、重试、熔断、降级
    
    使用方式:
        client = ResilientAPIClient(
            primary=APIConfig("nvidia", "https://integrate.api.nvidia.com/v1", "key", "z-ai/glm-5.1"),
            fallback=APIConfig("nous", "https://inference-api.nousresearch.com/v1", "ollama", "moonshotai/kimi-k2.6"),
            rate_limit=TokenBucketRateLimiter(capacity=40, refill_per_minute=40),
        )
        result = client.chat_completion(messages=[...])
    """
    
    def __init__(
        self,
        primary: APIConfig,
        fallback: Optional[APIConfig] = None,
        rate_limit: Optional[TokenBucketRateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        adaptive_timeout: Optional[AdaptiveTimeout] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.rate_limit = rate_limit or TokenBucketRateLimiter()
        self.circuit = circuit_breaker or CircuitBreaker()
        self.timeout_mgr = adaptive_timeout or AdaptiveTimeout()
        self.metrics: deque = deque(maxlen=100)
        
    def _call_api(self, config: APIConfig, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """底层API调用"""
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{config.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
            
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断是否可重试的错误"""
        if isinstance(error, urllib.error.HTTPError):
            # 429=限流, 503=服务不可用, 504=网关超时, 502=坏网关
            return error.code in (429, 502, 503, 504)
        if isinstance(error, TimeoutError):
            return True
        return False
        
    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 500,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，自动处理重试和降级
        
        Returns:
            {"success": bool, "content": str, "model": str, "provider": str,
             "latency": float, "tokens": dict, "used_fallback": bool,
             "error": Optional[str]}
        """
        # 构建payload
        payload = {
            "model": self.primary.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
            
        # 速率限制等待
        if not self.rate_limit.acquire(blocking=True, timeout=60.0):
            logger.warning("[速率限制] 等待令牌超时，准备降级")
            if use_fallback and self.fallback:
                return self._fallback_call(messages, temperature, max_tokens, tools, tool_choice, reason="rate_limit_timeout")
            raise RuntimeError("速率限制：无法获取调用令牌")
            
        # 检查熔断器
        if not self.circuit.can_execute():
            logger.warning("[熔断器] 当前OPEN状态，直接降级")
            if use_fallback and self.fallback:
                return self._fallback_call(messages, temperature, max_tokens, tools, tool_choice, reason="circuit_open")
            raise RuntimeError("熔断器开启中，请稍后重试")
            
        # 主API调用（带重试）
        last_error = None
        for attempt in range(self.primary.max_retries):
            timeout = self.timeout_mgr.get_timeout() * (2 ** attempt)  # 指数增长: 45s, 90s, 180s
            timeout = min(timeout, 180.0)  # 硬上限3分钟
            
            start = time.time()
            try:
                logger.info(f"[API调用] {self.primary.name} 第{attempt+1}次尝试，超时{timeout:.0f}s")
                result = self._call_api(self.primary, payload, timeout)
                latency = time.time() - start
                
                # 记录成功
                self.circuit.record_success()
                self.timeout_mgr.record(latency, success=True)
                self.metrics.append(CallMetrics(start_time=start, end_time=time.time(), success=True, tokens_used=result.get('usage', {}).get('total_tokens', 0)))
                
                content = result['choices'][0]['message'].get('content', '')
                tool_calls = result['choices'][0]['message'].get('tool_calls')
                
                return {
                    "success": True,
                    "content": content,
                    "tool_calls": tool_calls,
                    "model": result.get('model', self.primary.model),
                    "provider": self.primary.name,
                    "latency": round(latency, 2),
                    "tokens": result.get('usage', {}),
                    "used_fallback": False,
                    "error": None,
                }
                
            except Exception as e:
                latency = time.time() - start
                self.timeout_mgr.record(latency, success=False)
                last_error = e
                error_type = type(e).__name__
                
                if isinstance(e, urllib.error.HTTPError):
                    error_type = f"HTTP{e.code}"
                    
                logger.warning(f"[API调用] {self.primary.name} 第{attempt+1}次失败: {error_type} ({latency:.1f}s)")
                
                if not self._is_retryable_error(e):
                    # 非可重试错误（如401认证失败），直接熔断
                    self.circuit.record_failure()
                    break
                    
                if attempt < self.primary.max_retries - 1:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"[API调用] {backoff}秒后重试...")
                    time.sleep(backoff)
                    
        # 全部重试失败
        self.circuit.record_failure()
        self.metrics.append(CallMetrics(start_time=start, end_time=time.time(), success=False, error_type=error_type))
        
        # 尝试降级
        if use_fallback and self.fallback:
            return self._fallback_call(messages, temperature, max_tokens, tools, tool_choice, reason=f"primary_failed_after_{self.primary.max_retries}_retries: {error_type}")
            
        return {
            "success": False,
            "content": "",
            "model": self.primary.model,
            "provider": self.primary.name,
            "latency": round(time.time() - start, 2),
            "tokens": {},
            "used_fallback": False,
            "error": str(last_error),
        }
        
    def _fallback_call(
        self, messages, temperature, max_tokens, tools, tool_choice, reason: str
    ) -> Dict[str, Any]:
        """备用模型调用"""
        if not self.fallback:
            raise RuntimeError("无备用模型配置")
            
        payload = {
            "model": self.fallback.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
            
        start = time.time()
        try:
            logger.info(f"[降级] 切换到备用模型 {self.fallback.name}/{self.fallback.model}，原因: {reason}")
            result = self._call_api(self.fallback, payload, timeout=60.0)
            latency = time.time() - start
            
            content = result['choices'][0]['message'].get('content', '')
            tool_calls = result['choices'][0]['message'].get('tool_calls')
            
            return {
                "success": True,
                "content": content,
                "tool_calls": tool_calls,
                "model": result.get('model', self.fallback.model),
                "provider": self.fallback.name,
                "latency": round(latency, 2),
                "tokens": result.get('usage', {}),
                "used_fallback": True,
                "fallback_reason": reason,
                "error": None,
            }
        except Exception as e:
            logger.error(f"[降级] 备用模型也失败: {e}")
            return {
                "success": False,
                "content": "",
                "model": self.fallback.model,
                "provider": self.fallback.name,
                "latency": round(time.time() - start, 2),
                "tokens": {},
                "used_fallback": True,
                "fallback_reason": reason,
                "error": f"备用模型失败: {e}",
            }
            
    def get_health_report(self) -> Dict[str, Any]:
        """获取客户端健康报告"""
        total = len(self.metrics)
        successes = sum(1 for m in self.metrics if m.success)
        failures = total - successes
        avg_latency = sum(m.latency for m in self.metrics if m.success) / max(successes, 1)
        
        return {
            "primary": self.primary.name,
            "fallback": self.fallback.name if self.fallback else None,
            "circuit_state": self.circuit.get_state(),
            "rate_limit": self.rate_limit.get_stats(),
            "adaptive_timeout": self.timeout_mgr.get_stats(),
            "total_calls": total,
            "success_count": successes,
            "failure_count": failures,
            "success_rate": f"{successes/max(total,1)*100:.1f}%",
            "avg_latency_sec": round(avg_latency, 2),
            "recent_errors": [m.error_type for m in list(self.metrics)[-5:] if not m.success],
        }
