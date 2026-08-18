"""熔断中间件（Circuit Breaker）。

解决的问题：
    ModelRetryMiddleware 会在失败后重试（最多 MAX_RETRIES 次），
    ModelFallbackMiddleware 会在失败后切换备用模型继续调用。
    但两者都没有「暂停调用」机制：如果模型服务持续故障，
    每个请求都会白白重试/降级多次，拖垮响应时间。

    熔断在「连续失败达到阈值」后，直接短路：
    在一段时间内不再真正调用模型，直接返回降级提示，
    让服务快速失败（fail fast），等冷却期过后再放行试探。

配合顺序（关键）：
    熔断必须放在中间件链的【最外层】，
    这样才能在熔断打开时拦住对【重试/降级】的整个调用，
    否则重试和降级会在熔断之前把请求耗尽。
    见 agent_factory.py 中 build_agent 的 middleware 列表注释。
"""

import threading
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

# ── 状态常量 ──
_CLOSED = "closed"      # 正常：放行请求
_OPEN = "open"          # 熔断打开：直接拒绝，不调用模型
_HALF_OPEN = "half_open"  # 半开：放行一个试探请求，判断服务是否恢复

_DEFAULT = object()     # 用于区分「没传」和「传了 None」


class CircuitBreakerMiddleware(AgentMiddleware):
    """基于连续失败次数的模型调用熔断中间件。

    三态状态机：
        closed    正常放行；连续失败达到 failure_threshold 后进入 open。
        open      直接短路，返回降级消息；cool_down_seconds 后进入 half_open。
        half_open 放行一个试探请求；成功回到 closed，失败重新回到 open。

    线程安全：使用 threading.Lock 保护计数和状态切换。
    （FastAPI 多线程处理请求，Agent 由 ainvoke 在异步环境调用，
    但状态读写必须加锁，避免竞态。）

    参数：
        failure_threshold: 连续失败多少次后熔断打开（默认 3）。
        cool_down_seconds: 熔断打开后多久进入半开试探（默认 30）。
        on_open_message: 熔断打开时返回给用户的提示消息。
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cool_down_seconds: float = 30.0,
        on_open_message: str = "😥 服务当前繁忙，请稍后再试或转人工客服。",
    ):
        super().__init__()
        self.failure_threshold = failure_threshold
        self.cool_down_seconds = cool_down_seconds
        self.on_open_message = on_open_message

        self._state = _CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    # ── 状态查询（暴露给外部/调试）──
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    # ── 内部状态机 ──
    def _is_open(self) -> bool:
        """判断当前是否处于 open 状态（考虑半开试探）。"""
        if self._state == _CLOSED:
            return False
        if self._state == _OPEN:
            # 冷却期已过，进入半开，放行一个试探请求
            if time.monotonic() - (self._opened_at or 0) >= self.cool_down_seconds:
                self._state = _HALF_OPEN
                return False
            return True
        return False  # half_open：放行试探

    def _record_success(self) -> None:
        """记录一次成功：重置失败计数，回到 closed。"""
        self._consecutive_failures = 0
        self._state = _CLOSED
        self._opened_at = None

    def _record_failure(self) -> None:
        """记录一次失败：累计失败数，达到阈值则打开熔断。"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = _OPEN
            self._opened_at = time.monotonic()

    # ── 中间件钩子（同步 + 异步）──
    def wrap_model_call(self, request, handler):
        # 熔断打开（含冷却期判断）：直接短路，不调用模型
        with self._lock:
            if self._is_open():
                return AIMessage(content=self.on_open_message)

        try:
            result = handler(request)
        except Exception as exc:
            with self._lock:
                self._record_failure()
            raise exc  # 交还给重试/降级中间件处理

        with self._lock:
            self._record_success()
        return result

    async def awrap_model_call(self, request, handler):
        # 熔断打开（含冷却期判断）：直接短路，不调用模型
        with self._lock:
            if self._is_open():
                return AIMessage(content=self.on_open_message)

        try:
            result = await handler(request)
        except Exception as exc:
            with self._lock:
                self._record_failure()
            raise exc  # 交还给重试/降级中间件处理

        with self._lock:
            self._record_success()
        return result


# ── 模块级单例 ──
# build_agent 每个请求都会调用。熔断状态（连续失败次数、开关状态）
# 必须【跨请求共享】才能生效：如果每个请求都新建一个熔断实例，
# 失败计数会在请求之间丢失，熔断永远不会打开。
# 因此这里提供一个进程级单例，所有 Agent 共享同一份熔断状态。
# （多进程部署时每个进程独立计数，属于可接受的近似；如需精确可换 Redis。）
_circuit_breaker_instance: "CircuitBreakerMiddleware | None" = None


def get_circuit_breaker() -> "CircuitBreakerMiddleware":
    """获取全局共享的熔断中间件单例。"""
    global _circuit_breaker_instance
    if _circuit_breaker_instance is None:
        _circuit_breaker_instance = CircuitBreakerMiddleware(
            failure_threshold=3,
            cool_down_seconds=30.0,
            on_open_message="😥 服务当前繁忙，请稍后再试或转人工客服。",
        )
    return _circuit_breaker_instance
