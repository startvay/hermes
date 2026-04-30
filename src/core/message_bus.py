"""
消息总线
Agent间通信的基础设施，基于发布-订阅模式
"""

from typing import Any, Callable, Dict, List
from queue import Queue
from threading import Lock, Thread
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Message:
    """消息数据结构"""
    
    def __init__(self, event_type: str, data: Any, source: str = "unknown"):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.message_id = f"msg_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"
    
    def __repr__(self) -> str:
        return f"<Message type={self.event_type} source={self.source}>"


class MessageBus:
    """
    消息总线
    
    基于发布-订阅模式，实现Agent间解耦通信
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: Queue = Queue()
        self.lock = Lock()
        self.running = False
        self.processing_thread: Optional[Thread] = None
        self.message_history: List[Message] = []
        
        logger.info("消息总线已初始化")
    
    def subscribe(self, event_type: str, handler: Callable):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
            logger.info(f"已订阅事件: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        with self.lock:
            if event_type in self.subscribers:
                if handler in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(handler)
                    logger.info(f"已取消订阅事件: {event_type}")
    
    def publish(self, event_type: str, data: Any, source: str = "unknown"):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
        """
        message = Message(event_type, data, source)
        self.message_queue.put(message)
        logger.debug(f"事件已发布: {message}")
    
    def process_messages(self):
        """处理消息队列"""
        processed = 0
        
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                
                with self.lock:
                    handlers = self.subscribers.get(message.event_type, [])
                
                for handler in handlers:
                    try:
                        handler(message.data)
                    except Exception as e:
                        logger.error(f"处理消息异常: {e}")
                
                # 记录历史
                self.message_history.append(message)
                if len(self.message_history) > 10000:
                    self.message_history = self.message_history[-10000:]
                
                processed += 1
                
            except Exception:
                break
        
        return processed
    
    def start_processing(self, interval: float = 0.1):
        """
        启动消息处理线程
        
        Args:
            interval: 处理间隔（秒）
        """
        if self.running:
            return
        
        self.running = True
        
        def processing_loop():
            import time
            while self.running:
                self.process_messages()
                time.sleep(interval)
        
        self.processing_thread = Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("消息处理线程已启动")
    
    def stop_processing(self):
        """停止消息处理"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("消息处理已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_messages": len(self.message_history),
            "queue_size": self.message_queue.qsize(),
            "subscriber_count": sum(len(handlers) for handlers in self.subscribers.values()),
            "event_types": list(self.subscribers.keys()),
        }


# 全局消息总线实例
_global_bus = None

def get_message_bus() -> MessageBus:
    """获取全局消息总线"""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus
