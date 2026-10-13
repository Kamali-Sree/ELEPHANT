import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List

class EventManager:
    def __init__(self):
        self._queues: List[asyncio.Queue] = []

    async def subscribe(self) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        self._queues.append(queue)
        try:
            while True:
                event_type, data = await queue.get()
                payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                yield payload
        finally:
            self.unsubscribe(queue)

    def unsubscribe(self, queue: asyncio.Queue):
        if queue in self._queues:
            self._queues.remove(queue)

    def publish_event(self, event_type: str, data: Dict[str, Any]):
        for queue in self._queues:
            queue.put_nowait((event_type, data))

# Global instance
event_manager = EventManager()
