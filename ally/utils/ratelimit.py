import time
from collections import defaultdict
from typing import Dict

class TokenBucket:
    def __init__(self, calls: int, per_seconds: float):
        self.capacity = calls
        self.refill = calls / per_seconds
        self.tokens: Dict[str, float] = defaultdict(lambda: self.capacity)
        self.last: Dict[str, float] = defaultdict(time.monotonic)

    def acquire(self, key: str) -> None:
        now = time.monotonic()
        elapsed = now - self.last[key]
        self.last[key] = now
        self.tokens[key] = min(self.capacity, self.tokens[key] + elapsed * self.refill)
        if self.tokens[key] < 1.0:
            sleep = (1.0 - self.tokens[key]) / self.refill
            time.sleep(max(0.0, sleep))
            self.tokens[key] = 0.0
        else:
            self.tokens[key] -= 1.0