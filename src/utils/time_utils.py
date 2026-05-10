import time
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_iso_timestamp() -> str:
    return datetime.now().isoformat()

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
