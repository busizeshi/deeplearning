import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        print(
            f"时间消耗为，{(self.end_time - self.start_time) / 3600}时, {(self.end_time - self.start_time) / 60}分, {(self.end_time - self.start_time)}秒")
