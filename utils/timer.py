import time
import numpy as np


class LoopTimer:
    def __init__(self):
        self.prev_event = None
        self.events = {}
        self.history = {}
        self.count = 100

    def start_loop(self):
        self.prev_event = time.process_time()

    def event(self, name):
        if name in self.events.items():
            raise ValueError(f"Event {name} already happened")
        self.events[name] = (ctime:=time.process_time()) - self.prev_event
        self.prev_event = ctime

    def end_loop(self):
        self.event("END_LOOP")
        for k, v in self.events.items():
            if k not in self.history:
                self.history[k] = []  # means first loop
            if len(self.history[k]) > self.count:
                self.history[k].pop(0)
            self.history[k].append(v)

    def report(self):
        result = {}
        for k, v in self.history.items():
            result[k] = 1000*np.mean(v)
        return result
        
        