from src.utils.schemas import Control

class Controller:
    def __init__(self, driver=None):
        self.driver = driver
        self.last = None
    def send(self, c: Control):
        self.last = (c.v, c.w)
        if self.driver is None:
            return
        self.driver.send(c.v, c.w)
