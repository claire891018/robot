import math
from typing import Optional, Tuple
from src.utils.schemas import Pose, Control
from src.utils.config import KP_V, KP_W, GOAL_TOL, MAX_V, MAX_W

class Navigator:
    def __init__(self):
        self.kv = KP_V
        self.kw = KP_W
        self.tol = GOAL_TOL
    def control_to(self, cur: Pose, goal: Tuple[float,float]) -> Optional[Control]:
        dx = goal[0] - cur.x
        dy = goal[1] - cur.y
        d = (dx*dx + dy*dy) ** 0.5
        if d < self.tol:
            return Control(0.0, 0.0)
        ang = math.atan2(dy, dx) - cur.theta
        while ang > math.pi: ang -= 2*math.pi
        while ang < -math.pi: ang += 2*math.pi
        v = max(0.0, min(self.kv * d, MAX_V))
        w = max(-MAX_W, min(self.kw * ang, MAX_W))
        return Control(v, w)
