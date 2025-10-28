from typing import Optional, Tuple
from src.utils.schemas import Pose, Perception, Control
from src.memory import Memory
from src.navigator import Navigator
from src.controller import Controller

class Brain:
    def __init__(self):
        self.mem = Memory()
        self.nav = Navigator()
        self.ctrl = Controller()
        self.pose = Pose(0.0, 0.0, 0.0)
        self.goal = None
        self.last_instruction = None
    def on_asr(self, evt: dict):
        if evt.get("type")!="utterance":
            return
        self.last_instruction = evt["text"]
    def _infer_goal_from_perception(self, cur_pose: Pose, p: Perception):
        if p.intent!="navigate":
            return
        if p.target is None and p.rel_dir is None and p.depth_m is None:
            return
        gx = cur_pose.x + (p.depth_m if p.depth_m is not None else 1.0)
        gy = cur_pose.y
        name = p.target or "target"
        self.mem.remember_object(name, (gx, gy))
        self.goal = (gx, gy)
    def step_perception(self, cur_pose: Pose, p: Perception) -> Optional[Control]:
        self.pose = cur_pose
        if self.goal is None:
            self._infer_goal_from_perception(cur_pose, p)
        c = None
        if self.goal:
            c = self.nav.control_to(self.pose, self.goal)
            if c:
                self.ctrl.send(c)
        self.mem.remember_pose(self.pose)
        return c
