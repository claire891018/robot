import time
from typing import Dict, Tuple, Optional, List
from src.utils.schemas import Pose

class Memory:
    def __init__(self):
        self.visited: List[Pose] = []
        self.objects: Dict[str, Tuple[float,float,float]] = {}
    def remember_pose(self, pose: Pose):
        self.visited.append(pose)
    def remember_object(self, name: str, pos: Tuple[float,float]):
        self.objects[name] = (pos[0], pos[1], time.time())
    def recall_object(self, name: str) -> Optional[Tuple[float,float,float]]:
        return self.objects.get(name)
    def recent_path(self, n: int = 200):
        return self.visited[-n:]
