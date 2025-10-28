from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Perception:
    intent: str
    target: Optional[str]
    rel_dir: Optional[str]
    dist_label: Optional[str]
    bbox: Optional[Tuple[int,int,int,int]]
    depth_m: Optional[float]

@dataclass
class Pose:
    x: float
    y: float
    theta: float

@dataclass
class Control:
    v: float
    w: float
