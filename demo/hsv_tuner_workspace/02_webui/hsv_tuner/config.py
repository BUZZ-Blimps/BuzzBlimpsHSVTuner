from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def clamp(self) -> None:
        self.h_min = max(0, min(179, int(self.h_min)))
        self.h_max = max(0, min(179, int(self.h_max)))
        self.s_min = max(0, min(255, int(self.s_min)))
        self.s_max = max(0, min(255, int(self.s_max)))
        self.v_min = max(0, min(255, int(self.v_min)))
        self.v_max = max(0, min(255, int(self.v_max)))


@dataclass
class CustomObjectConfig:
    object_id: str = ""
    name: str = "custom"
    detector_kind: str = "custom_hsv"
    enabled: bool = False
    hsv: HSVRange = field(default_factory=lambda: HSVRange(0, 179, 0, 255, 0, 255))
    min_area: int = 250
    min_fill: float = 20.0
    ignore_top_ratio: float = 0.0
    color_bgr: List[int] = field(default_factory=lambda: [255, 255, 0])


def _default_video_path() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "demo" / "video1.avi")


@dataclass
class ToolConfig:
    # Source options
    source_mode: str = "video"  # image | video | camera
    image_path: str = ""
    video_path: str = field(default_factory=_default_video_path)
    camera_index: int = 0
    loop_video: bool = True

    # Stereo routing
    stereo_sbs: bool = True
    camera_side: str = "both"  # left | right | both

    # Detection target mode
    target_mode: str = "all"  # balloon | goal | all | custom
    goal_color: str = "orange"  # orange | yellow
    max_bboxes: int = 5

    # Balloon params (from original blob detector defaults)
    green_hsv: HSVRange = field(default_factory=lambda: HSVRange(39, 76, 78, 153, 58, 219))
    purple_hsv: HSVRange = field(default_factory=lambda: HSVRange(106, 133, 30, 189, 62, 180))
    include_green: bool = True
    include_purple: bool = True
    min_area: int = 250
    min_percent_filled: float = 60.0
    ignore_top_ratio: float = 0.0

    # Balloon tracking toggles
    use_kalman: bool = False
    use_optical_flow: bool = False
    use_lock: bool = False

    # Goal params (from contour goal defaults)
    goal_orange_hsv: HSVRange = field(default_factory=lambda: HSVRange(0, 25, 124, 255, 240, 255))
    goal_yellow_hsv: HSVRange = field(default_factory=lambda: HSVRange(27, 37, 107, 255, 197, 255))
    include_goal_orange: bool = True
    include_goal_yellow: bool = True
    goal_score_threshold: float = 0.2

    # Extensibility for additional HSV objects
    custom_objects: List[CustomObjectConfig] = field(default_factory=list)

    def clamp(self) -> None:
        self.source_mode = self.source_mode if self.source_mode in {"image", "video", "camera"} else "video"
        self.camera_side = self.camera_side if self.camera_side in {"left", "right", "both"} else "both"
        self.target_mode = self.target_mode if self.target_mode in {"balloon", "goal", "all", "custom"} else "all"
        self.goal_color = self.goal_color if self.goal_color in {"orange", "yellow"} else "orange"
        self.max_bboxes = max(1, min(50, int(self.max_bboxes)))
        self.camera_index = int(self.camera_index)
        self.min_area = max(1, int(self.min_area))
        self.min_percent_filled = max(0.0, min(100.0, float(self.min_percent_filled)))
        self.ignore_top_ratio = max(0.0, min(1.0, float(self.ignore_top_ratio)))
        self.goal_score_threshold = max(0.0, min(1.0, float(self.goal_score_threshold)))
        self.green_hsv.clamp()
        self.purple_hsv.clamp()
        self.goal_orange_hsv.clamp()
        self.goal_yellow_hsv.clamp()
        for i, obj in enumerate(self.custom_objects):
            if not obj.object_id:
                obj.object_id = f"custom_{i + 1}"
            obj.hsv.clamp()
            obj.min_area = max(1, int(obj.min_area))
            obj.min_fill = max(0.0, min(100.0, float(obj.min_fill)))
            obj.ignore_top_ratio = max(0.0, min(1.0, float(obj.ignore_top_ratio)))
            if obj.detector_kind not in {"custom_hsv"}:
                obj.detector_kind = "custom_hsv"
            if len(obj.color_bgr) != 3:
                obj.color_bgr = [255, 255, 0]
            obj.color_bgr = [max(0, min(255, int(c))) for c in obj.color_bgr]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ToolConfig":
        cfg = ToolConfig()
        update_config_from_dict(cfg, data)
        return cfg


def _parse_hsv_range(data: Dict[str, Any], fallback: HSVRange) -> HSVRange:
    return HSVRange(
        h_min=data.get("h_min", fallback.h_min),
        h_max=data.get("h_max", fallback.h_max),
        s_min=data.get("s_min", fallback.s_min),
        s_max=data.get("s_max", fallback.s_max),
        v_min=data.get("v_min", fallback.v_min),
        v_max=data.get("v_max", fallback.v_max),
    )


def update_config_from_dict(cfg: ToolConfig, data: Dict[str, Any]) -> None:
    for key in [
        "source_mode", "image_path", "video_path", "camera_index", "loop_video",
        "stereo_sbs", "camera_side", "target_mode", "goal_color", "max_bboxes",
        "include_green", "include_purple", "min_area", "min_percent_filled", "ignore_top_ratio",
        "use_kalman", "use_optical_flow", "use_lock", "goal_score_threshold",
        "include_goal_orange", "include_goal_yellow",
    ]:
        if key in data:
            setattr(cfg, key, data[key])

    if "green_hsv" in data and isinstance(data["green_hsv"], dict):
        cfg.green_hsv = _parse_hsv_range(data["green_hsv"], cfg.green_hsv)
    if "purple_hsv" in data and isinstance(data["purple_hsv"], dict):
        cfg.purple_hsv = _parse_hsv_range(data["purple_hsv"], cfg.purple_hsv)
    if "goal_orange_hsv" in data and isinstance(data["goal_orange_hsv"], dict):
        cfg.goal_orange_hsv = _parse_hsv_range(data["goal_orange_hsv"], cfg.goal_orange_hsv)
    if "goal_yellow_hsv" in data and isinstance(data["goal_yellow_hsv"], dict):
        cfg.goal_yellow_hsv = _parse_hsv_range(data["goal_yellow_hsv"], cfg.goal_yellow_hsv)

    if "custom_objects" in data and isinstance(data["custom_objects"], list):
        objects: List[CustomObjectConfig] = []
        for item in data["custom_objects"]:
            if not isinstance(item, dict):
                continue
            base = CustomObjectConfig()
            if "object_id" in item:
                base.object_id = str(item["object_id"])
            if "name" in item:
                base.name = str(item["name"])
            if "detector_kind" in item:
                base.detector_kind = str(item["detector_kind"])
            if "enabled" in item:
                base.enabled = bool(item["enabled"])
            if "hsv" in item and isinstance(item["hsv"], dict):
                base.hsv = _parse_hsv_range(item["hsv"], base.hsv)
            if "min_area" in item:
                base.min_area = item["min_area"]
            if "min_fill" in item:
                base.min_fill = item["min_fill"]
            if "ignore_top_ratio" in item:
                base.ignore_top_ratio = item["ignore_top_ratio"]
            if "color_bgr" in item and isinstance(item["color_bgr"], list):
                base.color_bgr = item["color_bgr"]
            objects.append(base)
        cfg.custom_objects = objects

    cfg.clamp()


def default_schema() -> Dict[str, Any]:
    return {
        "ranges": {
            "h": [0, 179],
            "s": [0, 255],
            "v": [0, 255],
            "min_area": [1, 50000],
            "min_percent_filled": [0, 100],
            "ignore_top_ratio": [0, 1],
            "max_bboxes": [1, 50],
        },
        "enums": {
            "source_mode": ["image", "video", "camera"],
            "camera_side": ["left", "right", "both"],
            "target_mode": ["balloon", "goal", "all", "custom"],
            "goal_color": ["orange", "yellow"],
        },
    }
