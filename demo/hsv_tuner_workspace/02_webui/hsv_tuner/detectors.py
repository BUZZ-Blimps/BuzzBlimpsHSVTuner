from __future__ import annotations

import math
import os
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import CustomObjectConfig, HSVRange, ToolConfig


def _ensure_blimp_vision_on_path() -> None:
    candidates: list[Path] = []
    env_path = os.environ.get("BLIMP_VISION_PATH")
    if env_path:
        env = Path(env_path).expanduser().resolve()
        candidates.append(env)
        candidates.append(env / "blimp_vision")

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "blimp_vision")
        candidates.append(parent)

    for candidate in candidates:
        root = candidate.resolve()
        direct_pkg = root / "blob_detector.py"
        nested_pkg = root / "blimp_vision" / "blob_detector.py"
        if not direct_pkg.exists() and not nested_pkg.exists():
            continue
        if direct_pkg.exists():
            path_to_add = str(root.parent)
        else:
            path_to_add = str(root)
        if path_to_add not in sys.path:
            sys.path.insert(0, path_to_add)
        return


try:
    from blimp_vision.blob_detector import BlobDetectorClass
    from blimp_vision import contour_goal_detection as goal_module
except Exception:
    _ensure_blimp_vision_on_path()
    importlib.invalidate_caches()
    if "blimp_vision" in sys.modules:
        del sys.modules["blimp_vision"]
    from blimp_vision.blob_detector import BlobDetectorClass
    from blimp_vision import contour_goal_detection as goal_module


@dataclass
class DetectionResult:
    label: str
    bbox: Tuple[float, float, float, float]  # center x/y + width/height
    confidence: float
    side: str = "mono"


class BalloonDetectorAdapter:
    def __init__(self) -> None:
        self.detector = BlobDetectorClass()

    def configure(self, cfg: ToolConfig) -> None:
        d = self.detector
        d.green_lh = cfg.green_hsv.h_min
        d.green_uh = cfg.green_hsv.h_max
        d.green_ls = cfg.green_hsv.s_min
        d.green_us = cfg.green_hsv.s_max
        d.green_lv = cfg.green_hsv.v_min
        d.green_uv = cfg.green_hsv.v_max

        d.purple_lh = cfg.purple_hsv.h_min
        d.purple_uh = cfg.purple_hsv.h_max
        d.purple_ls = cfg.purple_hsv.s_min
        d.purple_us = cfg.purple_hsv.s_max
        d.purple_lv = cfg.purple_hsv.v_min
        d.purple_uv = cfg.purple_hsv.v_max

        d.include_green = cfg.include_green
        d.include_purple = cfg.include_purple
        d.min_area = cfg.min_area
        d.min_percent_filled = cfg.min_percent_filled
        d.ignore_top_ratio = cfg.ignore_top_ratio

        d.use_kalman = cfg.use_kalman
        d.use_optical_flow = cfg.use_optical_flow
        d.use_lock = cfg.use_lock

    def mask(self, frame: np.ndarray) -> np.ndarray:
        return self.detector._build_balloon_mask(frame)

    def mask_for_color(self, frame: np.ndarray, color: str) -> np.ndarray:
        keep_green = self.detector.include_green
        keep_purple = self.detector.include_purple
        try:
            if color == "green":
                self.detector.include_green = True
                self.detector.include_purple = False
            elif color == "purple":
                self.detector.include_green = False
                self.detector.include_purple = True
            else:
                return np.zeros(frame.shape[:2], dtype=np.uint8)
            return self.detector._build_balloon_mask(frame)
        finally:
            self.detector.include_green = keep_green
            self.detector.include_purple = keep_purple

    def detect(self, frame: np.ndarray, max_detections: int) -> List[DetectionResult]:
        detections = self.detector.contour_find_balls(frame, max_detections=max_detections)
        out: List[DetectionResult] = []
        for det in detections:
            bbox = tuple(float(v) for v in det.bbox)
            out.append(
                DetectionResult(
                    label="balloon",
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    confidence=float(det.confidence),
                )
            )
        return out


class GoalDetectorAdapter:
    def configure(self, cfg: ToolConfig) -> None:
        goal_module.orange_hsv["h_min"] = cfg.goal_orange_hsv.h_min
        goal_module.orange_hsv["h_max"] = cfg.goal_orange_hsv.h_max
        goal_module.orange_hsv["s_min"] = cfg.goal_orange_hsv.s_min
        goal_module.orange_hsv["s_max"] = cfg.goal_orange_hsv.s_max
        goal_module.orange_hsv["v_min"] = cfg.goal_orange_hsv.v_min
        goal_module.orange_hsv["v_max"] = cfg.goal_orange_hsv.v_max

        goal_module.yellow_hsv["h_min"] = cfg.goal_yellow_hsv.h_min
        goal_module.yellow_hsv["h_max"] = cfg.goal_yellow_hsv.h_max
        goal_module.yellow_hsv["s_min"] = cfg.goal_yellow_hsv.s_min
        goal_module.yellow_hsv["s_max"] = cfg.goal_yellow_hsv.s_max
        goal_module.yellow_hsv["v_min"] = cfg.goal_yellow_hsv.v_min
        goal_module.yellow_hsv["v_max"] = cfg.goal_yellow_hsv.v_max

        goal_module.score_threshold = cfg.goal_score_threshold

    def _range_by_color(self, cfg: ToolConfig, color: str) -> HSVRange:
        if color == "yellow":
            return cfg.goal_yellow_hsv
        return cfg.goal_orange_hsv

    def mask(self, frame: np.ndarray, cfg: ToolConfig, color: str) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r = self._range_by_color(cfg, color)
        lower = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
        upper = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask, kernel, iterations=3)

    def detect(self, frame: np.ndarray, cfg: ToolConfig, color: str, max_detections: int) -> List[DetectionResult]:
        detections = goal_module.contour_find_goals(
            frame,
            yellow_goal_mode=(color == "yellow"),
            max_detections=max_detections,
        )
        out: List[DetectionResult] = []
        for det in detections:
            bbox = tuple(float(v) for v in det.bbox)
            out.append(
                DetectionResult(
                    label=f"goal_{color}",
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    confidence=float(det.confidence),
                )
            )
        return out


class CustomHSVDetector:
    def detect_and_mask(
        self,
        frame: np.ndarray,
        object_cfg: CustomObjectConfig,
        max_detections: int,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r = object_cfg.hsv
        lower = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
        upper = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[float, DetectionResult]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < object_cfg.min_area:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            enclosing_area = math.pi * radius * radius
            if enclosing_area <= 1e-6:
                continue

            fill = (area / enclosing_area) * 100.0
            if fill < object_cfg.min_fill:
                continue
            if cy <= frame.shape[0] * object_cfg.ignore_top_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            det = DetectionResult(
                label=object_cfg.name,
                bbox=(x + w / 2.0, y + h / 2.0, float(w), float(h)),
                confidence=float(min(1.0, fill / 100.0)),
            )
            candidates.append((area, det))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return mask, [d for _, d in candidates[: max(1, int(max_detections))]]


class DetectorSuite:
    def __init__(self) -> None:
        self.balloon = BalloonDetectorAdapter()
        self.goal = GoalDetectorAdapter()
        self.custom = CustomHSVDetector()

    def configure(self, cfg: ToolConfig) -> None:
        self.balloon.configure(cfg)
        self.goal.configure(cfg)

    def detect_custom_objects_with_masks(self, frame: np.ndarray, cfg: ToolConfig) -> Tuple[dict[str, np.ndarray], List[DetectionResult]]:
        masks: dict[str, np.ndarray] = {}
        detections: List[DetectionResult] = []
        for obj in cfg.custom_objects:
            if not obj.enabled:
                continue
            mask, dets = self.custom.detect_and_mask(frame, obj, cfg.max_bboxes)
            object_id = obj.object_id or obj.name or "custom"
            masks[object_id] = mask
            detections.extend(dets)
        return masks, detections


def draw_detections(frame: np.ndarray, detections: Sequence[DetectionResult]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x, y, w, h = [int(v) for v in det.bbox]
        p1 = (x - w // 2, y - h // 2)
        p2 = (x + w // 2, y + h // 2)

        if det.label.startswith("goal"):
            color = (0, 200, 255)
        elif det.label == "balloon":
            color = (0, 255, 0)
        else:
            color = (255, 200, 0)

        cv2.rectangle(out, p1, p2, color, 2)
        cv2.circle(out, (x, y), 3, color, -1)
        txt = f"{det.label} {det.confidence:.2f}"
        cv2.putText(out, txt, (p1[0], max(10, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out
