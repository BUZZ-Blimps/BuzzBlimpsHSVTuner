from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import ToolConfig
from .detectors import DetectionResult, DetectorSuite, draw_detections


@dataclass
class PipelineResult:
    raw: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    detections: List[DetectionResult]
    mask_components: Dict[str, np.ndarray]


class VisionPipeline:
    def __init__(self) -> None:
        self.detectors = DetectorSuite()
        self._fps_time = time.time()
        self._fps_count = 0
        self._fps = 0.0

    def _update_fps(self) -> float:
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 0.5:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_time = now
        return self._fps

    def _split_stereo(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        width = frame.shape[1] // 2
        if width <= 0:
            return frame, frame
        return frame[:, :width], frame[:, width : width * 2]

    def _process_single(self, frame: np.ndarray, cfg: ToolConfig, side: str) -> PipelineResult:
        self.detectors.configure(cfg)

        detections: List[DetectionResult] = []
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_components: Dict[str, np.ndarray] = {}

        include_balloon = cfg.target_mode in {"balloon", "all"}
        include_goal = cfg.target_mode in {"goal", "all"}
        include_custom = cfg.target_mode in {"custom", "all"}

        if include_balloon:
            if cfg.include_green:
                gmask = self.detectors.balloon.mask_for_color(frame, "green")
                mask_components["balloon_green"] = gmask
                combined_mask = cv2.bitwise_or(combined_mask, gmask)
            if cfg.include_purple:
                pmask = self.detectors.balloon.mask_for_color(frame, "purple")
                mask_components["balloon_purple"] = pmask
                combined_mask = cv2.bitwise_or(combined_mask, pmask)
            bdet = self.detectors.balloon.detect(frame, cfg.max_bboxes)
            detections.extend(bdet)

        if include_goal:
            if cfg.include_goal_orange:
                gmask_o = self.detectors.goal.mask(frame, cfg, color="orange")
                gdet_o = self.detectors.goal.detect(frame, cfg, color="orange", max_detections=cfg.max_bboxes)
                mask_components["goal_orange"] = gmask_o
                combined_mask = cv2.bitwise_or(combined_mask, gmask_o)
                detections.extend(gdet_o)
            if cfg.include_goal_yellow:
                gmask_y = self.detectors.goal.mask(frame, cfg, color="yellow")
                gdet_y = self.detectors.goal.detect(frame, cfg, color="yellow", max_detections=cfg.max_bboxes)
                mask_components["goal_yellow"] = gmask_y
                combined_mask = cv2.bitwise_or(combined_mask, gmask_y)
                detections.extend(gdet_y)

        if include_custom:
            custom_masks, cdet = self.detectors.detect_custom_objects_with_masks(frame, cfg)
            for object_id, mask in custom_masks.items():
                mask_components[object_id] = mask
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            detections.extend(cdet)

        for det in detections:
            det.side = side

        overlay = draw_detections(frame, detections)
        overlay = self._draw_status(overlay, cfg, side)

        return PipelineResult(
            raw=frame.copy(),
            mask=combined_mask,
            overlay=overlay,
            detections=detections,
            mask_components=mask_components,
        )

    def _draw_status(self, frame: np.ndarray, cfg: ToolConfig, side: str) -> np.ndarray:
        out = frame.copy()
        lines = [
            f"side:{side} mode:{cfg.target_mode} fps:{self._update_fps():.1f}",
            f"source:{cfg.source_mode} stereo:{int(cfg.stereo_sbs)} camera_side:{cfg.camera_side}",
            f"balloon area:{cfg.min_area} fill:{cfg.min_percent_filled:.1f} top_ignore:{cfg.ignore_top_ratio:.2f}",
            f"goals o/y:{int(cfg.include_goal_orange)}/{int(cfg.include_goal_yellow)} score:{cfg.goal_score_threshold:.2f} max_bboxes:{cfg.max_bboxes}",
        ]
        y = 20
        for line in lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 0), 1)
            y += 18
        return out

    def process(self, frame: np.ndarray, cfg: ToolConfig) -> PipelineResult:
        if not cfg.stereo_sbs:
            return self._process_single(frame, cfg, side="mono")

        left, right = self._split_stereo(frame)

        if cfg.camera_side == "left":
            return self._process_single(left, cfg, side="left")

        if cfg.camera_side == "right":
            return self._process_single(right, cfg, side="right")

        left_res = self._process_single(left, cfg, side="left")
        right_res = self._process_single(right, cfg, side="right")

        raw = np.hstack([left_res.raw, right_res.raw])
        mask = np.hstack([left_res.mask, right_res.mask])
        overlay = np.hstack([left_res.overlay, right_res.overlay])
        detections = left_res.detections + right_res.detections
        mask_components: Dict[str, np.ndarray] = {}
        keys = set(left_res.mask_components.keys()) | set(right_res.mask_components.keys())
        for key in keys:
            left_mask = left_res.mask_components.get(key)
            right_mask = right_res.mask_components.get(key)
            if left_mask is None:
                left_mask = np.zeros(left.shape[:2], dtype=np.uint8)
            if right_mask is None:
                right_mask = np.zeros(right.shape[:2], dtype=np.uint8)
            mask_components[key] = np.hstack([left_mask, right_mask])

        return PipelineResult(raw=raw, mask=mask, overlay=overlay, detections=detections, mask_components=mask_components)
