from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from .config import CustomObjectConfig, HSVRange, ToolConfig, default_schema, update_config_from_dict
from .pipeline import PipelineResult, VisionPipeline
from .profiles import ProfileStore, sanitize_name
from .ros_yaml import export_ros_yaml, import_ros_yaml
from .sources import (
    FrameSource,
    FrameSourceError,
    SourceSpec,
    create_source,
)


def _blank_frame(text: str, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv2.putText(frame, text, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return frame


class RuntimeEngine:
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.upload_dir = self.workspace_root / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = ToolConfig()
        self.schema = default_schema()

        self.profile_store = ProfileStore(self.workspace_root / "profiles")
        self.export_dir = self.workspace_root / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.workspace_root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = VisionPipeline()
        self.source: Optional[FrameSource] = None
        self.last_result = PipelineResult(
            raw=_blank_frame("Starting source..."),
            mask=np.zeros((540, 960), dtype=np.uint8),
            overlay=_blank_frame("Starting source..."),
            detections=[],
            mask_components={},
        )
        self.last_error = ""
        self.paused = False
        self.step_once = False
        self._fps_value = 0.0
        self._fps_count = 0
        self._fps_time = time.time()
        self.mask_view_mode = "combined"  # combined | selected
        self.mask_color_overlay = False
        self.mask_selected_object_id = ""

        self._lock = threading.RLock()
        self._running = True

        self._open_source_locked()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            if self.source is not None:
                self.source.release()
                self.source = None

    def _source_spec(self) -> SourceSpec:
        return SourceSpec(
            mode=self.cfg.source_mode,
            image_path=self.cfg.image_path,
            video_path=self.cfg.video_path,
            camera_index=self.cfg.camera_index,
            loop_video=self.cfg.loop_video,
        )

    def _source_key(self) -> tuple:
        return (
            self.cfg.source_mode,
            self.cfg.image_path,
            self.cfg.video_path,
            int(self.cfg.camera_index),
            bool(self.cfg.loop_video),
        )

    def _open_source_locked(self) -> None:
        if self.source is not None:
            self.source.release()
            self.source = None

        try:
            self.source = create_source(self._source_spec())
            self.last_error = ""
        except FrameSourceError as exc:
            self.last_error = str(exc)
            self.last_result = PipelineResult(
                raw=_blank_frame(self.last_error),
                mask=np.zeros((540, 960), dtype=np.uint8),
                overlay=_blank_frame(self.last_error),
                detections=[],
                mask_components={},
            )

    def _run_loop(self) -> None:
        while self._running:
            with self._lock:
                source = self.source
                cfg = ToolConfig.from_dict(self.cfg.to_dict())
                paused = self.paused
                step_once = self.step_once

            if source is None:
                time.sleep(0.1)
                continue
            if paused and not step_once:
                time.sleep(0.03)
                continue

            ok, frame = source.read()
            if not ok or frame is None:
                with self._lock:
                    self.last_error = "No frames from source"
                time.sleep(0.03)
                continue

            result = self.pipeline.process(frame, cfg)
            with self._lock:
                self.last_result = result
                self.last_error = ""
                self.step_once = False
                self._fps_count += 1
                now = time.time()
                elapsed = now - self._fps_time
                if elapsed >= 0.5:
                    self._fps_value = self._fps_count / elapsed
                    self._fps_count = 0
                    self._fps_time = now

            time.sleep(0.001)

    def update_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            before_key = self._source_key()
            update_config_from_dict(self.cfg, patch)
            after_key = self._source_key()
            if before_key != after_key:
                # Switching media/source should immediately resume playback.
                self.paused = False
                self.step_once = False
                self._open_source_locked()
            return self.cfg.to_dict()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": self.cfg.to_dict(),
                "object_registry": self._object_registry_locked(),
                "schema": self.schema,
                "profiles": self.profile_store.list_names(),
                "yaml_files": sorted([p.name for p in self.export_dir.glob("*.yaml")]),
                "error": self.last_error,
                "runtime": self._runtime_locked(),
                "mask_view": self._mask_view_locked(),
                "debug": self._debug_locked(),
            }

    def _runtime_locked(self) -> Dict[str, Any]:
        return {
            "paused": bool(self.paused),
            "fps": float(self._fps_value),
        }

    def _mask_view_locked(self) -> Dict[str, Any]:
        return {
            "mode": self.mask_view_mode,
            "color_overlay": bool(self.mask_color_overlay),
            "selected_object_id": self.mask_selected_object_id,
        }

    def _debug_locked(self) -> Dict[str, Any]:
        result = self.last_result
        return {
            "paused": bool(self.paused),
            "fps": float(self._fps_value),
            "source_mode": self.cfg.source_mode,
            "camera_index": int(self.cfg.camera_index),
            "target_mode": self.cfg.target_mode,
            "camera_side": self.cfg.camera_side,
            "frame_shape": list(result.raw.shape) if result.raw is not None else [],
            "mask_nonzero": int(np.count_nonzero(result.mask)) if result.mask is not None else 0,
            "mask_view": self._mask_view_locked(),
            "detections_count": len(result.detections),
            "detections": [
                {
                    "label": d.label,
                    "bbox": [round(float(v), 2) for v in d.bbox],
                    "confidence": round(float(d.confidence), 3),
                    "side": d.side,
                }
                for d in result.detections[:20]
            ],
            "error": self.last_error,
        }

    def set_runtime(self, paused: Optional[bool] = None, step: bool = False) -> Dict[str, Any]:
        with self._lock:
            if paused is not None:
                self.paused = bool(paused)
            if step:
                self.step_once = True
            return self._runtime_locked()

    def set_mask_view(
        self,
        mode: Optional[str] = None,
        color_overlay: Optional[bool] = None,
        selected_object_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if mode in {"combined", "selected"}:
                self.mask_view_mode = mode
            if color_overlay is not None:
                self.mask_color_overlay = bool(color_overlay)
            if selected_object_id is not None:
                self.mask_selected_object_id = str(selected_object_id)
            return self._mask_view_locked()

    def save_uploaded_source(self, payload: bytes, filename: str, source_mode: str) -> Dict[str, Any]:
        if not payload:
            raise ValueError("Uploaded file is empty")

        raw_name = Path(filename or "upload.bin").name
        stem = sanitize_name(Path(raw_name).stem, default="upload")
        suffix = Path(raw_name).suffix.lower() or ".bin"
        timestamp = int(time.time() * 1000)
        saved_name = f"{stem}_{timestamp}{suffix}"
        saved_path = (self.upload_dir / saved_name).resolve()

        saved_path.write_bytes(payload)

        mode = source_mode if source_mode in {"image", "video"} else "video"
        patch: Dict[str, Any] = {"source_mode": mode}
        if mode == "image":
            patch["image_path"] = str(saved_path)
        else:
            patch["video_path"] = str(saved_path)

        config = self.update_config(patch)
        return {"path": str(saved_path), "config": config}

    def import_uploaded_profile(self, payload: bytes, filename: str) -> Dict[str, Any]:
        if not payload:
            raise ValueError("Uploaded profile is empty")

        try:
            loaded = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON profile: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError("Profile JSON must be an object")

        patch = loaded.get("config") if isinstance(loaded.get("config"), dict) else loaded
        if not isinstance(patch, dict):
            raise ValueError("Profile data does not contain a valid config object")

        config = self.update_config(patch)
        profile_name = sanitize_name(Path(filename or "profile").stem, default="profile")
        saved_name = self.save_profile(profile_name)
        return {"profile_name": saved_name, "config": config}

    def import_uploaded_yaml(self, payload: bytes, filename: str) -> Dict[str, Any]:
        if not payload:
            raise ValueError("Uploaded YAML is empty")

        stem = sanitize_name(Path(filename or "import").stem, default="import")
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            suffix = ".yaml"
        saved_name = f"{stem}_upload{suffix}"
        saved_path = self.export_dir / saved_name
        saved_path.write_bytes(payload)

        config = self.import_yaml(saved_name)
        return {"filename": saved_name, "config": config}

    def _hsv_dict(self, hsv: HSVRange) -> Dict[str, int]:
        return {
            "h_min": int(hsv.h_min),
            "h_max": int(hsv.h_max),
            "s_min": int(hsv.s_min),
            "s_max": int(hsv.s_max),
            "v_min": int(hsv.v_min),
            "v_max": int(hsv.v_max),
        }

    def _object_registry_locked(self) -> list[Dict[str, Any]]:
        cfg = self.cfg
        objects: list[Dict[str, Any]] = [
            {
                "object_id": "balloon_green",
                "name": "Balloon Green",
                "detector_kind": "balloon_green",
                "builtin": True,
                "enabled": bool(cfg.include_green),
                "hsv": self._hsv_dict(cfg.green_hsv),
                "min_area": int(cfg.min_area),
                "min_fill": float(cfg.min_percent_filled),
                "ignore_top_ratio": float(cfg.ignore_top_ratio),
                "color_bgr": [0, 255, 0],
            },
            {
                "object_id": "balloon_purple",
                "name": "Balloon Purple",
                "detector_kind": "balloon_purple",
                "builtin": True,
                "enabled": bool(cfg.include_purple),
                "hsv": self._hsv_dict(cfg.purple_hsv),
                "min_area": int(cfg.min_area),
                "min_fill": float(cfg.min_percent_filled),
                "ignore_top_ratio": float(cfg.ignore_top_ratio),
                "color_bgr": [255, 0, 255],
            },
            {
                "object_id": "goal_orange",
                "name": "Goal Orange",
                "detector_kind": "goal_orange",
                "builtin": True,
                "enabled": bool(cfg.include_goal_orange),
                "hsv": self._hsv_dict(cfg.goal_orange_hsv),
                "score_threshold": float(cfg.goal_score_threshold),
                "color_bgr": [0, 140, 255],
            },
            {
                "object_id": "goal_yellow",
                "name": "Goal Yellow",
                "detector_kind": "goal_yellow",
                "builtin": True,
                "enabled": bool(cfg.include_goal_yellow),
                "hsv": self._hsv_dict(cfg.goal_yellow_hsv),
                "score_threshold": float(cfg.goal_score_threshold),
                "color_bgr": [0, 255, 255],
            },
        ]

        for obj in cfg.custom_objects:
            objects.append(
                {
                    "object_id": obj.object_id,
                    "name": obj.name,
                    "detector_kind": obj.detector_kind,
                    "builtin": False,
                    "enabled": bool(obj.enabled),
                    "hsv": self._hsv_dict(obj.hsv),
                    "min_area": int(obj.min_area),
                    "min_fill": float(obj.min_fill),
                    "ignore_top_ratio": float(obj.ignore_top_ratio),
                    "color_bgr": [int(c) for c in obj.color_bgr],
                }
            )
        return objects

    def _parse_hsv_patch(self, src: Dict[str, Any], base: HSVRange) -> HSVRange:
        return HSVRange(
            h_min=src.get("h_min", base.h_min),
            h_max=src.get("h_max", base.h_max),
            s_min=src.get("s_min", base.s_min),
            s_max=src.get("s_max", base.s_max),
            v_min=src.get("v_min", base.v_min),
            v_max=src.get("v_max", base.v_max),
        )

    def _find_custom_locked(self, object_id: str) -> Optional[CustomObjectConfig]:
        for obj in self.cfg.custom_objects:
            if obj.object_id == object_id:
                return obj
        return None

    def update_object(self, object_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cfg = self.cfg

            if object_id == "balloon_green":
                if "enabled" in patch:
                    cfg.include_green = bool(patch["enabled"])
                if "hsv" in patch and isinstance(patch["hsv"], dict):
                    cfg.green_hsv = self._parse_hsv_patch(patch["hsv"], cfg.green_hsv)
                if "min_area" in patch:
                    cfg.min_area = patch["min_area"]
                if "min_fill" in patch:
                    cfg.min_percent_filled = patch["min_fill"]
                if "ignore_top_ratio" in patch:
                    cfg.ignore_top_ratio = patch["ignore_top_ratio"]
            elif object_id == "balloon_purple":
                if "enabled" in patch:
                    cfg.include_purple = bool(patch["enabled"])
                if "hsv" in patch and isinstance(patch["hsv"], dict):
                    cfg.purple_hsv = self._parse_hsv_patch(patch["hsv"], cfg.purple_hsv)
                if "min_area" in patch:
                    cfg.min_area = patch["min_area"]
                if "min_fill" in patch:
                    cfg.min_percent_filled = patch["min_fill"]
                if "ignore_top_ratio" in patch:
                    cfg.ignore_top_ratio = patch["ignore_top_ratio"]
            elif object_id == "goal_orange":
                if "enabled" in patch:
                    cfg.include_goal_orange = bool(patch["enabled"])
                if "hsv" in patch and isinstance(patch["hsv"], dict):
                    cfg.goal_orange_hsv = self._parse_hsv_patch(patch["hsv"], cfg.goal_orange_hsv)
                if "score_threshold" in patch:
                    cfg.goal_score_threshold = patch["score_threshold"]
            elif object_id == "goal_yellow":
                if "enabled" in patch:
                    cfg.include_goal_yellow = bool(patch["enabled"])
                if "hsv" in patch and isinstance(patch["hsv"], dict):
                    cfg.goal_yellow_hsv = self._parse_hsv_patch(patch["hsv"], cfg.goal_yellow_hsv)
                if "score_threshold" in patch:
                    cfg.goal_score_threshold = patch["score_threshold"]
            else:
                obj = self._find_custom_locked(object_id)
                if obj is None:
                    raise ValueError(f"Unknown object_id: {object_id}")
                if "name" in patch:
                    obj.name = str(patch["name"])[:64]
                if "enabled" in patch:
                    obj.enabled = bool(patch["enabled"])
                if "hsv" in patch and isinstance(patch["hsv"], dict):
                    obj.hsv = self._parse_hsv_patch(patch["hsv"], obj.hsv)
                if "min_area" in patch:
                    obj.min_area = patch["min_area"]
                if "min_fill" in patch:
                    obj.min_fill = patch["min_fill"]
                if "ignore_top_ratio" in patch:
                    obj.ignore_top_ratio = patch["ignore_top_ratio"]
                if "color_bgr" in patch and isinstance(patch["color_bgr"], list):
                    obj.color_bgr = patch["color_bgr"]

            cfg.clamp()
            return {
                "config": cfg.to_dict(),
                "object_registry": self._object_registry_locked(),
            }

    def add_custom_object(self, name: str = "custom") -> Dict[str, Any]:
        with self._lock:
            cfg = self.cfg
            object_id = f"custom_{uuid.uuid4().hex[:8]}"
            display = (name or "custom").strip()[:64] or "custom"
            cfg.custom_objects.append(
                CustomObjectConfig(
                    object_id=object_id,
                    name=display,
                    detector_kind="custom_hsv",
                    enabled=True,
                    hsv=HSVRange(0, 179, 0, 255, 0, 255),
                    min_area=250,
                    min_fill=20.0,
                    ignore_top_ratio=0.0,
                    color_bgr=[255, 255, 0],
                )
            )
            cfg.clamp()
            return {
                "object_id": object_id,
                "state": self.get_state(),
            }

    def duplicate_object(self, object_id: str) -> Dict[str, Any]:
        with self._lock:
            source = None
            for obj in self._object_registry_locked():
                if obj["object_id"] == object_id:
                    source = obj
                    break
            if source is None:
                raise ValueError(f"Unknown object_id: {object_id}")

            new_id = f"custom_{uuid.uuid4().hex[:8]}"
            self.cfg.custom_objects.append(
                CustomObjectConfig(
                    object_id=new_id,
                    name=f"{source['name']}_copy",
                    detector_kind="custom_hsv",
                    enabled=bool(source.get("enabled", True)),
                    hsv=self._parse_hsv_patch(source.get("hsv", {}), HSVRange(0, 179, 0, 255, 0, 255)),
                    min_area=int(source.get("min_area", 250)),
                    min_fill=float(source.get("min_fill", 20.0)),
                    ignore_top_ratio=float(source.get("ignore_top_ratio", 0.0)),
                    color_bgr=[int(c) for c in source.get("color_bgr", [255, 255, 0])],
                )
            )
            self.cfg.clamp()
            return {
                "object_id": new_id,
                "state": self.get_state(),
            }

    def remove_object(self, object_id: str) -> Dict[str, Any]:
        with self._lock:
            before = len(self.cfg.custom_objects)
            self.cfg.custom_objects = [o for o in self.cfg.custom_objects if o.object_id != object_id]
            if len(self.cfg.custom_objects) == before:
                raise ValueError("Only custom objects can be removed")
            self.cfg.clamp()
            return self.get_state()

    def _object_color_map_locked(self) -> Dict[str, tuple[int, int, int]]:
        colors: Dict[str, tuple[int, int, int]] = {
            "balloon_green": (0, 255, 0),
            "balloon_purple": (255, 0, 255),
            "goal_orange": (0, 140, 255),
            "goal_yellow": (0, 255, 255),
        }
        for obj in self.cfg.custom_objects:
            object_id = obj.object_id or obj.name
            if len(obj.color_bgr) == 3:
                colors[object_id] = (int(obj.color_bgr[0]), int(obj.color_bgr[1]), int(obj.color_bgr[2]))
        return colors

    def _render_mask_frame_locked(self) -> np.ndarray:
        result = self.last_result
        components = result.mask_components or {}

        if self.mask_view_mode == "selected" and self.mask_selected_object_id:
            selected_mask = components.get(self.mask_selected_object_id)
            if selected_mask is None:
                selected_mask = np.zeros(result.mask.shape[:2], dtype=np.uint8)
            if not self.mask_color_overlay:
                return cv2.cvtColor(selected_mask, cv2.COLOR_GRAY2BGR)

            out = np.zeros((*selected_mask.shape, 3), dtype=np.uint8)
            color = self._object_color_map_locked().get(self.mask_selected_object_id, (255, 255, 255))
            out[selected_mask > 0] = color
            return out

        if not self.mask_color_overlay:
            return cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)

        out = np.zeros((*result.mask.shape, 3), dtype=np.uint8)
        colors = self._object_color_map_locked()
        for object_id, mask in components.items():
            color = colors.get(object_id, (255, 255, 255))
            pixels = mask > 0
            if np.any(pixels):
                out[pixels] = np.maximum(out[pixels], np.array(color, dtype=np.uint8))
        return out

    def _frame_by_name(self, stream_name: str) -> np.ndarray:
        with self._lock:
            if stream_name == "raw":
                return self.last_result.raw.copy()
            if stream_name == "mask":
                return self._render_mask_frame_locked().copy()
            else:
                return self.last_result.overlay.copy()

    def frame_jpeg(self, stream_name: str, quality: int = 80) -> bytes:
        frame = self._frame_by_name(stream_name)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return b""
        return encoded.tobytes()

    def sample_pixel(self, x: int, y: int) -> Dict[str, Any]:
        with self._lock:
            frame = self.last_result.raw.copy()

        h, w = frame.shape[:2]
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))

        b, g, r = [int(v) for v in frame[y, x]]
        hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]

        return {
            "x": x,
            "y": y,
            "bgr": [b, g, r],
            "hsv": [int(hsv[0]), int(hsv[1]), int(hsv[2])],
        }

    def save_profile(self, name: str) -> str:
        safe = sanitize_name(name)
        with self._lock:
            data = self.cfg.to_dict()
        self.profile_store.save(safe, data)
        return safe

    def load_profile(self, name: str) -> Dict[str, Any]:
        data = self.profile_store.load(name)
        return self.update_config(data)

    def export_yaml(self, name: str, node_name: str = "vision_tuning") -> str:
        safe = sanitize_name(name, default="vision_tuning")
        with self._lock:
            cfg = ToolConfig.from_dict(self.cfg.to_dict())
        out_path = self.export_dir / f"{safe}.yaml"
        export_ros_yaml(cfg, out_path, node_name=node_name)
        return out_path.name

    def import_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.export_dir / filename
        patch = import_ros_yaml(path)
        return self.update_config(patch)

    def save_snapshot(self, name: str) -> str:
        safe = sanitize_name(name, default="snapshot")
        path = self.snapshot_dir / f"{safe}.jpg"
        frame = self._frame_by_name("overlay")
        cv2.imwrite(str(path), frame)
        return path.name


class _AppHandler(BaseHTTPRequestHandler):
    server_version = "HSVTunerHTTP/0.1"

    def log_message(self, format: str, *args) -> None:
        # Keep terminal output focused on runtime status.
        return

    def _json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _raw_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return b""
        return self.rfile.read(content_length)

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, stream_name: str) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        engine: RuntimeEngine = self.server.engine  # type: ignore[attr-defined]

        try:
            while True:
                jpg = engine.frame_jpeg(stream_name)
                if not jpg:
                    time.sleep(0.03)
                    continue

                self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        engine: RuntimeEngine = self.server.engine  # type: ignore[attr-defined]
        request_path = urlparse(self.path).path

        if request_path == "/":
            self._send_html(_INDEX_HTML)
            return

        if request_path == "/api/state":
            self._send_json(engine.get_state())
            return

        if request_path.startswith("/resources/"):
            resource_name = unquote(request_path.removeprefix("/resources/"))
            resource_root = (engine.workspace_root / "resources").resolve()
            candidate = (resource_root / resource_name).resolve()
            if resource_root not in candidate.parents and candidate != resource_root:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid resource path")
                return
            content_type = "application/octet-stream"
            if candidate.suffix.lower() == ".png":
                content_type = "image/png"
            elif candidate.suffix.lower() in {".jpg", ".jpeg"}:
                content_type = "image/jpeg"
            elif candidate.suffix.lower() == ".svg":
                content_type = "image/svg+xml"
            self._send_file(candidate, content_type)
            return

        if request_path == "/stream/raw.mjpg":
            self._send_stream("raw")
            return

        if request_path == "/stream/mask.mjpg":
            self._send_stream("mask")
            return

        if request_path == "/stream/overlay.mjpg":
            self._send_stream("overlay")
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        engine: RuntimeEngine = self.server.engine  # type: ignore[attr-defined]
        request_path = urlparse(self.path).path

        if request_path == "/api/upload_source":
            try:
                payload = self._raw_body()
                filename = self.headers.get("X-Filename", "upload.bin")
                source_mode = self.headers.get("X-Source-Mode", "video")
                out = engine.save_uploaded_source(payload=payload, filename=filename, source_mode=source_mode)
                self._send_json({"ok": True, **out, "state": engine.get_state()})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if request_path == "/api/upload_profile":
            try:
                payload = self._raw_body()
                filename = self.headers.get("X-Filename", "profile.json")
                out = engine.import_uploaded_profile(payload=payload, filename=filename)
                self._send_json({"ok": True, **out, "state": engine.get_state()})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if request_path == "/api/upload_yaml":
            try:
                payload = self._raw_body()
                filename = self.headers.get("X-Filename", "vision_tuning.yaml")
                out = engine.import_uploaded_yaml(payload=payload, filename=filename)
                self._send_json({"ok": True, **out, "state": engine.get_state()})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            body = self._json_body()
        except json.JSONDecodeError as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if request_path == "/api/config":
                config = engine.update_config(body)
                self._send_json({"ok": True, "config": config})
                return

            if request_path == "/api/sample":
                sample = engine.sample_pixel(body.get("x", 0), body.get("y", 0))
                self._send_json({"ok": True, "sample": sample})
                return

            if request_path == "/api/runtime":
                paused = body["paused"] if "paused" in body else None
                step = bool(body.get("step", False))
                runtime = engine.set_runtime(paused=paused, step=step)
                self._send_json({"ok": True, "runtime": runtime, "state": engine.get_state()})
                return

            if request_path == "/api/mask_view":
                mode = body.get("mode") if "mode" in body else None
                color_overlay = body.get("color_overlay") if "color_overlay" in body else None
                selected_object_id = body.get("selected_object_id") if "selected_object_id" in body else None
                mask_view = engine.set_mask_view(mode=mode, color_overlay=color_overlay, selected_object_id=selected_object_id)
                self._send_json({"ok": True, "mask_view": mask_view, "state": engine.get_state()})
                return

            if request_path == "/api/object/update":
                object_id = str(body.get("object_id", ""))
                patch = body.get("patch", {})
                out = engine.update_object(object_id, patch if isinstance(patch, dict) else {})
                self._send_json({"ok": True, **out})
                return

            if request_path == "/api/object/add":
                name = str(body.get("name", "custom"))
                out = engine.add_custom_object(name)
                self._send_json({"ok": True, **out})
                return

            if request_path == "/api/object/duplicate":
                object_id = str(body.get("object_id", ""))
                out = engine.duplicate_object(object_id)
                self._send_json({"ok": True, **out})
                return

            if request_path == "/api/object/remove":
                object_id = str(body.get("object_id", ""))
                state = engine.remove_object(object_id)
                self._send_json({"ok": True, "state": state})
                return

            if request_path == "/api/profile/save":
                name = body.get("name", "profile")
                saved = engine.save_profile(str(name))
                self._send_json({"ok": True, "name": saved, "state": engine.get_state()})
                return

            if request_path == "/api/profile/load":
                name = str(body.get("name", ""))
                config = engine.load_profile(name)
                self._send_json({"ok": True, "config": config, "state": engine.get_state()})
                return

            if request_path == "/api/export/yaml":
                name = str(body.get("name", "vision_tuning"))
                node_name = str(body.get("node_name", "vision_tuning"))
                filename = engine.export_yaml(name=name, node_name=node_name)
                self._send_json({"ok": True, "filename": filename, "state": engine.get_state()})
                return

            if request_path == "/api/import/yaml":
                filename = str(body.get("filename", ""))
                config = engine.import_yaml(filename)
                self._send_json({"ok": True, "config": config, "state": engine.get_state()})
                return

            if request_path == "/api/snapshot":
                name = str(body.get("name", "snapshot"))
                filename = engine.save_snapshot(name)
                self._send_json({"ok": True, "filename": filename})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        except Exception as exc:  # keep handler alive for debugging sessions
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


class HSVTunerServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, engine: RuntimeEngine):
        super().__init__(server_address, RequestHandlerClass)
        self.engine = engine


def run_server(host: str, port: int, workspace_root: Path) -> None:
    engine = RuntimeEngine(workspace_root=workspace_root)
    httpd = HSVTunerServer((host, port), _AppHandler, engine=engine)
    print(f"HSV tuner running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        httpd.server_close()


_INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HSV Tuning Tool</title>
  <style>
    :root {
      --bg: #f2eee4;
      --card: #fffdfa;
      --ink: #1b2124;
      --accent: #0d746d;
      --accent-strong: #0a5e58;
      --muted: #64707d;
      --border: #d7d1c3;
      --panel-tint: #f8f5ed;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body { margin: 0; background: radial-gradient(circle at 18% 0%, #fff8e8 0%, #f6f2e9 35%, var(--bg) 70%); color: var(--ink); font-family: "Avenir Next", "Segoe UI", sans-serif; }
    .layout { height: 100vh; display: grid; grid-template-columns: minmax(430px, 500px) minmax(0, 1fr); gap: 12px; padding: 12px; overflow: hidden; }
    .card { min-height: 0; background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 12px; box-shadow: 0 10px 28px rgba(21, 24, 31, 0.08); }
    .title { margin: 0; font-size: 22px; letter-spacing: 0.4px; line-height: 1.05; }
    .brand-sub { margin-top: 3px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.9px; color: #49616f; }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .brand-logo { width: 42px; height: 42px; object-fit: contain; border-radius: 9px; border: 1px solid var(--border); background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .controls { overflow: auto; }
    .group { border: 1px solid var(--border); border-radius: 11px; padding: 8px; margin-top: 8px; background: linear-gradient(180deg, #fffefb 0%, var(--panel-tint) 100%); }
    details.group > summary { cursor: pointer; user-select: none; color: #124c4a; margin-bottom: 6px; font-weight: 700; }
    details.group[open] > summary { margin-bottom: 8px; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; margin: 6px 0; min-height: 32px; }
    .row label { font-size: 13px; color: #2b3339; }
    .row input[type="range"] { width: 240px; max-width: 100%; }
    .row input[type="text"], .row input[type="number"], .row select { width: 220px; max-width: 100%; border: 1px solid #c8d0d8; border-radius: 8px; background: #fff; color: var(--ink); padding: 6px 8px; }
    .row input[type="range"] { accent-color: var(--accent); }
    input:focus, select:focus, button:focus { outline: none; box-shadow: 0 0 0 3px rgba(13, 116, 109, 0.17); }
    .inline-field { display: flex; gap: 6px; align-items: center; }
    .inline-field input[type="text"] { flex: 1; min-width: 0; width: auto; }
    .slider-with-number { display: flex; gap: 6px; align-items: center; width: 290px; max-width: 100%; }
    .slider-with-number input[type="range"] { flex: 1 1 auto; width: 100%; min-width: 170px; }
    .slider-with-number input[type="number"] { width: 74px; }
    #max_bboxes_num { width: 74px; }
    .small { color: var(--muted); font-size: 12px; }
    .pill { display: inline-block; background: #e8faf7; border: 1px solid #9ad6ca; color: #0b5c57; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .top-note { display: flex; justify-content: space-between; align-items: center; margin: 4px 0 8px; }
    .video-grid { flex: 1; min-height: 0; display: grid; grid-template-rows: 3fr 2fr; gap: 12px; }
    .video-box { position: relative; overflow: hidden; border-radius: 11px; border: 1px solid #c7c9cd; background: #101419; min-height: 0; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05); }
    .video-box img { width: 100%; height: 100%; display: block; object-fit: contain; }
    .video-label { position: absolute; left: 8px; top: 8px; background: rgba(4, 10, 14, 0.72); color: #ecfbff; padding: 3px 8px; border-radius: 7px; font-size: 11px; font-weight: 600; letter-spacing: 0.25px; }
    .view-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; padding: 6px 8px; border: 1px solid var(--border); border-radius: 10px; background: linear-gradient(180deg, #fffefc 0%, #f7f3ea 100%); }
    .spacer { flex: 1 1 auto; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    button { border: 1px solid var(--accent); border-radius: 8px; background: linear-gradient(180deg, var(--accent) 0%, var(--accent-strong) 100%); color: #fff; padding: 6px 10px; cursor: pointer; font-weight: 600; transition: transform 90ms ease, filter 120ms ease, box-shadow 120ms ease; }
    button:hover { filter: brightness(1.03); box-shadow: 0 4px 12px rgba(13, 116, 109, 0.22); }
    button:active { transform: translateY(1px); }
    button.alt { background: #fff; color: var(--accent-strong); border-color: #7da69f; }
    #chooseFile { padding: 4px 8px; font-size: 12px; border-radius: 6px; white-space: nowrap; }
    .status { margin-top: 10px; min-height: 22px; font-size: 13px; color: #0b5d57; background: #ecfaf7; border: 1px solid #b4ddd4; border-radius: 8px; padding: 4px 8px; }
    .status:empty { border-color: transparent; background: transparent; padding: 0; min-height: 0; }
    .status.error { color: #b42318; background: #fff1f1; border-color: #f0b9b9; }
    .error { color: #c81e1e; }
    .right-bottom { flex: 0 0 210px; min-height: 0; display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-top: 12px; }
    .panel { border: 1px solid var(--border); border-radius: 10px; padding: 8px; background: linear-gradient(180deg, #fffefc 0%, #f8f5ee 100%); }
    .debug-console { margin: 0; max-height: 180px; overflow: auto; background: #101215; color: #d7f2ff; border-radius: 8px; padding: 8px; font-size: 11px; }
    .sample-box { min-height: 140px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .view-card { display: flex; flex-direction: column; overflow: hidden; }
    .drop-zone { border: 1px dashed #8dcfc2; border-radius: 8px; background: #f1fffb; color: #0f4f49; text-align: center; padding: 10px; transition: background 120ms ease, border-color 120ms ease, transform 120ms ease; }
    .drop-zone.mini { margin-top: 8px; padding: 8px; font-size: 12px; }
    .drop-zone.drag-over { background: #dffaf1; border-color: #0f766e; }
    .drop-zone.busy { opacity: 0.6; pointer-events: none; }
    @media (max-width: 1100px) {
      html, body { overflow: hidden; }
      .layout { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); }
      .row input[type="range"], .row input[type="text"], .row input[type="number"], .row select { width: 100%; }
      .video-grid { grid-template-rows: 3fr 2fr; }
      .right-bottom { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="card controls">
      <div class="brand">
        <img class="brand-logo" src="/resources/123127873.png" alt="BuzzBlimp logo" />
        <div>
          <h1 class="title">BUZZBLIMP HSV TUNER</h1>
          <div class="brand-sub">Live Vision Calibration Console</div>
        </div>
      </div>
      <div id="errorText" class="small error"></div>

      <details class="group" open>
        <summary><strong>Source + Runtime</strong></summary>
        <div class="top-note"><span class="small">Mode-aware source selector</span><span id="runtimePill" class="pill">LIVE</span></div>
        <div class="row"><label>Source mode</label><select id="source_mode"><option>image</option><option>video</option><option>camera</option></select></div>
        <div id="sourcePathRow" class="row">
          <label>Source path</label>
          <div class="inline-field">
            <input id="source_path" type="text" readonly />
            <button id="chooseFile" class="alt" type="button">Choose file</button>
            <input id="sourceFileInput" type="file" style="display:none" />
          </div>
        </div>
        <div id="dropZone" class="panel small" style="margin-top:6px;">Drag & drop image/video file here</div>
        <div id="cameraPresetRow" class="row">
          <label>Camera</label>
          <select id="camera_preset">
            <option value="webcam">Webcam (index 0)</option>
            <option value="usb">USB camera (index 1)</option>
            <option value="manual">Manual index</option>
          </select>
        </div>
        <div id="cameraIndexRow" class="row"><label>Manual index</label><input id="camera_index" type="number" min="0" step="1" /></div>
        <div id="loopVideoRow" class="row"><label>Loop video</label><input id="loop_video" type="checkbox" /></div>
        <div class="row"><label>Target mode</label><select id="target_mode"><option>balloon</option><option>goal</option><option>all</option><option>custom</option></select></div>
        <div class="row"><label>Stereo side-by-side</label><input id="stereo_sbs" type="checkbox" /></div>
        <div class="row"><label>Camera side</label><select id="camera_side"><option>left</option><option>right</option><option>both</option></select></div>
        <div class="row">
          <label>Max bboxes</label>
          <div class="inline-field">
            <input id="max_bboxes" type="range" min="1" max="20" />
            <input id="max_bboxes_num" type="number" min="1" max="20" step="1" />
          </div>
        </div>
      </details>

      <details class="group" open>
        <summary><strong>Object Registry</strong></summary>
        <div class="row"><label>Selected object</label><select id="objectSelect"></select></div>
        <div class="btn-row">
          <button id="addObject">+ Add object</button>
          <button class="alt" id="duplicateObject">Duplicate</button>
          <button class="alt" id="removeObject">Remove</button>
        </div>
        <div id="objectControls"></div>
      </details>

      <details class="group">
        <summary><strong>Profiles + YAML</strong></summary>
        <div class="btn-row">
          <button id="saveProfile">Save profile</button>
          <button class="alt" id="chooseProfileFile" type="button">Load JSON File</button>
          <input id="profileFileInput" type="file" accept=".json,application/json" style="display:none" />
        </div>
        <div id="profileDropZone" class="drop-zone mini">Drag & drop profile JSON here</div>
        <div class="btn-row">
          <button id="exportYaml">Export ROS2 YAML</button>
          <button class="alt" id="chooseYamlFile" type="button">Import YAML File</button>
          <input id="yamlFileInput" type="file" accept=".yaml,.yml,text/yaml,application/x-yaml" style="display:none" />
        </div>
        <div id="yamlDropZone" class="drop-zone mini">Drag & drop ROS2 YAML here</div>
      </details>

      <div id="status" class="status"></div>
    </div>

    <div class="card view-card">
      <div class="view-toolbar">
        <button id="togglePause">Freeze</button>
        <button class="alt" id="stepFrame">Step</button>
        <button class="alt" id="snapshot">Save snapshot</button>
        <select id="maskModeSelect">
          <option value="combined">Mask: Combined</option>
          <option value="selected">Mask: Selected Object</option>
        </select>
        <label class="small"><input id="maskColorOverlay" type="checkbox" /> color overlay</label>
        <select id="maskObjectSelect"></select>
        <span class="spacer"></span>
        <span id="fpsBadge" class="pill">FPS: 0.0</span>
      </div>
      <div class="video-grid">
        <div class="video-box"><span class="video-label">Detection View (click)</span><img id="overlayView" src="/stream/overlay.mjpg" /></div>
        <div class="video-box"><span class="video-label">Mask</span><img id="maskView" src="/stream/mask.mjpg" /></div>
      </div>
      <div class="right-bottom">
        <div class="panel">
          <div class="small">Debug Console</div>
          <pre id="debugConsole" class="debug-console">-</pre>
        </div>
        <div class="panel">
          <div class="small">Pixel Sample</div>
          <div id="sampleText" class="sample-box">Click detection view to sample.</div>
        </div>
      </div>
    </div>
  </div>

<script>
const state = {
  config: null,
  object_registry: [],
  profiles: [],
  yaml_files: [],
  selected_object_id: '',
  runtime: { paused: false, fps: 0 },
  mask_view: { mode: 'combined', color_overlay: false, selected_object_id: '' },
  debug: {}
};
let applyTimer = null;
let patchQueue = {};
let objectApplyTimer = null;
let objectPatchQueue = {};
let isUploadingSource = false;
let isUploadingProfile = false;
let isUploadingYaml = false;

function setStatus(msg, isError=false) {
  const el = document.getElementById('status');
  el.textContent = msg || '';
  el.className = isError ? 'status error' : 'status';
}

function jsonSig(v) {
  try { return JSON.stringify(v ?? null); }
  catch (_) { return ''; }
}

function hasFocus(id) {
  const el = document.getElementById(id);
  const active = document.activeElement;
  return !!(el && active && (active === el || el.contains(active)));
}

function deepPatch(target, path, value) {
  const keys = path.split('.');
  let cur = target;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    if (!cur[k] || typeof cur[k] !== 'object') cur[k] = {};
    cur = cur[k];
  }
  cur[keys[keys.length - 1]] = value;
}

async function apiPost(path, body) {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || `Request failed: ${path}`);
  return data;
}

function getSelectedObject() {
  return state.object_registry.find((x) => x.object_id === state.selected_object_id) || null;
}

function getSourceMode() {
  const modeEl = document.getElementById('source_mode');
  return modeEl ? modeEl.value : 'video';
}

function fileAllowedForMode(name, mode) {
  const n = (name || '').toLowerCase();
  const imageExt = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'];
  const videoExt = ['.avi', '.mp4', '.mov', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg'];
  if (mode === 'image') return imageExt.some((ext) => n.endsWith(ext));
  if (mode === 'video') return videoExt.some((ext) => n.endsWith(ext));
  return true;
}

function queuePatch(path, value) {
  deepPatch(patchQueue, path, value);
  if (applyTimer) clearTimeout(applyTimer);
  applyTimer = setTimeout(applyPatch, 180);
}

async function applyPatch() {
  const patch = patchQueue;
  patchQueue = {};
  if (!Object.keys(patch).length) return;
  try {
    const data = await apiPost('/api/config', patch);
    state.config = data.config;
    syncViewFromConfig();
    setStatus('Config updated');
  } catch (err) {
    setStatus(err.message, true);
  }
}

function queueObjectPatch(path, value) {
  if (!state.selected_object_id) return;
  deepPatch(objectPatchQueue, path, value);
  if (objectApplyTimer) clearTimeout(objectApplyTimer);
  objectApplyTimer = setTimeout(applyObjectPatch, 180);
}

async function applyObjectPatch() {
  const patch = objectPatchQueue;
  objectPatchQueue = {};
  if (!Object.keys(patch).length || !state.selected_object_id) return;
  try {
    const out = await apiPost('/api/object/update', { object_id: state.selected_object_id, patch });
    state.config = out.config;
    state.object_registry = out.object_registry || [];
    renderObjectSelector();
    renderObjectControls();
    renderMaskControls();
    setStatus(`Updated ${state.selected_object_id}`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function bindField(id, path, kind='text') {
  const el = document.getElementById(id);
  el.addEventListener(kind === 'checkbox' ? 'change' : 'input', () => {
    let v;
    if (kind === 'checkbox') v = !!el.checked;
    else if (kind === 'number') v = Number(el.value);
    else v = el.value;
    queuePatch(path, v);
  });
}

function syncViewFromConfig() {
  const c = state.config;
  if (!c) return;

  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (document.activeElement === el) return;
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = String(v ?? '');
  };

  ['source_mode','camera_index','target_mode','camera_side','max_bboxes'].forEach((id) => setVal(id, c[id]));
  setVal('max_bboxes_num', c.max_bboxes);
  ['loop_video','stereo_sbs'].forEach((id) => setVal(id, c[id]));

  const mode = getSourceMode();
  const sourcePathInput = document.getElementById('source_path');
  if (sourcePathInput) {
    if (mode === 'image') sourcePathInput.value = c.image_path || '';
    else if (mode === 'video') sourcePathInput.value = c.video_path || '';
    else sourcePathInput.value = '';
  }

  const sourcePathRow = document.getElementById('sourcePathRow');
  const loopVideoRow = document.getElementById('loopVideoRow');
  const cameraPresetRow = document.getElementById('cameraPresetRow');
  const cameraIndexRow = document.getElementById('cameraIndexRow');
  const chooseBtn = document.getElementById('chooseFile');
  const fileInput = document.getElementById('sourceFileInput');
  const dropZone = document.getElementById('dropZone');
  if (sourcePathRow) sourcePathRow.style.display = mode === 'camera' ? 'none' : 'grid';
  if (loopVideoRow) loopVideoRow.style.display = mode === 'video' ? 'grid' : 'none';
  if (cameraPresetRow) cameraPresetRow.style.display = mode === 'camera' ? 'grid' : 'none';
  if (cameraIndexRow) cameraIndexRow.style.display = mode === 'camera' ? 'grid' : 'none';
  if (dropZone) dropZone.style.display = mode === 'camera' ? 'none' : 'block';
  if (chooseBtn) chooseBtn.disabled = mode === 'camera';
  if (fileInput) {
    if (mode === 'image') fileInput.accept = 'image/*';
    else if (mode === 'video') fileInput.accept = 'video/*';
    else fileInput.accept = '';
  }

  const preset = document.getElementById('camera_preset');
  if (preset) {
    const idx = Number(c.camera_index || 0);
    if (idx === 0) preset.value = 'webcam';
    else if (idx === 1) preset.value = 'usb';
    else preset.value = 'manual';
  }

}

function renderRuntimeStatus() {
  document.getElementById('errorText').textContent = state.error || '';
  const pauseBtn = document.getElementById('togglePause');
  if (pauseBtn) pauseBtn.textContent = state.runtime && state.runtime.paused ? 'Resume' : 'Freeze';
  const runtimePill = document.getElementById('runtimePill');
  if (runtimePill) runtimePill.textContent = state.runtime && state.runtime.paused ? 'PAUSED' : 'LIVE';
  const fpsBadge = document.getElementById('fpsBadge');
  if (fpsBadge) fpsBadge.textContent = `FPS: ${((state.runtime && state.runtime.fps) || 0).toFixed(1)}`;
}

function renderDebugConsole() {
  const el = document.getElementById('debugConsole');
  if (!el) return;
  const debug = state.debug || {};
  el.textContent = JSON.stringify(debug, null, 2);
}

function renderMaskControls() {
  const modeSel = document.getElementById('maskModeSelect');
  const colorChk = document.getElementById('maskColorOverlay');
  const objSel = document.getElementById('maskObjectSelect');
  if (!modeSel || !colorChk || !objSel) return;

  modeSel.value = (state.mask_view && state.mask_view.mode) || 'combined';
  colorChk.checked = !!(state.mask_view && state.mask_view.color_overlay);

  const keep = (state.mask_view && state.mask_view.selected_object_id) || state.selected_object_id || '';
  objSel.innerHTML = '<option value="">selected object...</option>';
  for (const obj of state.object_registry || []) {
    const opt = document.createElement('option');
    opt.value = obj.object_id;
    const preset = obj.builtin ? 'preset' : 'custom';
    opt.textContent = `${obj.name} (${preset})`;
    objSel.appendChild(opt);
  }
  objSel.value = keep && [...objSel.options].some((o) => o.value === keep) ? keep : '';
  if (modeSel.value === 'selected' && !objSel.value && objSel.options.length > 1) {
    objSel.value = objSel.options[1].value;
  }
  objSel.style.display = modeSel.value === 'selected' ? 'inline-block' : 'none';
}

async function updateMaskView(patch) {
  try {
    const out = await apiPost('/api/mask_view', patch);
    state.mask_view = out.mask_view || state.mask_view;
    if (out.state) {
      state.runtime = out.state.runtime || state.runtime;
      state.debug = out.state.debug || state.debug;
    }
    renderMaskControls();
    renderDebugConsole();
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function uploadSourceFile(file) {
  if (!file) return;
  if (isUploadingSource) return;
  const mode = getSourceMode();
  if (mode === 'camera') {
    setStatus('Switch source mode to image or video first', true);
    return;
  }
  if (!fileAllowedForMode(file.name, mode)) {
    setStatus(`Selected file does not match ${mode} mode`, true);
    return;
  }
  isUploadingSource = true;
  const chooseBtn = document.getElementById('chooseFile');
  const fileInput = document.getElementById('sourceFileInput');
  const dropZone = document.getElementById('dropZone');
  if (chooseBtn) chooseBtn.disabled = true;
  if (fileInput) fileInput.disabled = true;
  if (dropZone) dropZone.classList.add('busy');

  try {
    setStatus(`Loading ${file.name}...`);
    const res = await fetch('/api/upload_source', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Filename': file.name,
        'X-Source-Mode': mode,
      },
      body: file,
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || 'Upload failed');
    }
    state.config = data.config || state.config;
    await refreshState();
    setStatus(`Loaded ${file.name}`);
  } finally {
    isUploadingSource = false;
    if (chooseBtn) chooseBtn.disabled = false;
    if (fileInput) fileInput.disabled = false;
    if (dropZone) dropZone.classList.remove('busy');
  }
}

async function uploadProfileFile(file) {
  if (!file) return;
  if (isUploadingProfile) return;
  const name = (file.name || '').toLowerCase();
  if (!name.endsWith('.json')) {
    setStatus('Profile upload expects a .json file', true);
    return;
  }

  isUploadingProfile = true;
  const chooseBtn = document.getElementById('chooseProfileFile');
  const fileInput = document.getElementById('profileFileInput');
  const dropZone = document.getElementById('profileDropZone');
  if (chooseBtn) chooseBtn.disabled = true;
  if (fileInput) fileInput.disabled = true;
  if (dropZone) dropZone.classList.add('busy');

  try {
    setStatus(`Importing profile ${file.name}...`);
    const res = await fetch('/api/upload_profile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Filename': file.name,
      },
      body: file,
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || 'Profile upload failed');
    }
    await refreshState();
    setStatus(`Imported profile ${data.profile_name || file.name}`);
  } finally {
    isUploadingProfile = false;
    if (chooseBtn) chooseBtn.disabled = false;
    if (fileInput) fileInput.disabled = false;
    if (dropZone) dropZone.classList.remove('busy');
  }
}

async function uploadYamlFile(file) {
  if (!file) return;
  if (isUploadingYaml) return;
  const name = (file.name || '').toLowerCase();
  if (!name.endsWith('.yaml') && !name.endsWith('.yml')) {
    setStatus('YAML import expects a .yaml or .yml file', true);
    return;
  }

  isUploadingYaml = true;
  const chooseBtn = document.getElementById('chooseYamlFile');
  const fileInput = document.getElementById('yamlFileInput');
  const dropZone = document.getElementById('yamlDropZone');
  if (chooseBtn) chooseBtn.disabled = true;
  if (fileInput) fileInput.disabled = true;
  if (dropZone) dropZone.classList.add('busy');

  try {
    setStatus(`Importing YAML ${file.name}...`);
    const res = await fetch('/api/upload_yaml', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Filename': file.name,
      },
      body: file,
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || 'YAML upload failed');
    }
    await refreshState();
    setStatus(`Imported YAML ${data.filename || file.name}`);
  } finally {
    isUploadingYaml = false;
    if (chooseBtn) chooseBtn.disabled = false;
    if (fileInput) fileInput.disabled = false;
    if (dropZone) dropZone.classList.remove('busy');
  }
}

function refreshLists() {
  const p = document.getElementById('profileList');
  if (p) {
    p.innerHTML = '';
    for (const name of state.profiles || []) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      p.appendChild(opt);
    }
  }

  const y = document.getElementById('yamlList');
  if (y) {
    y.innerHTML = '';
    for (const name of state.yaml_files || []) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      y.appendChild(opt);
    }
  }
}

function renderObjectSelector() {
  const sel = document.getElementById('objectSelect');
  const keep = state.selected_object_id;
  sel.innerHTML = '';
  for (const obj of state.object_registry || []) {
    const opt = document.createElement('option');
    opt.value = obj.object_id;
    const preset = obj.builtin ? 'preset' : 'custom';
    opt.textContent = `${obj.name} (${preset})`;
    sel.appendChild(opt);
  }
  if ((state.object_registry || []).length === 0) {
    state.selected_object_id = '';
    return;
  }
  if (keep && (state.object_registry || []).some((o) => o.object_id === keep)) {
    state.selected_object_id = keep;
  } else {
    state.selected_object_id = state.object_registry[0].object_id;
  }
  sel.value = state.selected_object_id;
  document.getElementById('removeObject').disabled = !!(getSelectedObject() && getSelectedObject().builtin);
}

function makeRow(labelText) {
  const row = document.createElement('div');
  row.className = 'row';
  const label = document.createElement('label');
  label.textContent = labelText;
  const right = document.createElement('div');
  row.appendChild(label);
  row.appendChild(right);
  return { row, right };
}

function addObjText(container, label, value, onChange, disabled=false) {
  const { row, right } = makeRow(label);
  const input = document.createElement('input');
  input.type = 'text';
  input.value = String(value ?? '');
  input.disabled = disabled;
  input.addEventListener('input', () => onChange(input.value));
  right.appendChild(input);
  container.appendChild(row);
}

function addObjCheckbox(container, label, value, onChange) {
  const { row, right } = makeRow(label);
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = !!value;
  input.addEventListener('change', () => onChange(!!input.checked));
  right.appendChild(input);
  container.appendChild(row);
}

function addObjSlider(container, label, value, min, max, step, onChange) {
  const { row, right } = makeRow(label);
  const wrap = document.createElement('div');
  wrap.className = 'slider-with-number';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(value);

  const number = document.createElement('input');
  number.type = 'number';
  number.min = String(min);
  number.max = String(max);
  number.step = String(step);
  number.value = String(value);

  const clampValue = (raw) => {
    let v = Number(raw);
    if (Number.isNaN(v)) v = Number(slider.value);
    v = Math.max(Number(min), Math.min(Number(max), v));
    if (Number(step) === 1) v = Math.round(v);
    return v;
  };

  const setBoth = (v) => {
    slider.value = String(v);
    number.value = String(v);
  };

  slider.addEventListener('input', () => {
    const v = clampValue(slider.value);
    number.value = String(v);
    onChange(v);
  });

  const applyTypedValue = () => {
    const v = clampValue(number.value);
    setBoth(v);
    onChange(v);
  };

  number.addEventListener('change', applyTypedValue);
  number.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      applyTypedValue();
      number.blur();
    }
  });

  wrap.appendChild(slider);
  wrap.appendChild(number);
  right.appendChild(wrap);
  container.appendChild(row);
}

function renderObjectControls() {
  const box = document.getElementById('objectControls');
  box.innerHTML = '';
  const obj = getSelectedObject();
  if (!obj) return;

  const kindInfo = document.createElement('div');
  kindInfo.className = 'small';
  kindInfo.textContent = `kind: ${obj.detector_kind} | id: ${obj.object_id}`;
  box.appendChild(kindInfo);

  addObjText(box, 'name', obj.name, (v) => queueObjectPatch('name', v), !!obj.builtin);
  addObjCheckbox(box, 'enabled', obj.enabled, (v) => queueObjectPatch('enabled', v));

  addObjSlider(box, 'h_min', obj.hsv.h_min, 0, 179, 1, (v) => queueObjectPatch('hsv.h_min', v));
  addObjSlider(box, 'h_max', obj.hsv.h_max, 0, 179, 1, (v) => queueObjectPatch('hsv.h_max', v));
  addObjSlider(box, 's_min', obj.hsv.s_min, 0, 255, 1, (v) => queueObjectPatch('hsv.s_min', v));
  addObjSlider(box, 's_max', obj.hsv.s_max, 0, 255, 1, (v) => queueObjectPatch('hsv.s_max', v));
  addObjSlider(box, 'v_min', obj.hsv.v_min, 0, 255, 1, (v) => queueObjectPatch('hsv.v_min', v));
  addObjSlider(box, 'v_max', obj.hsv.v_max, 0, 255, 1, (v) => queueObjectPatch('hsv.v_max', v));

  if (obj.detector_kind.startsWith('goal_')) {
    addObjSlider(box, 'score_threshold', obj.score_threshold ?? 0.2, 0, 1, 0.01, (v) => queueObjectPatch('score_threshold', v));
  } else {
    addObjSlider(box, 'min_area', obj.min_area ?? 250, 1, 10000, 1, (v) => queueObjectPatch('min_area', v));
    addObjSlider(box, 'min_fill', obj.min_fill ?? 20, 0, 100, 1, (v) => queueObjectPatch('min_fill', v));
    addObjSlider(box, 'ignore_top_ratio', obj.ignore_top_ratio ?? 0, 0, 1, 0.01, (v) => queueObjectPatch('ignore_top_ratio', v));
  }

  document.getElementById('removeObject').disabled = !!obj.builtin;
}

async function refreshState() {
  const prevConfigSig = jsonSig(state.config);
  const prevRegistrySig = jsonSig(state.object_registry);
  const prevProfilesSig = jsonSig(state.profiles);
  const prevYamlSig = jsonSig(state.yaml_files);
  const prevMaskViewSig = jsonSig(state.mask_view);

  const res = await fetch('/api/state');
  const data = await res.json();
  state.config = data.config;
  state.object_registry = data.object_registry || [];
  state.profiles = data.profiles || [];
  state.yaml_files = data.yaml_files || [];
  state.runtime = data.runtime || { paused: false, fps: 0 };
  state.mask_view = data.mask_view || { mode: 'combined', color_overlay: false, selected_object_id: '' };
  state.debug = data.debug || {};
  state.error = data.error || '';

  const configChanged = jsonSig(state.config) !== prevConfigSig;
  const registryChanged = jsonSig(state.object_registry) !== prevRegistrySig;
  const profilesChanged = jsonSig(state.profiles) !== prevProfilesSig;
  const yamlChanged = jsonSig(state.yaml_files) !== prevYamlSig;
  const maskViewChanged = jsonSig(state.mask_view) !== prevMaskViewSig;

  const editingSourceUi = hasFocus('source_mode') || hasFocus('sourcePathRow') || hasFocus('cameraIndexRow') ||
    hasFocus('target_mode') || hasFocus('camera_side') || hasFocus('max_bboxes') ||
    hasFocus('max_bboxes_num') || hasFocus('camera_preset');
  if (configChanged && !editingSourceUi) {
    syncViewFromConfig();
  } else if (!state.config) {
    syncViewFromConfig();
  }
  renderRuntimeStatus();

  if (profilesChanged || yamlChanged) {
    refreshLists();
  }

  if (registryChanged && !hasFocus('objectControls') && !hasFocus('objectSelect')) {
    renderObjectSelector();
    renderObjectControls();
  }
  if ((registryChanged || maskViewChanged) && !hasFocus('maskModeSelect') && !hasFocus('maskObjectSelect')) {
    renderMaskControls();
  }
  renderDebugConsole();
}

function bindBasicInputs() {
  bindField('source_mode', 'source_mode');
  bindField('camera_index', 'camera_index', 'number');
  bindField('loop_video', 'loop_video', 'checkbox');
  bindField('target_mode', 'target_mode');
  bindField('stereo_sbs', 'stereo_sbs', 'checkbox');
  bindField('camera_side', 'camera_side');
  bindField('max_bboxes', 'max_bboxes', 'number');

  const maxBboxesRange = document.getElementById('max_bboxes');
  const maxBboxesNum = document.getElementById('max_bboxes_num');
  maxBboxesRange.addEventListener('input', () => {
    maxBboxesNum.value = maxBboxesRange.value;
  });
  maxBboxesNum.addEventListener('change', () => {
    let v = Number(maxBboxesNum.value || 1);
    if (Number.isNaN(v)) v = 1;
    v = Math.max(1, Math.min(20, Math.round(v)));
    maxBboxesNum.value = String(v);
    maxBboxesRange.value = String(v);
    queuePatch('max_bboxes', v);
  });

  document.getElementById('source_mode').addEventListener('change', () => {
    const mode = getSourceMode();
    const v = (document.getElementById('source_path').value || '').trim();
    if (mode === 'image') queuePatch('image_path', v);
    if (mode === 'video') queuePatch('video_path', v);
    syncViewFromConfig();
  });

  document.getElementById('camera_preset').addEventListener('change', () => {
    const preset = document.getElementById('camera_preset').value;
    if (preset === 'webcam') queuePatch('camera_index', 0);
    else if (preset === 'usb') queuePatch('camera_index', 1);
  });

  document.getElementById('source_path').addEventListener('change', () => {
    const mode = getSourceMode();
    const v = (document.getElementById('source_path').value || '').trim();
    if (mode === 'image') queuePatch('image_path', v);
    if (mode === 'video') queuePatch('video_path', v);
  });

  document.getElementById('chooseFile').addEventListener('click', () => {
    if (getSourceMode() === 'camera') return;
    document.getElementById('sourceFileInput').click();
  });

  document.getElementById('sourceFileInput').addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      await uploadSourceFile(file);
    } catch (err) { setStatus(err.message, true); }
    e.target.value = '';
  });

  const dropZone = document.getElementById('dropZone');
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    try {
      await uploadSourceFile(file);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('chooseProfileFile').addEventListener('click', () => {
    document.getElementById('profileFileInput').click();
  });
  document.getElementById('profileFileInput').addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      await uploadProfileFile(file);
    } catch (err) { setStatus(err.message, true); }
    e.target.value = '';
  });
  const profileDropZone = document.getElementById('profileDropZone');
  profileDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    profileDropZone.classList.add('drag-over');
  });
  profileDropZone.addEventListener('dragleave', () => {
    profileDropZone.classList.remove('drag-over');
  });
  profileDropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    profileDropZone.classList.remove('drag-over');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    try {
      await uploadProfileFile(file);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('chooseYamlFile').addEventListener('click', () => {
    document.getElementById('yamlFileInput').click();
  });
  document.getElementById('yamlFileInput').addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      await uploadYamlFile(file);
    } catch (err) { setStatus(err.message, true); }
    e.target.value = '';
  });
  const yamlDropZone = document.getElementById('yamlDropZone');
  yamlDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    yamlDropZone.classList.add('drag-over');
  });
  yamlDropZone.addEventListener('dragleave', () => {
    yamlDropZone.classList.remove('drag-over');
  });
  yamlDropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    yamlDropZone.classList.remove('drag-over');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    try {
      await uploadYamlFile(file);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('togglePause').addEventListener('click', async () => {
    try {
      const paused = !(state.runtime && state.runtime.paused);
      const out = await apiPost('/api/runtime', { paused });
      state.runtime = out.runtime || state.runtime;
      await refreshState();
      setStatus(paused ? 'Video frozen' : 'Video resumed');
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('stepFrame').addEventListener('click', async () => {
    try {
      await apiPost('/api/runtime', { step: true });
      await refreshState();
      setStatus('Stepped one frame');
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('objectSelect').addEventListener('change', (e) => {
    state.selected_object_id = e.target.value;
    objectPatchQueue = {};
    renderObjectControls();
    renderMaskControls();
  });

  document.getElementById('addObject').addEventListener('click', async () => {
    const name = prompt('New object name?', 'custom_object');
    if (!name) return;
    try {
      const out = await apiPost('/api/object/add', { name });
      state.selected_object_id = out.object_id;
      await refreshState();
      setStatus(`Added object ${name}`);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('duplicateObject').addEventListener('click', async () => {
    if (!state.selected_object_id) return;
    try {
      const out = await apiPost('/api/object/duplicate', { object_id: state.selected_object_id });
      state.selected_object_id = out.object_id;
      await refreshState();
      setStatus('Object duplicated');
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('removeObject').addEventListener('click', async () => {
    const obj = getSelectedObject();
    if (!obj || obj.builtin) return;
    if (!confirm(`Remove object ${obj.name}?`)) return;
    try {
      await apiPost('/api/object/remove', { object_id: obj.object_id });
      state.selected_object_id = '';
      await refreshState();
      setStatus('Object removed');
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('saveProfile').addEventListener('click', async () => {
    const name = prompt('Profile name?', 'arena_profile');
    if (!name) return;
    try {
      await apiPost('/api/profile/save', { name });
      await refreshState();
      setStatus(`Saved profile ${name}`);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('exportYaml').addEventListener('click', async () => {
    const name = prompt('YAML file name?', 'vision_tuning');
    if (!name) return;
    try {
      const out = await apiPost('/api/export/yaml', { name, node_name: 'vision_tuning' });
      await refreshState();
      setStatus(`Exported ${out.filename}`);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('snapshot').addEventListener('click', async () => {
    const name = prompt('Snapshot file name?', 'snapshot');
    if (!name) return;
    try {
      const out = await apiPost('/api/snapshot', { name });
      setStatus(`Saved ${out.filename}`);
    } catch (err) { setStatus(err.message, true); }
  });

  document.getElementById('overlayView').addEventListener('click', async (e) => {
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    if (!img.naturalWidth || !img.naturalHeight) return;
    const x = Math.round((e.clientX - rect.left) * img.naturalWidth / rect.width);
    const y = Math.round((e.clientY - rect.top) * img.naturalHeight / rect.height);
    try {
      const out = await apiPost('/api/sample', { x, y });
      const s = out.sample;
      document.getElementById('sampleText').textContent = `x=${s.x} y=${s.y} BGR=${s.bgr.join(',')} HSV=${s.hsv.join(',')}`;
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  const maskModeSelect = document.getElementById('maskModeSelect');
  const maskColorOverlay = document.getElementById('maskColorOverlay');
  const maskObjectSelect = document.getElementById('maskObjectSelect');

  maskModeSelect.addEventListener('change', async () => {
    const mode = maskModeSelect.value || 'combined';
    const selected = maskObjectSelect.value || state.selected_object_id || '';
    await updateMaskView({ mode, selected_object_id: selected });
    renderMaskControls();
  });

  maskColorOverlay.addEventListener('change', async () => {
    await updateMaskView({ color_overlay: !!maskColorOverlay.checked });
  });

  maskObjectSelect.addEventListener('change', async () => {
    await updateMaskView({ selected_object_id: maskObjectSelect.value || '' });
  });
}

(async function init() {
  bindBasicInputs();
  await refreshState();
  setInterval(refreshState, 1000);
})();
</script>
</body>
</html>
"""
