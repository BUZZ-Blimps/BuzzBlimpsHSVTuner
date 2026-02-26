from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .config import ToolConfig


def _node_block(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    for value in doc.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            params = value.get("ros__parameters")
            return params if isinstance(params, dict) else {}
    return {}


def _hsv_dict(hsv_obj: Any) -> Dict[str, Any]:
    return {
        "h_min": int(hsv_obj.h_min),
        "h_max": int(hsv_obj.h_max),
        "s_min": int(hsv_obj.s_min),
        "s_max": int(hsv_obj.s_max),
        "v_min": int(hsv_obj.v_min),
        "v_max": int(hsv_obj.v_max),
    }


def _legacy_flat_parameters(cfg: ToolConfig) -> Dict[str, Any]:
    return {
        "green_lh": cfg.green_hsv.h_min,
        "green_uh": cfg.green_hsv.h_max,
        "green_ls": cfg.green_hsv.s_min,
        "green_us": cfg.green_hsv.s_max,
        "green_lv": cfg.green_hsv.v_min,
        "green_uv": cfg.green_hsv.v_max,
        "purple_lh": cfg.purple_hsv.h_min,
        "purple_uh": cfg.purple_hsv.h_max,
        "purple_ls": cfg.purple_hsv.s_min,
        "purple_us": cfg.purple_hsv.s_max,
        "purple_lv": cfg.purple_hsv.v_min,
        "purple_uv": cfg.purple_hsv.v_max,
        "min_area": cfg.min_area,
        "min_percent_filled": cfg.min_percent_filled,
        "ignore_top_ratio": cfg.ignore_top_ratio,
        "include_green": cfg.include_green,
        "include_purple": cfg.include_purple,
        "use_kalman": cfg.use_kalman,
        "use_optical_flow": cfg.use_optical_flow,
        "use_lock": cfg.use_lock,
        "goal_orange_h_min": cfg.goal_orange_hsv.h_min,
        "goal_orange_h_max": cfg.goal_orange_hsv.h_max,
        "goal_orange_s_min": cfg.goal_orange_hsv.s_min,
        "goal_orange_s_max": cfg.goal_orange_hsv.s_max,
        "goal_orange_v_min": cfg.goal_orange_hsv.v_min,
        "goal_orange_v_max": cfg.goal_orange_hsv.v_max,
        "goal_yellow_h_min": cfg.goal_yellow_hsv.h_min,
        "goal_yellow_h_max": cfg.goal_yellow_hsv.h_max,
        "goal_yellow_s_min": cfg.goal_yellow_hsv.s_min,
        "goal_yellow_s_max": cfg.goal_yellow_hsv.s_max,
        "goal_yellow_v_min": cfg.goal_yellow_hsv.v_min,
        "goal_yellow_v_max": cfg.goal_yellow_hsv.v_max,
        "include_goal_orange": cfg.include_goal_orange,
        "include_goal_yellow": cfg.include_goal_yellow,
        "goal_score_threshold": cfg.goal_score_threshold,
    }


def to_ros_parameters(cfg: ToolConfig) -> Dict[str, Any]:
    """
    Export a ROS2-compatible parameter tree.

    The modular tree is the primary format.
    Legacy flat keys are also exported for backward compatibility with
    existing nodes that still expect the old names.
    """
    detectors = {
        "green_balloon": {
            "enabled": bool(cfg.include_green),
            "hsv": _hsv_dict(cfg.green_hsv),
            "filters": {
                "min_area": int(cfg.min_area),
                "min_fill": float(cfg.min_percent_filled),
                "ignore_top_ratio": float(cfg.ignore_top_ratio),
            },
        },
        "purple_balloon": {
            "enabled": bool(cfg.include_purple),
            "hsv": _hsv_dict(cfg.purple_hsv),
            "filters": {
                "min_area": int(cfg.min_area),
                "min_fill": float(cfg.min_percent_filled),
                "ignore_top_ratio": float(cfg.ignore_top_ratio),
            },
        },
        "goal_orange": {
            "enabled": bool(cfg.include_goal_orange),
            "hsv": _hsv_dict(cfg.goal_orange_hsv),
            "score_threshold": float(cfg.goal_score_threshold),
        },
        "goal_yellow": {
            "enabled": bool(cfg.include_goal_yellow),
            "hsv": _hsv_dict(cfg.goal_yellow_hsv),
            "score_threshold": float(cfg.goal_score_threshold),
        },
    }

    custom_objects: Dict[str, Any] = {}
    for obj in cfg.custom_objects:
        custom_id = obj.object_id or obj.name
        custom_objects[custom_id] = {
            "name": obj.name,
            "detector_kind": obj.detector_kind,
            "enabled": bool(obj.enabled),
            "hsv": _hsv_dict(obj.hsv),
            "filters": {
                "min_area": int(obj.min_area),
                "min_fill": float(obj.min_fill),
                "ignore_top_ratio": float(obj.ignore_top_ratio),
            },
            "color_bgr": [int(c) for c in obj.color_bgr],
        }

    params: Dict[str, Any] = {
        "format_version": 2,
        "detectors": detectors,
        "tracking": {
            "use_kalman": bool(cfg.use_kalman),
            "use_optical_flow": bool(cfg.use_optical_flow),
            "use_lock": bool(cfg.use_lock),
        },
    }
    if custom_objects:
        params["custom_objects"] = custom_objects

    # Keep old names so existing ROS2 nodes can consume this file unchanged.
    params.update(_legacy_flat_parameters(cfg))
    return params


def export_ros_yaml(cfg: ToolConfig, path: Path, node_name: str = "vision_tuning") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {node_name: {"ros__parameters": to_ros_parameters(cfg)}}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path


def _flatten_dict(data: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_dict(value, full))
    else:
        out[prefix] = data
    return out


def _first(flat: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in flat:
            return flat[key]
    return None


def import_ros_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    params = _node_block(doc)
    flat = _flatten_dict(params)

    update: Dict[str, Any] = {}

    # Balloon HSV (new modular + legacy fallback)
    green_h = _first(flat, "detectors.green_balloon.hsv.h_min", "green_lh")
    if green_h is not None:
        update["green_hsv"] = {
            "h_min": _first(flat, "detectors.green_balloon.hsv.h_min", "green_lh"),
            "h_max": _first(flat, "detectors.green_balloon.hsv.h_max", "green_uh"),
            "s_min": _first(flat, "detectors.green_balloon.hsv.s_min", "green_ls"),
            "s_max": _first(flat, "detectors.green_balloon.hsv.s_max", "green_us"),
            "v_min": _first(flat, "detectors.green_balloon.hsv.v_min", "green_lv"),
            "v_max": _first(flat, "detectors.green_balloon.hsv.v_max", "green_uv"),
        }

    purple_h = _first(flat, "detectors.purple_balloon.hsv.h_min", "purple_lh")
    if purple_h is not None:
        update["purple_hsv"] = {
            "h_min": _first(flat, "detectors.purple_balloon.hsv.h_min", "purple_lh"),
            "h_max": _first(flat, "detectors.purple_balloon.hsv.h_max", "purple_uh"),
            "s_min": _first(flat, "detectors.purple_balloon.hsv.s_min", "purple_ls"),
            "s_max": _first(flat, "detectors.purple_balloon.hsv.s_max", "purple_us"),
            "v_min": _first(flat, "detectors.purple_balloon.hsv.v_min", "purple_lv"),
            "v_max": _first(flat, "detectors.purple_balloon.hsv.v_max", "purple_uv"),
        }

    # Shared contour filters
    min_area = _first(
        flat,
        "detectors.green_balloon.filters.min_area",
        "detectors.purple_balloon.filters.min_area",
        "min_area",
    )
    if min_area is not None:
        update["min_area"] = min_area

    min_fill = _first(
        flat,
        "detectors.green_balloon.filters.min_fill",
        "detectors.purple_balloon.filters.min_fill",
        "min_percent_filled",
    )
    if min_fill is not None:
        update["min_percent_filled"] = min_fill

    ignore_top = _first(
        flat,
        "detectors.green_balloon.filters.ignore_top_ratio",
        "detectors.purple_balloon.filters.ignore_top_ratio",
        "ignore_top_ratio",
    )
    if ignore_top is not None:
        update["ignore_top_ratio"] = ignore_top

    # Enabled flags + tracking toggles
    include_green = _first(flat, "detectors.green_balloon.enabled", "include_green")
    if include_green is not None:
        update["include_green"] = include_green

    include_purple = _first(flat, "detectors.purple_balloon.enabled", "include_purple")
    if include_purple is not None:
        update["include_purple"] = include_purple

    include_goal_orange = _first(flat, "detectors.goal_orange.enabled", "include_goal_orange")
    if include_goal_orange is not None:
        update["include_goal_orange"] = include_goal_orange

    include_goal_yellow = _first(flat, "detectors.goal_yellow.enabled", "include_goal_yellow")
    if include_goal_yellow is not None:
        update["include_goal_yellow"] = include_goal_yellow

    for key, dotted in [
        ("use_kalman", "tracking.use_kalman"),
        ("use_optical_flow", "tracking.use_optical_flow"),
        ("use_lock", "tracking.use_lock"),
    ]:
        value = _first(flat, dotted, key)
        if value is not None:
            update[key] = value

    # Goal HSV + score threshold
    goal_orange_h = _first(flat, "detectors.goal_orange.hsv.h_min", "goal_orange_h_min")
    if goal_orange_h is not None:
        update["goal_orange_hsv"] = {
            "h_min": _first(flat, "detectors.goal_orange.hsv.h_min", "goal_orange_h_min"),
            "h_max": _first(flat, "detectors.goal_orange.hsv.h_max", "goal_orange_h_max"),
            "s_min": _first(flat, "detectors.goal_orange.hsv.s_min", "goal_orange_s_min"),
            "s_max": _first(flat, "detectors.goal_orange.hsv.s_max", "goal_orange_s_max"),
            "v_min": _first(flat, "detectors.goal_orange.hsv.v_min", "goal_orange_v_min"),
            "v_max": _first(flat, "detectors.goal_orange.hsv.v_max", "goal_orange_v_max"),
        }

    goal_yellow_h = _first(flat, "detectors.goal_yellow.hsv.h_min", "goal_yellow_h_min")
    if goal_yellow_h is not None:
        update["goal_yellow_hsv"] = {
            "h_min": _first(flat, "detectors.goal_yellow.hsv.h_min", "goal_yellow_h_min"),
            "h_max": _first(flat, "detectors.goal_yellow.hsv.h_max", "goal_yellow_h_max"),
            "s_min": _first(flat, "detectors.goal_yellow.hsv.s_min", "goal_yellow_s_min"),
            "s_max": _first(flat, "detectors.goal_yellow.hsv.s_max", "goal_yellow_s_max"),
            "v_min": _first(flat, "detectors.goal_yellow.hsv.v_min", "goal_yellow_v_min"),
            "v_max": _first(flat, "detectors.goal_yellow.hsv.v_max", "goal_yellow_v_max"),
        }

    goal_score = _first(
        flat,
        "detectors.goal_orange.score_threshold",
        "detectors.goal_yellow.score_threshold",
        "goal_score_threshold",
    )
    if goal_score is not None:
        update["goal_score_threshold"] = goal_score

    # Custom objects (modular format)
    custom_bucket: Dict[str, Dict[str, Any]] = {}
    for key, value in flat.items():
        if not key.startswith("custom_objects."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        object_id = parts[1]
        field = ".".join(parts[2:])
        item = custom_bucket.setdefault(object_id, {"object_id": object_id})
        if field == "name":
            item["name"] = value
        elif field == "detector_kind":
            item["detector_kind"] = value
        elif field == "enabled":
            item["enabled"] = value
        elif field.startswith("hsv."):
            item.setdefault("hsv", {})[field.split(".", 1)[1]] = value
        elif field.startswith("filters."):
            suffix = field.split(".", 1)[1]
            if suffix == "min_area":
                item["min_area"] = value
            elif suffix == "min_fill":
                item["min_fill"] = value
            elif suffix == "ignore_top_ratio":
                item["ignore_top_ratio"] = value
        elif field == "color_bgr":
            item["color_bgr"] = value

    if custom_bucket:
        update["custom_objects"] = list(custom_bucket.values())

    # Remove missing keys before applying update.
    for section in ["green_hsv", "purple_hsv", "goal_orange_hsv", "goal_yellow_hsv"]:
        if section in update:
            update[section] = {k: v for k, v in update[section].items() if v is not None}

    return update
