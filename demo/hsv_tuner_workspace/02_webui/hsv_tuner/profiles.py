from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


_VALID_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_name(name: str, default: str = "profile") -> str:
    cleaned = _VALID_NAME.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


class ProfileStore:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root_dir / f"{sanitize_name(name)}.json"

    def save(self, name: str, data: Dict) -> Path:
        path = self._path(name)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def load(self, name: str) -> Dict:
        path = self._path(name)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_names(self) -> List[str]:
        out = []
        for p in sorted(self.root_dir.glob("*.json")):
            out.append(p.stem)
        return out
