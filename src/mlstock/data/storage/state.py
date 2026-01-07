from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict


def read_state(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_state(state: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))
