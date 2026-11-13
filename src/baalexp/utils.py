
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"Cannot serialize object of type {type(o)}")

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list_arg(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def env_report(include_torch: bool = False) -> dict[str, Any]:
    import platform
    import pandas
    import scipy
    import sklearn

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pid": os.getpid(),
    }
    if include_torch:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            report.update(
                {
                    "torch": torch.__version__,
                    "cuda_available": cuda_available,
                    "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
                    "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                }
            )
        except Exception as exc:
            report["torch"] = None
            report["torch_error"] = repr(exc)
    return report

