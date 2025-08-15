import json
import os
import hashlib
import datetime as dt
import platform
import subprocess
from typing import Any, Dict, Iterable, Optional

from src.config import RESULTS_DIR


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    return obj


def compute_tag(config_slice: Dict[str, Any], prefix: Optional[str] = None) -> str:
    canon = _canonical(config_slice)
    payload = json.dumps(canon, separators=(",", ":"), ensure_ascii=False)
    short = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{short}" if prefix else short


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def ensure_experiment_dir(tag: str) -> str:
    base = os.path.join(RESULTS_DIR, "experiments", tag)
    os.makedirs(base, exist_ok=True)
    return base


def get_full_config_snapshot() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of all uppercase items in src.config.

    Non-serializable values are stringified to avoid save failures.
    """
    import src.config as cfg  # local import to avoid cyclics at module import

    snapshot: Dict[str, Any] = {}
    for name in dir(cfg):
        if not name.isupper():
            continue
        if name.startswith("__"):
            continue
        val = getattr(cfg, name)
        # Try JSON encode to validate, else fallback to str
        try:
            json.dumps(val, ensure_ascii=False)
            snapshot[name] = val
        except TypeError:
            snapshot[name] = str(val)
    return snapshot


def save_manifest(
    tag: str, full_config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> str:
    base = ensure_experiment_dir(tag)
    manifest = {
        "tag": tag,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "config": _canonical(full_config),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
        },
    }
    if extra:
        manifest["extra"] = extra
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_canonical(full_config), f, ensure_ascii=False, indent=2)
    return base


def save_feature_list(tag: str, features: Iterable[str]) -> None:
    base = ensure_experiment_dir(tag)
    with open(os.path.join(base, "features.json"), "w", encoding="utf-8") as f:
        json.dump(list(features), f, ensure_ascii=False, indent=2)


def save_predictions(tag: str, df, filename: str = "predictions.csv") -> str:
    base = ensure_experiment_dir(tag)
    out = os.path.join(base, filename)
    df.to_csv(out, index=False)
    return out
