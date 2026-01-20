import json
import os
import hashlib
import datetime as dt
import platform
import subprocess
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from src.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Module-level session tag: once set, all scripts in the same process share it
# ---------------------------------------------------------------------------
_SESSION_TAG: Optional[str] = None


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    return obj


def _generate_date_tag() -> str:
    """Generate a date-based tag like '251208-1', '251208-2', etc.
    
    Format: YYMMDD-N where N is an incrementing number for same-day experiments.
    """
    experiments_dir = os.path.join(RESULTS_DIR, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    today = dt.datetime.now().strftime("%y%m%d")  # e.g., "251208" for Dec 8, 2025
    
    # Find existing folders for today
    existing = []
    pattern = re.compile(rf"^{today}-(\d+)(_.*)?$") # Match suffix too
    if os.path.exists(experiments_dir):
        for name in os.listdir(experiments_dir):
            match = pattern.match(name)
            if match:
                existing.append(int(match.group(1)))
    
    # Next number
    next_num = max(existing, default=0) + 1
    return f"{today}-{next_num}"


def get_or_create_session_tag() -> str:
    """Get the current session tag, creating one if needed.
    
    All scripts in the same process will share the same tag.
    """
    global _SESSION_TAG
    if _SESSION_TAG is None:
        _SESSION_TAG = _generate_date_tag()
    return _SESSION_TAG


def set_session_tag(tag: str) -> None:
    """Manually set the session tag (useful for resuming an experiment)."""
    global _SESSION_TAG
    _SESSION_TAG = tag


def reset_session_tag() -> None:
    """Reset the session tag (will generate a new one on next call)."""
    global _SESSION_TAG
    _SESSION_TAG = None


def compute_tag(config_slice: Dict[str, Any], prefix: Optional[str] = None) -> str:
    """Compute a hash-based tag from config (legacy, still available)."""
    canon = _canonical(config_slice)
    payload = json.dumps(canon, separators=(",", ":"), ensure_ascii=False)
    short = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{short}" if prefix else short


def compute_tag_with_signature(
    prefix: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Return a date-based experiment tag for the current session.

    Returns (tag, config_signature, full_config_snapshot, config_slice_used_for_tag).
    
    The tag is date-based (YYMMDD-N) and shared across all scripts in the session.
    The config_signature is still computed for tracking config changes.
    """
    # Get or create the session tag (date-based)
    tag = get_or_create_session_tag()
    
    # Still compute config signature for tracking
    full_cfg_snapshot = get_full_config_snapshot()
    signature_payload = json.dumps(
        full_cfg_snapshot, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    cfg_signature = hashlib.sha256(signature_payload).hexdigest()[:12]
    cfg_slice = {
        "PREPROCESSING": full_cfg_snapshot.get("PREPROCESSING"),
        "EXPERIMENT": full_cfg_snapshot.get("EXPERIMENT"),
        "CONFIG_SIGNATURE": cfg_signature,
    }
    
    return tag, cfg_signature, full_cfg_snapshot, cfg_slice


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
