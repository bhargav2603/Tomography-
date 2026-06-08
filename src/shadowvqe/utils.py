"""Shared utilities: logging, seeding, result containers, input validation."""

from __future__ import annotations

import json
import logging
import time
from json import JSONEncoder
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a sensible default format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> np.random.Generator:
    """Seed numpy and return a fresh default_rng for downstream use."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager wall-clock timer."""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed: float = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class _NumpyEncoder(JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class IterationRecord:
    iteration: int
    energy: float
    variance: float | None = None
    n_shots: int | None = None
    runtime_s: float | None = None


@dataclass
class OptimizationResult:
    """Common result container shared by VQE and ShadowVQE."""

    method: str
    ground_state_energy: float
    optimal_parameters: list[float]
    n_iterations: int
    n_function_evals: int
    converged: bool
    history: list[IterationRecord] = field(default_factory=list)
    total_runtime_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def energy_history(self) -> list[float]:
        return [r.energy for r in self.history]

    def variance_history(self) -> list[float | None]:
        return [r.variance for r in self.history]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["history"] = [asdict(r) for r in self.history]
        return d

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def validate_positive_int(value: Any, name: str) -> int:
    """Raise ValueError with a clear message if value is not a positive int."""
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from exc
    if v < 1:
        raise ValueError(f"{name} must be >= 1, got {v}")
    return v


def validate_seed(seed: Any) -> int:
    """Accept int or None; return a concrete int seed (0 if None)."""
    if seed is None:
        return 0
    try:
        s = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer or None, got {type(seed).__name__}") from exc
    if not (0 <= s <= 2**31 - 1):
        raise ValueError(f"seed must be in [0, 2^31-1], got {s}")
    return s
