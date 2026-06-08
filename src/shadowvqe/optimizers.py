"""
Optimizer wrappers for VQE.

Both COBYLA and SPSA are wrapped around SciPy/custom implementations and
expose a uniform ``minimize`` interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .utils import get_logger, validate_positive_int

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Common callback that records energy history
# ---------------------------------------------------------------------------

@dataclass
class _ConvergenceTracker:
    """Tracks function evaluations and energy history inside an optimizer."""

    tol_energy: float = 1e-6
    patience: int = 3           # consecutive steps within tol before declaring convergence
    history: list[float] = field(default_factory=list, init=False)
    _plateau_count: int = field(default=0, init=False, repr=False)

    def record(self, energy: float) -> None:
        self.history.append(energy)
        if len(self.history) >= 2:
            delta = abs(self.history[-1] - self.history[-2])
            if delta < self.tol_energy:
                self._plateau_count += 1
            else:
                self._plateau_count = 0

    @property
    def converged(self) -> bool:
        return self._plateau_count >= self.patience


# ---------------------------------------------------------------------------
# COBYLA optimizer
# ---------------------------------------------------------------------------

class COBYLA:
    """
    Constrained Optimisation By Linear Approximation via SciPy.

    Parameters
    ----------
    max_iter  : maximum number of function evaluations.
    tol       : convergence tolerance on energy change.
    rhobeg    : COBYLA initial step size.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-6,
        rhobeg: float = 0.1,
    ) -> None:
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = float(tol)
        self.rhobeg = float(rhobeg)

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, list[float], bool, int]:
        """
        Run optimisation.

        Parameters
        ----------
        objective : callable — takes parameter array, returns scalar energy.
        x0        : initial parameters.

        Returns
        -------
        (optimal_params, energy_history, converged, n_evals)
        """
        tracker = _ConvergenceTracker(tol_energy=self.tol)

        def wrapped(x: np.ndarray) -> float:
            e = float(objective(x))
            tracker.record(e)
            _log.debug("COBYLA iter %d: E = %.8f", len(tracker.history), e)
            return e

        result = scipy_minimize(
            wrapped,
            x0,
            method="COBYLA",
            options={
                "maxiter": self.max_iter,
                "rhobeg": self.rhobeg,
                "catol": self.tol,
            },
        )

        _log.info(
            "COBYLA finished: %d evals, E = %.8f, success = %s",
            result.nfev,
            result.fun,
            result.success,
        )
        return result.x, tracker.history, tracker.converged, result.nfev


# ---------------------------------------------------------------------------
# SPSA optimizer
# ---------------------------------------------------------------------------

class SPSA:
    """
    Simultaneous Perturbation Stochastic Approximation.

    Gradient-free, noisy-function-tolerant optimizer well-suited for
    shadow-based energy estimates.

    Parameters
    ----------
    max_iter    : number of SPSA steps.
    a, c        : learning-rate and perturbation parameters.
    alpha, gamma: decay exponents (standard: alpha=0.602, gamma=0.101).
    seed        : reproducibility for perturbation directions.
    tol         : energy-delta convergence tolerance.
    """

    def __init__(
        self,
        max_iter: int = 200,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: int = 0,
        tol: float = 1e-6,
    ) -> None:
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.a = float(a)
        self.c = float(c)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tol = float(tol)
        self._rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, list[float], bool, int]:
        """
        Parameters
        ----------
        objective : callable — parameter array → scalar energy.
        x0        : initial parameters.

        Returns
        -------
        (optimal_params, energy_history, converged, n_evals)
        """
        x = x0.copy().astype(float)
        tracker = _ConvergenceTracker(tol_energy=self.tol)
        n_evals = 0
        A = 0.01 * self.max_iter  # stability constant

        for k in range(1, self.max_iter + 1):
            a_k = self.a / (k + A) ** self.alpha
            c_k = self.c / k**self.gamma

            # Rademacher perturbation vector
            delta = self._rng.choice([-1.0, 1.0], size=len(x))

            e_plus = float(objective(x + c_k * delta))
            e_minus = float(objective(x - c_k * delta))
            n_evals += 2

            gradient_approx = (e_plus - e_minus) / (2.0 * c_k * delta)
            x -= a_k * gradient_approx

            # Record mid-step energy (one extra eval)
            e_current = float(objective(x))
            n_evals += 1
            tracker.record(e_current)

            _log.debug("SPSA step %d: E = %.8f", k, e_current)

            if tracker.converged:
                _log.info("SPSA converged at step %d.", k)
                break

        _log.info(
            "SPSA finished: %d evals, E = %.8f, converged = %s",
            n_evals,
            tracker.history[-1] if tracker.history else float("nan"),
            tracker.converged,
        )
        return x, tracker.history, tracker.converged, n_evals
