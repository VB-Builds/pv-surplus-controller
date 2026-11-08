"""Day-ahead style EV charging plan: a small linear programme over hourly slots until departure.

Minimise  buy*import - feed_in*export  subject to  import - export = base + EV - PV  per slot,
EV energy until departure >= remaining need, EV power <= charger max, import <= connection limit.
Inputs are a PV forecast and a typical base load, not the realised values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Plan:
    kw: np.ndarray  # EV power per slot
    dt_h: np.ndarray  # slot lengths
    shortfall_kwh: float
    cost_chf: float
    start_step: int = 0

    def kw_at(self, step: int, steps_per_hour: int = 12) -> float:
        """Planned EV power at a simulation step (0 after the horizon)."""
        edges = self.start_step + np.concatenate([[0], np.cumsum(np.rint(self.dt_h * steps_per_hour))])
        k = int(np.searchsorted(edges, step, side="right")) - 1
        return float(self.kw[k]) if 0 <= k < len(self.kw) else 0.0

    @property
    def now_kw(self) -> float:
        return float(self.kw[0]) if len(self.kw) else 0.0


def slot_lengths(now_step: int, end_step: int, steps_per_hour: int = 12, max_hours: int = 36) -> np.ndarray:
    """Clock-aligned hourly slots from now to departure (first and last may be partial)."""
    end_step = min(end_step, now_step + max_hours * steps_per_hour)
    edges = [now_step]
    nxt = (now_step // steps_per_hour + 1) * steps_per_hour
    while nxt < end_step:
        edges.append(nxt)
        nxt += steps_per_hour
    edges.append(end_step)
    return np.diff(edges) / steps_per_hour


def plan_ev(
    remaining_kwh: float,
    pv_kw: np.ndarray,
    base_kw: np.ndarray,
    dt_h: np.ndarray,
    buy: float | np.ndarray,
    feed_in: float,
    max_kw: float,
    import_limit_kw: float,
) -> Plan:
    """Solve the LP. pv_kw, base_kw, dt_h are per-slot arrays of equal length."""
    t = len(dt_h)
    if t == 0 or remaining_kwh <= 0:
        return Plan(np.zeros(t), np.asarray(dt_h, dtype=float), max(0.0, remaining_kwh), 0.0)
    if not (len(pv_kw) == len(base_kw) == t):
        raise ValueError("pv_kw, base_kw and dt_h must have equal length")
    # variables: e[0..t), g[0..t), x[0..t), s (shortfall)
    n = 3 * t + 1
    c = np.zeros(n)
    c[:t] = 1e-5 * np.arange(t) * dt_h  # tie-break: prefer earlier slots when costs are equal
    c[t:2 * t] = np.asarray(buy, dtype=float) * dt_h
    c[2 * t:3 * t] = -feed_in * dt_h
    c[3 * t] = 1e3
    a_eq = np.zeros((t, n))
    for i in range(t):
        a_eq[i, i] = -1.0
        a_eq[i, t + i] = 1.0
        a_eq[i, 2 * t + i] = -1.0
    b_eq = np.asarray(base_kw, dtype=float) - np.asarray(pv_kw, dtype=float)
    a_ub = np.zeros((1, n))
    a_ub[0, :t] = -dt_h
    a_ub[0, 3 * t] = -1.0
    b_ub = np.array([-remaining_kwh])
    bounds = [(0, max_kw)] * t + [(0, import_limit_kw)] * t + [(0, None)] * t + [(0, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"planner LP failed: {res.message}")
    x = res.x
    cost = float(np.sum(dt_h * (np.asarray(buy, dtype=float) * x[t:2 * t] - feed_in * x[2 * t:3 * t])))
    return Plan(x[:t].copy(), np.asarray(dt_h, dtype=float), float(x[3 * t]), cost)
