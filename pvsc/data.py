"""Inputs: real irradiance (Open-Meteo, Zurich 2024) turned into PV power; simulated household load, hot-water draw
and EV arrival/departure scenarios (seeded, reproducible)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STEPS_PER_DAY, STEPS_PER_HOUR, EvSpec, PvSpec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEAR = 2024
N_DAYS = 366
N_STEPS = N_DAYS * STEPS_PER_DAY
SCENARIOS = ("mixed", "commuter", "home")
LOAD_SCALE = 0.76
PHASE_CAP_KW = 3.2  # uncontrollable load per phase is capped at ~14 A (scenario fixture)


def load_irradiance(kind: str = "observed", data_dir: Path | None = None) -> pd.DataFrame:
    """kind: 'observed' (reanalysis) or 'forecast' (archived model forecast)."""
    path = (data_dir or DATA_DIR) / f"zurich_{YEAR}_{kind}.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    if len(df) != N_DAYS * 24:
        raise ValueError(f"{path.name}: expected {N_DAYS * 24} hourly rows, got {len(df)}")
    return df


def pv_power_kw(gti_wm2: np.ndarray, temp_c: np.ndarray, spec: PvSpec) -> np.ndarray:
    """AC power of the PV system from plane-of-array irradiance and air temperature."""
    gti = np.clip(np.nan_to_num(np.asarray(gti_wm2, dtype=float)), 0.0, None)
    t_cell = np.asarray(temp_c, dtype=float) + 0.03 * gti
    derate = np.clip(1.0 + spec.temp_coeff * (t_cell - 25.0), 0.5, 1.2)
    p = spec.kwp * gti / 1000.0 * derate * spec.system_factor
    return np.minimum(p, spec.inverter_kw)


def hourly_to_steps(hourly: np.ndarray) -> np.ndarray:
    """Hourly values are means over the preceding hour (value at t covers t-1h..t), so they belong at t-30min.
    Linear interpolation onto the 5-minute grid, clipped at >= 0."""
    h = np.asarray(hourly, dtype=float)
    centres = (np.arange(len(h)) - 0.5) * STEPS_PER_HOUR
    x = np.arange(len(h) * STEPS_PER_HOUR)
    return np.clip(np.interp(x, centres, h), 0.0, None)


def pv_series(kind: str, spec: PvSpec, data_dir: Path | None = None) -> np.ndarray:
    df = load_irradiance(kind, data_dir)
    return hourly_to_steps(pv_power_kw(df.gti_wm2.to_numpy(), df.temp_c.to_numpy(), spec))


def temperature_series(data_dir: Path | None = None) -> np.ndarray:
    df = load_irradiance("observed", data_dir)
    return hourly_to_steps(df.temp_c.to_numpy() + 50.0) - 50.0  # shift keeps negatives through the clip


# ----------------------------------------------------------------------------- household load

_WEEKDAY = [0.18, 0.15, 0.14, 0.14, 0.15, 0.25, 0.45, 0.55, 0.35, 0.25, 0.24, 0.28,
            0.35, 0.28, 0.24, 0.25, 0.32, 0.48, 0.62, 0.62, 0.50, 0.40, 0.28, 0.21]
_WEEKEND = [0.19, 0.16, 0.15, 0.14, 0.14, 0.16, 0.20, 0.32, 0.45, 0.50, 0.50, 0.52,
            0.55, 0.45, 0.38, 0.35, 0.38, 0.50, 0.60, 0.62, 0.52, 0.42, 0.30, 0.22]


def _add(arr: np.ndarray, day: int, start_h: float, dur_min: float, kw: float, phases: tuple[int, ...]) -> None:
    s = day * STEPS_PER_DAY + int(start_h * STEPS_PER_HOUR)
    e = min(s + max(1, int(round(dur_min / 5))), arr.shape[0])
    for p in phases:
        arr[s:e, p] += kw


def gen_base_load(seed: int = 7, n_days: int = N_DAYS) -> np.ndarray:
    """Simulated uncontrollable household load, shape (steps, 3 phases) in kW. About 3.7 MWh/a.
    Smooth daily shape + random appliance events (cooking, dish washer, washing machine, dryer, oven on 3 phases)."""
    rng = np.random.default_rng(seed)
    n = n_days * STEPS_PER_DAY
    out = np.zeros((n, 3))
    hour = np.arange(n) / STEPS_PER_HOUR
    hod = hour % 24
    day = (hour // 24).astype(int)
    weekend = (day % 7) >= 5
    shape = np.where(weekend, np.interp(hod, np.arange(24) + 0.5, _WEEKEND, period=24),
                     np.interp(hod, np.arange(24) + 0.5, _WEEKDAY, period=24))
    season = 1.0 + 0.12 * np.cos(2 * np.pi * (day - 15) / 366)  # more lighting/standby in winter
    smooth = shape * season
    out += (smooth / 3.0)[:, None]
    for d in range(n_days):
        wk = d % 7 >= 5
        ph = int(rng.integers(0, 3))
        _add(out, d, 17.5 + rng.uniform(0, 1.5), rng.uniform(25, 45), rng.uniform(1.2, 2.2), (ph,))  # cooking
        if rng.random() < 0.6:  # dish washer
            _add(out, d, rng.choice([12.5, 20.0, 20.5, 21.0]), 90, 1.6, (int(rng.integers(0, 3)),))
        if rng.random() < (0.55 if not wk else 0.8):  # washing machine
            _add(out, d, rng.uniform(8, 16) if not wk else rng.uniform(9, 14), 60, 1.9, (int(rng.integers(0, 3)),))
        if rng.random() < 0.3:  # dryer
            _add(out, d, rng.uniform(9, 19), 120, 0.8, (int(rng.integers(0, 3)),))
        if rng.random() < 0.4:  # oven on three phases
            _add(out, d, rng.uniform(17.5, 19.0), 35, 1.6, (0, 1, 2))
        for _ in range(int(rng.integers(1, 4))):  # kettle / coffee machine
            _add(out, d, rng.uniform(6.2, 8.5) if not wk else rng.uniform(8, 10), 5, 1.8, (int(rng.integers(0, 3)),))
    return np.minimum(out * LOAD_SCALE, PHASE_CAP_KW)


def gen_dhw_draw(seed: int = 11, n_days: int = N_DAYS, kwh_per_day: float = 4.5) -> np.ndarray:
    """Thermal energy drawn from the hot-water tank per step (kWh)."""
    rng = np.random.default_rng(seed)
    n = n_days * STEPS_PER_DAY
    hod = (np.arange(n) / STEPS_PER_HOUR) % 24
    w = np.exp(-0.5 * ((hod - 7.0) / 1.0) ** 2) + 0.8 * np.exp(-0.5 * ((hod - 19.5) / 1.5) ** 2) + 0.05
    w = w * rng.uniform(0.7, 1.3, n)
    w = w.reshape(n_days, STEPS_PER_DAY)
    w = w / w.sum(axis=1, keepdims=True) * kwh_per_day * rng.uniform(0.8, 1.2, (n_days, 1))
    return w.reshape(-1)


# ----------------------------------------------------------------------------- EV scenarios


@dataclass(frozen=True)
class Session:
    start: int  # step index of plug-in
    end: int  # step index of departure
    need_kwh: float  # AC energy required to be back at target SoC
    trip_km: float = 0.0


def gen_ev_sessions(scenario: str, ev: EvSpec, seed: int = 21, n_days: int = N_DAYS) -> list[Session]:
    """Plug-in sessions. 'mixed': commute Mon/Tue/Thu, Wed/Fri 40% commute else home office, weekend errands.
    'commuter': commute every weekday. 'home': works from home, short errands only.
    The energy needed after a trip is what that trip consumed (car was at target before leaving)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    trips: list[tuple[int, int, float]] = []  # (depart_step, arrive_step, km)
    for d in range(n_days):
        wd = d % 7
        base = d * STEPS_PER_DAY
        if wd < 5:
            commute = {"commuter": True, "mixed": wd in (0, 1, 3) or rng.random() < 0.4, "home": False}[scenario]
            if commute:
                dep = base + int((7.0 + rng.uniform(-0.5, 0.5)) * STEPS_PER_HOUR)
                arr = base + int((18.0 + rng.uniform(-0.75, 0.75)) * STEPS_PER_HOUR)
                trips.append((dep, arr, float(np.clip(rng.normal(60, 10), 30, 100))))
            elif scenario == "home" and rng.random() < 0.3:
                dep = base + int(rng.uniform(10, 14) * STEPS_PER_HOUR)
                trips.append((dep, dep + int(rng.uniform(1, 3) * STEPS_PER_HOUR), float(rng.uniform(10, 40))))
        elif rng.random() < 0.6:
            dep = base + int(rng.uniform(9, 12) * STEPS_PER_HOUR)
            trips.append((dep, dep + int(rng.uniform(3, 6) * STEPS_PER_HOUR), float(rng.uniform(20, 120))))
    sessions = []
    for i, (_, arr, km) in enumerate(trips):
        nxt = trips[i + 1][0] if i + 1 < len(trips) else n_days * STEPS_PER_DAY
        need = km * ev.consumption_kwh_100km / 100.0 / ev.efficiency
        if nxt - arr >= 6:
            sessions.append(Session(arr, nxt, float(min(need, ev.capacity_kwh * 0.8)), km))
    return sessions


def buy_series(tariff, n: int) -> np.ndarray:
    """Purchase price per step (CHF/kWh)."""
    if tariff.nt_buy is None:
        return np.full(n, tariff.buy)
    return np.array([tariff.buy_at(i) for i in range(n)])
