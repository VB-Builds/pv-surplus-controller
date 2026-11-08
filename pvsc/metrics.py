"""Metrics from a simulation Result, over any step range (month, season, year)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DT_H, STEPS_PER_DAY, Site
from .data import buy_series
from .sim import Result

MONTH_DAYS_2024 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SEASONS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}


def month_slice(month: int) -> slice:
    start = sum(MONTH_DAYS_2024[: month - 1]) * STEPS_PER_DAY
    return slice(start, start + MONTH_DAYS_2024[month - 1] * STEPS_PER_DAY)


def steps_of(months: tuple[int, ...]) -> np.ndarray:
    idx = [np.arange(s.start, s.stop) for s in (month_slice(m) for m in months)]
    return np.concatenate(idx)


def compute(r: Result, site: Site, steps: np.ndarray | slice | None = None) -> dict:
    sel = slice(0, r.n) if steps is None else steps
    grid = r.grid_kw[sel]
    imp = np.clip(grid, 0.0, None)
    exp = np.clip(-grid, 0.0, None)
    pv = r.pv[sel]
    pv_kwh = float(pv.sum() * DT_H)
    exp_kwh = float(exp.sum() * DT_H)
    imp_kwh = float(imp.sum() * DT_H)
    load_kwh = float((r.base_kw[sel] + r.ev_kw[sel] + r.hp_kw[sel] + r.boiler_kw[sel]).sum() * DT_H)
    sw = r.sw_steps[sel].sum(axis=0)
    price = buy_series(site.tariff, r.n)[sel]
    cost = float((imp * price).sum() * DT_H) - exp_kwh * site.tariff.feed_in
    imp15 = pd.Series(imp).rolling(3).mean().max()
    phase_max = float(r.phase_a[sel].max())
    # sessions fully inside the selected window
    lo = 0 if isinstance(sel, slice) and sel.start is None else (sel.start if isinstance(sel, slice) else int(sel.min()))
    hi = r.n if isinstance(sel, slice) and sel.stop is None else (sel.stop if isinstance(sel, slice) else int(sel.max()) + 1)
    sess = [s for s in r.sessions if lo <= s["end"] <= hi and not s.get("open")]
    need = sum(s["need"] for s in sess)
    got = sum(min(s["delivered"], s["need"]) for s in sess)
    ready = sum(1 for s in sess if s["delivered"] >= s["need"] - 0.5)
    return {
        "pv_kwh": round(pv_kwh, 1),
        "load_kwh": round(load_kwh, 1),
        "ev_kwh": round(float(r.ev_kw[sel].sum() * DT_H), 1),
        "import_kwh": round(imp_kwh, 1),
        "export_kwh": round(exp_kwh, 1),
        "self_consumption_pct": round(100.0 * (1.0 - exp_kwh / pv_kwh), 1) if pv_kwh > 0 else float("nan"),
        "self_sufficiency_pct": round(100.0 * (1.0 - imp_kwh / load_kwh), 1) if load_kwh > 0 else float("nan"),
        "cost_chf": round(cost, 2),
        "grid_peak_kw_5min": round(float(imp.max()), 2),
        "grid_peak_kw_15min": round(float(imp15), 2) if not np.isnan(imp15) else 0.0,
        "max_phase_a": round(phase_max, 1),
        "steps_over_limit": int((r.phase_a[sel].max(axis=1) > site.grid.limit_a + 1e-9).sum()),
        "ev_sessions": len(sess),
        "ev_need_kwh": round(need, 1),
        "ev_delivered_by_departure_pct": round(100.0 * got / need, 1) if need > 0 else float("nan"),
        "ev_sessions_ready_pct": round(100.0 * ready / len(sess), 1) if sess else float("nan"),
        "switches_total": int(sw.sum()),
        "ev_relay_switches": int(sw[0] + sw[1]),
        "boiler_relay_switches": int(sw[2]),
        "sg_switches": int(sw[3]),
    }


def compare(results: dict[str, Result], site: Site, steps=None) -> pd.DataFrame:
    rows = {k: compute(r, site, steps) for k, r in results.items()}
    df = pd.DataFrame(rows).T
    base = rows["dumb"]["cost_chf"] if "dumb" in rows else None
    if base is not None:
        df["chf_saved_vs_dumb"] = [round(base - rows[k]["cost_chf"], 2) for k in df.index]
    return df
