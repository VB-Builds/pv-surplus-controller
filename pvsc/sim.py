"""Year simulation at 5-minute resolution for three strategies:

dumb       EV at full power from plug-in, boiler on a night timer, no SG signal (reference).
rule       surplus controller: hysteresis, 1p/3p switching, deadline fallback, SG-Ready, boiler, grid limit.
optimised  rule + hourly re-planned LP (PV forecast) that decides when grid energy may be used for the EV.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import data as D
from .config import DT_H, STEPS_PER_DAY, STEPS_PER_HOUR, VOLTAGE, Site
from .control import Meas, SurplusController, dumb_setpoint
from .devices import Battery, Boiler, HeatPump
from .plan import plan_ev, slot_lengths

STRATEGIES = ("dumb", "rule", "optimised")
REPLAN_HOURS = 3


@dataclass
class Inputs:
    pv: np.ndarray
    pv_fc: np.ndarray
    t_out: np.ndarray
    base_ph: np.ndarray
    dhw: np.ndarray
    sessions: list[D.Session]
    scenario: str = "mixed"

    @property
    def n(self) -> int:
        return len(self.pv)


def build_inputs(scenario: str = "mixed", site: Site | None = None, data_dir=None, n_days: int = D.N_DAYS,
                 seed: int = 7) -> Inputs:
    site = site or Site()
    n = n_days * STEPS_PER_DAY
    return Inputs(
        pv=D.pv_series("observed", site.pv, data_dir)[:n],
        pv_fc=D.pv_series("forecast", site.pv, data_dir)[:n],
        t_out=D.temperature_series(data_dir)[:n],
        base_ph=D.gen_base_load(seed, n_days),
        dhw=D.gen_dhw_draw(seed + 4, n_days, site.boiler.draw_kwh_per_day),
        sessions=[s for s in D.gen_ev_sessions(scenario, site.ev, seed + 14, n_days) if s.start < n],
        scenario=scenario,
    )


@dataclass
class Result:
    strategy: str
    scenario: str
    n: int
    pv: np.ndarray
    base_kw: np.ndarray
    ev_kw: np.ndarray
    hp_kw: np.ndarray
    boiler_kw: np.ndarray
    bat_kw: np.ndarray  # + charging, - discharging
    grid_kw: np.ndarray  # + import, - export
    phase_a: np.ndarray  # (n, 3) import current per phase
    soc: np.ndarray
    t_in: np.ndarray
    tank_c: np.ndarray
    sw_steps: np.ndarray | None = None  # per-step switch events: ev_start_stop, ev_phase, boiler, sg
    sessions: list = field(default_factory=list)  # dicts per session
    limited_steps: int = 0
    plans: int = 0


def _slot_mean(cs: np.ndarray, a: int, b: int) -> float:
    return float((cs[b] - cs[a]) / (b - a))


def run(strategy: str, inp: Inputs, site: Site | None = None) -> Result:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    site = site or Site()
    n = inp.n
    ev_spec, tariff = site.ev, site.tariff
    ctl = SurplusController(site.ctl, ev_spec, site.grid, boiler_kw=site.boiler.power_kw)
    hp = HeatPump(site.hp, t_in=site.hp.setpoint_c)
    boiler = Boiler(site.boiler, t_c=site.boiler.normal_c)
    bat = Battery(site.battery)
    base_total = inp.base_ph.sum(axis=1)

    # forecast helpers for the planner
    cs_pv = np.concatenate([[0.0], np.cumsum(inp.pv_fc)])
    days = n // STEPS_PER_DAY
    bt = base_total[: days * STEPS_PER_DAY].reshape(days, STEPS_PER_DAY)
    prof = np.empty_like(bt)
    for d in range(days):
        lo = max(0, d - 7)
        prof[d] = bt[lo:d].mean(axis=0) if d > 0 else bt[:7].mean(axis=0)
    cs_base = np.concatenate([[0.0], np.cumsum(prof.reshape(-1))])
    cs_buy = np.concatenate([[0.0], np.cumsum(D.buy_series(tariff, n))])
    import_limit_kw = 3 * site.grid.limit_a * VOLTAGE / 1000.0

    out = {k: np.zeros(n) for k in ("ev", "hp", "boiler", "bat", "grid", "soc", "t_in", "tank")}
    phase_a = np.zeros((n, 3))
    sw_arr = np.zeros((n, 4), dtype=np.int8)  # ev_start_stop, ev_phase, boiler, sg
    sessions_out: list[dict] = []
    sess = inp.sessions
    sp_i = 0
    delivered = 0.0
    plan = None
    plan_hour = -1
    n_plans = 0
    limited_steps = 0
    prev = None  # previous setpoint
    sg_prev = 2
    cur_sess = -1

    for i in range(n):
        while sp_i < len(sess) and i >= sess[sp_i].end:
            s = sess[sp_i]
            sessions_out.append({"start": s.start, "end": s.end, "need": s.need_kwh, "delivered": delivered})
            sp_i += 1
            delivered = 0.0
        connected = sp_i < len(sess) and sess[sp_i].start <= i < sess[sp_i].end
        remaining = (sess[sp_i].need_kwh - delivered) if connected else 0.0
        hours_left = (sess[sp_i].end - i) / STEPS_PER_HOUR if connected else 12.0
        if connected and cur_sess != sp_i:
            cur_sess = sp_i
            plan, plan_hour = None, -1
        t_out = inp.t_out[i]
        hod = (i % STEPS_PER_DAY) / STEPS_PER_HOUR

        hp.thermostat(t_out, sg_prev)
        hp_kw = hp.electric_kw(t_out)
        m = Meas(
            pv_kw=inp.pv[i],
            base_ph_kw=(inp.base_ph[i, 0], inp.base_ph[i, 1], inp.base_ph[i, 2]),
            hp_kw=hp_kw,
            tank_c=boiler.t_c,
            heating_season=t_out < site.hp.heating_limit_c,
            hp_room_to_boost=hp.t_in < site.hp.setpoint_c + site.hp.boost_c - 0.1,
            ev_connected=connected,
            ev_remaining_kwh=remaining,
            ev_hours_left=hours_left,
            ev_session=sp_i if connected else -1,
        )
        if strategy == "dumb":
            sp = dumb_setpoint(m, hod, boiler.t_c, ev_spec, prev is not None and prev[2])
        else:
            plan_kw = None
            if strategy == "optimised" and connected and remaining > 0.05:
                hour = i // (STEPS_PER_HOUR * REPLAN_HOURS)
                if hour != plan_hour or plan is None:
                    end = sess[sp_i].end
                    dts = slot_lengths(i, end)
                    edges = (i + np.concatenate([[0], np.cumsum(np.rint(dts * STEPS_PER_HOUR))])).astype(int)
                    pvs = np.array([_slot_mean(cs_pv, edges[k], edges[k + 1]) for k in range(len(dts))])
                    bs = np.array([_slot_mean(cs_base, edges[k], edges[k + 1]) for k in range(len(dts))])
                    prices = np.array([_slot_mean(cs_buy, edges[k], edges[k + 1]) for k in range(len(dts))])
                    plan = plan_ev(remaining, pvs, bs, dts, prices, tariff.feed_in, ev_spec.max_kw_3p, import_limit_kw)
                    plan = replace(plan, start_step=i)
                    plan_hour = hour
                    n_plans += 1
                plan_kw = plan.kw_at(i)
            sp = ctl.decide(m, plan_kw)
            if sp.limited:
                limited_steps += 1

        # ---- apply setpoints to the physics
        ev_kw = 0.0
        phases = sp.ev_phases
        if connected and sp.ev_amps >= ev_spec.min_a and remaining > 0.0:
            ev_kw = min(ev_spec.power_kw(sp.ev_amps, phases), remaining / DT_H)
            delivered += ev_kw * DT_H
        bo_kw = boiler.step(sp.boiler_on, inp.dhw[i])
        hp.step(t_out)
        nonev = float(base_total[i]) + hp_kw + bo_kw
        surplus = inp.pv[i] - nonev - ev_kw
        ch = d = 0.0
        if surplus > 0:
            ch = bat.charge(surplus)
            grid = -(surplus - ch)
        else:
            deficit = -surplus
            d = bat.discharge(min(deficit, max(0.0, nonev - inp.pv[i])))
            grid = deficit - d
        gen_p = (inp.pv[i] + d - ch) / 3.0
        load_p = [inp.base_ph[i, p] + hp_kw / 3.0 for p in range(3)]
        load_p[0] += bo_kw
        if ev_kw > 0:
            if phases == 3:
                load_p = [x + ev_kw / 3.0 for x in load_p]
            else:
                load_p[0] += ev_kw
        for p in range(3):
            phase_a[i, p] = max(0.0, load_p[p] - gen_p) * 1000.0 / VOLTAGE

        # ---- bookkeeping
        out["ev"][i], out["hp"][i], out["boiler"][i] = ev_kw, hp_kw, bo_kw
        out["bat"][i], out["grid"][i], out["soc"][i] = ch - d, grid, bat.soc
        out["t_in"][i], out["tank"][i] = hp.t_in, boiler.t_c
        if prev is not None:
            was, now = prev[0] > 0, ev_kw > 0
            if was != now:
                sw_arr[i, 0] = 1
            elif was and now and prev[1] != phases:
                sw_arr[i, 1] = 1
            if prev[2] != sp.boiler_on:
                sw_arr[i, 2] = 1
            if prev[3] != sp.sg_state:
                sw_arr[i, 3] = 1
        prev = (ev_kw, phases, sp.boiler_on, sp.sg_state)
        sg_prev = sp.sg_state

    if sp_i < len(sess) and sess[sp_i].start < n:
        s = sess[sp_i]
        sessions_out.append({"start": s.start, "end": s.end, "need": s.need_kwh, "delivered": delivered,
                             "open": s.end > n})
    return Result(
        strategy=strategy, scenario=inp.scenario, n=n, pv=inp.pv.copy(), base_kw=base_total.copy(),
        ev_kw=out["ev"], hp_kw=out["hp"], boiler_kw=out["boiler"], bat_kw=out["bat"], grid_kw=out["grid"],
        phase_a=phase_a, soc=out["soc"], t_in=out["t_in"], tank_c=out["tank"], sw_steps=sw_arr,
        sessions=sessions_out, limited_steps=limited_steps, plans=n_plans,
    )
