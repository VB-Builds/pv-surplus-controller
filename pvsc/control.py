"""Surplus controller: decides EV current/phases, boiler relay and SG-Ready state each control step and keeps the
grid connection below its per-phase current limit. Pure logic, no I/O: the simulation and the live demo both call it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import DT_H, VOLTAGE, ControllerConfig, EvSpec, GridConnection


@dataclass
class Meas:
    pv_kw: float
    base_ph_kw: tuple[float, float, float]  # uncontrollable load per phase
    hp_kw: float = 0.0  # heat pump electric draw this step (3 phases, balanced)
    tank_c: float = 55.0
    heating_season: bool = False
    hp_room_to_boost: bool = False  # indoor temperature below the boosted setpoint
    # EV context
    ev_connected: bool = False
    ev_remaining_kwh: float = 0.0
    ev_hours_left: float = 12.0
    ev_session: int = -1


@dataclass
class Setpoint:
    ev_amps: float = 0.0
    ev_phases: int = 3
    boiler_on: bool = False
    sg_state: int = 2
    forced: bool = False
    limited: bool = False  # the grid limit reduced something this step


def phase_headroom_a(grid: GridConnection, net_wo_ev_kw: float) -> float:
    """Amps still available on one phase given its import without the EV (negative net counts as 0)."""
    return grid.limit_a - grid.margin_a - max(0.0, net_wo_ev_kw) * 1000.0 / VOLTAGE


@dataclass
class SurplusController:
    ctl: ControllerConfig
    ev: EvSpec
    grid: GridConnection
    boiler_kw: float = 3.0
    ev_1p_phase: int = 0  # single-phase charging uses L1
    boiler_phase: int = 0
    # state
    ev_on: bool = False
    ev_phases: int = 3
    ev_run_age: int = 0
    ev_phase_age: int = 99
    ev_low: int = 0
    ev_latched: bool = False
    _session: int = -1
    boiler_on: bool = False
    boiler_age: int = 99
    sg: int = 2
    sg_age: int = 99
    log: list = field(default_factory=list)

    # -- helpers ----------------------------------------------------------------------------------
    def _net_wo_ev(self, m: Meas, boiler_on: bool) -> list[float]:
        pv3 = m.pv_kw / 3.0
        net = [m.base_ph_kw[p] + m.hp_kw / 3.0 - pv3 for p in range(3)]
        if boiler_on:
            net[self.boiler_phase] += self.boiler_kw
        return net

    def _pick_phases(self, target_kw: float, forced: bool) -> int:
        if forced:
            return 3
        if not self.ev_on:
            return 3 if target_kw >= self.ctl.phase_up_kw else 1
        want_three = target_kw >= (self.ctl.phase_down_kw if self.ev_phases == 3 else self.ctl.phase_up_kw)
        want = 3 if want_three else 1
        if want != self.ev_phases and self.ev_phase_age < self.ctl.phase_dwell_steps:
            return self.ev_phases
        return want

    # -- main entry -------------------------------------------------------------------------------
    def decide(self, m: Meas, plan_kw: float | None = None) -> Setpoint:
        c, ev = self.ctl, self.ev
        if m.ev_session != self._session:
            self._session = m.ev_session
            self.ev_latched = False
            self.ev_low = 0
            self.ev_on = False
        self.ev_run_age += 1
        self.ev_phase_age += 1
        self.boiler_age += 1
        self.sg_age += 1

        remaining = m.pv_kw - sum(m.base_ph_kw) - m.hp_kw  # surplus before any controllable load
        sp = Setpoint()
        ev_amps, ev_phases = 0.0, self.ev_phases
        boiler_on = self.boiler_on
        sg = self.sg
        limited = False

        # boiler hysteresis and comfort floor first (decision depends on remaining at its priority slot)
        for dev in c.priority:
            if dev == "ev":
                forced = False
                if m.ev_connected and m.ev_remaining_kwh > 0.05:
                    usable_kw = ev.max_kw_3p
                    t_need = m.ev_remaining_kwh / usable_kw + c.deadline_margin_h
                    if m.ev_hours_left <= t_need:
                        self.ev_latched = True
                    forced = self.ev_latched
                    target = max(remaining, plan_kw or 0.0)
                    if forced:
                        target = ev.max_kw_3p
                    on_next = self.ev_on
                    if not self.ev_on:
                        on_next = forced or target >= c.start_kw
                        if on_next:
                            self.ev_run_age = 0
                    else:
                        self.ev_low = self.ev_low + 1 if (target < c.stop_kw and not forced) else 0
                        if self.ev_low >= c.stop_delay_steps and self.ev_run_age >= c.min_run_steps:
                            on_next = False
                    if on_next:
                        ph = self._pick_phases(target, forced)
                        if ph != self.ev_phases and self.ev_on:
                            self.ev_phase_age = 0
                        ev_phases = ph
                        ev_amps = min(ev.max_a, max(ev.min_a, math.floor(target * 1000 / (VOLTAGE * ph))))
                        remaining -= ev.power_kw(ev_amps, ph)
                    self.ev_on = on_next
                    sp.forced = forced
                else:
                    self.ev_on = False
                    self.ev_low = 0
            elif dev == "boiler":
                if m.tank_c < 0:  # no boiler
                    boiler_on = False
                elif m.tank_c < self._boiler_floor():
                    boiler_on = True
                elif m.tank_c >= self._boiler_ceiling():
                    boiler_on = False
                elif self.boiler_on:
                    if remaining < c.boiler_off_kw and self.boiler_age >= c.boiler_min_run_steps:
                        boiler_on = False
                elif remaining >= c.boiler_on_kw and self.boiler_age >= c.boiler_min_run_steps:
                    boiler_on = True
                if boiler_on:
                    remaining -= self.boiler_kw
            elif dev == "sg":
                season_ok = m.heating_season and m.hp_room_to_boost
                want = sg
                if not season_ok:
                    want = 2
                elif sg == 2 and remaining >= c.sg_on_kw:
                    want = 3
                elif sg == 3 and remaining < c.sg_off_kw:
                    want = 2
                if want != sg and (self.sg_age >= c.sg_min_steps or not season_ok):
                    sg = want
                    self.sg_age = 0

        # ---- grid connection limit: shed boiler first, then reduce / re-phase the EV
        net = self._net_wo_ev(m, boiler_on)
        if boiler_on and net[self.boiler_phase] * 1000 / VOLTAGE > self.grid.limit_a - self.grid.margin_a:
            boiler_on = False
            limited = True
            net = self._net_wo_ev(m, False)
        if ev_amps > 0:
            head = [phase_headroom_a(self.grid, n) for n in net]
            allow3 = min(head)
            allow1 = head[self.ev_1p_phase]
            if ev_phases == 3 and allow3 < ev_amps:
                limited = True
                if allow3 >= ev.min_a:
                    ev_amps = math.floor(allow3)
                elif allow1 >= ev.min_a:
                    ev_phases, ev_amps = 1, min(ev.max_a, math.floor(allow1))
                else:
                    ev_amps = 0.0
            elif ev_phases == 1 and allow1 < ev_amps:
                limited = True
                ev_amps = math.floor(allow1) if allow1 >= ev.min_a else 0.0
            if ev_amps == 0.0:
                self.ev_on = False
            elif ev_phases != self.ev_phases:
                self.ev_phase_age = 0
        if boiler_on != self.boiler_on:
            self.boiler_age = 0
        self.boiler_on = boiler_on
        self.sg = sg
        if ev_amps > 0:
            self.ev_phases = ev_phases
        sp.ev_amps, sp.ev_phases, sp.boiler_on, sp.sg_state, sp.limited = ev_amps, ev_phases, boiler_on, sg, limited
        return sp

    def _boiler_floor(self) -> float:
        return self.ctl.boiler_min_c

    def _boiler_ceiling(self) -> float:
        return self.ctl.boiler_max_c


def dumb_setpoint(m: Meas, hour_of_day: float, tank_c: float, ev: EvSpec, boiler_was_on: bool = False) -> Setpoint:
    """Reference behaviour: EV full power from plug-in, boiler on a night timer, no SG signal."""
    sp = Setpoint()
    if m.ev_connected and m.ev_remaining_kwh > 0.05:
        sp.ev_amps, sp.ev_phases = ev.max_a, 3
    night = hour_of_day >= 22 or hour_of_day < 6
    sp.boiler_on = night and (tank_c < 53.0 or (boiler_was_on and tank_c < 58.0))
    return sp


def energy_for_step(amps: float, phases: int, ev: EvSpec) -> float:
    return ev.power_kw(amps, phases) * DT_H
