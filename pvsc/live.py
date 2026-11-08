"""Closed-loop live demo over real protocols with simulated devices.

    plant (Modbus TCP server)  : simulated PV inverter + grid meter + house loads, owns the virtual clock
    charge point (OCPP client) : simulated charge point + EV, obeys SetChargingProfile
    controller                 : reads Modbus, talks OCPP (central system), switches boiler / SG-Ready relays

Virtual time runs 5 minutes per tick. The three roles can run in one process (`demo-live`) or as separate
containers (docker compose). Each step k has two passes, which models a fast control loop inside one 5-minute step:
  A  plant publishes the clock; the charge point reports EV power at its current limit; the plant publishes a
     provisional meter reading (a load step has just arrived, nothing has reacted yet);
  B  the controller reads it, sets limits over OCPP and relays over Modbus; the charge point applies them and reports
     the settled EV power; the plant publishes the settled meter reading for the step.
The provisional peak is the transient before the reaction; the settled value is what the step mean looks like.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

import numpy as np
from pymodbus.client import AsyncModbusTcpClient

from . import sunspec as S
from .config import DT_H, STEPS_PER_DAY, VOLTAGE, Site
from .control import Meas, SurplusController
from .data import N_DAYS, gen_base_load, pv_series
from .ocpp_cp import EvModel, connect_charge_point
from .ocpp_cs import CentralSystem

log = logging.getLogger("pvsc.live")
STEP_MIN = 5
SIM_EV_STEP = S.SIM + 10  # charge point: pass A done
SIM_METER_STEP = S.SIM + 11  # plant: provisional meter published
SIM_CTRL_STEP = S.SIM + 12  # controller: decision applied
SIM_EV_STEP2 = S.SIM + 13  # charge point: settled EV power published
SIM_METER_FINAL = S.SIM + 14  # plant: settled meter published
SIM_FINAL_A = S.SIM + 15  # 3 registers: settled import current per phase (0.01 A), written before SIM_METER_FINAL
_ALL_SYNC = (SIM_EV_STEP, SIM_METER_STEP, SIM_CTRL_STEP, SIM_EV_STEP2, SIM_METER_FINAL)
HP_BOOST_KW = 1.5  # extra heat pump draw while SG state 3 is active (demo plant only)


@dataclass
class Scenario:
    day: int  # day of year index 0..365
    arrive_min: int = 8 * 60 + 30
    depart_min: int = 17 * 60 + 30
    need_kwh: float = 14.0
    stress_kw: float = 1.2  # extra uncontrollable load per phase (oven + hob burst) between stress_from and stress_to
    stress_from: float = 11.5
    stress_to: float = 12.5
    seed: int = 7


def limit_scenario() -> Scenario:
    """Winter evening, no PV: the EV must be charged from the grid while the house load peaks.
    Shows the connection limit shaping the charging current."""
    return Scenario(day=344, arrive_min=16 * 60, depart_min=20 * 60 + 30, need_kwh=25.0,
                    stress_kw=1.0, stress_from=17.5, stress_to=19.5)


def best_summer_day(site: Site | None = None) -> int:
    """Day index (Jun-Aug) with the highest simulated PV energy."""
    pv = pv_series("observed", (site or Site()).pv)
    daily = pv[: N_DAYS * STEPS_PER_DAY].reshape(N_DAYS, STEPS_PER_DAY).sum(axis=1)
    lo, hi = 152, 244
    return int(lo + np.argmax(daily[lo:hi]))


class Plant:
    """Simulated inverter, meter and house for one day."""

    def __init__(self, regs: S.PlantRegisters, sc: Scenario, site: Site | None = None, tick_s: float = 0.1):
        self.regs, self.sc, self.tick_s = regs, sc, tick_s
        site = site or Site()
        s0 = sc.day * STEPS_PER_DAY
        self.pv = pv_series("observed", site.pv)[s0: s0 + STEPS_PER_DAY]
        self.base = gen_base_load(sc.seed)[s0: s0 + STEPS_PER_DAY].copy()
        if sc.stress_kw > 0:
            self.base[int(sc.stress_from * 12): int(sc.stress_to * 12)] += sc.stress_kw
        self.day_start_min = sc.day * 1440
        for a in _ALL_SYNC:
            regs.set_words(a, [0xFFFF])
        self.rows: list[dict] = []

    def _read(self, address: int) -> int:
        return self.regs.get_words(address, 1)[0]

    async def _wait(self, address: int, k: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(3.0, 8 * self.tick_s)
        while self._read(address) != (k & 0xFFFF) and loop.time() < deadline:
            await asyncio.sleep(0.002)

    def _grid(self, k: int) -> tuple[tuple[float, float, float], float, float, int, tuple[int, int]]:
        ev_w = float(S.u32_from(self.regs.get_words(S.SIM_EV_W, 2)))
        ev_phases = self._read(S.SIM_EV_PHASES) or 3
        boiler = self._read(S.SIM_BOILER_CMD)
        sg = (self._read(S.SIM_SG_K1), self._read(S.SIM_SG_K2))
        hp_w = HP_BOOST_KW * 1000.0 if sg == (0, 1) else 0.0
        boiler_w = 3000.0 if boiler else 0.0
        load = [self.base[k, p] * 1000.0 + hp_w / 3.0 for p in range(3)]
        load[0] += boiler_w
        if ev_phases == 3:
            load = [x + ev_w / 3.0 for x in load]
        else:
            load[0] += ev_w
        pv_w = self.pv[k] * 1000.0
        grid = tuple(load[p] - pv_w / 3.0 for p in range(3))
        return grid, ev_w, hp_w, ev_phases, sg

    async def run(self, stop: asyncio.Event) -> None:
        for k in range(STEPS_PER_DAY):
            clock = self.day_start_min + k * STEP_MIN
            self.regs.set_words(S.SIM_CLOCK, list(S.u32_words(clock)))
            pv_w = self.pv[k] * 1000.0
            # pass A: provisional reading with the EV still at its previous limit
            await self._wait(SIM_EV_STEP, k)
            grid_a, *_ = self._grid(k)
            self.regs.update(pv_w, grid_a, dt_h=0.0, clock_min=clock)
            self.regs.set_words(SIM_METER_STEP, [k & 0xFFFF])
            # pass B: after the controller reacted and the charge point settled
            await self._wait(SIM_EV_STEP2, k)
            grid, ev_w, hp_w, ev_phases, sg = self._grid(k)
            boiler_w = 3000.0 if self._read(S.SIM_BOILER_CMD) else 0.0
            self.regs.set_words(S.SIM_HP_W, [int(hp_w)])
            self.regs.set_words(S.SIM_BOILER_W, [int(boiler_w)])
            self.regs.update(pv_w, grid, dt_h=DT_H, clock_min=clock)
            self.regs.set_words(SIM_FINAL_A, [S.scaled(max(0.0, g) / VOLTAGE, S.A_SF) for g in grid])
            self.regs.set_words(SIM_METER_FINAL, [k & 0xFFFF])
            self.rows.append({
                "step": k, "hour": k * STEP_MIN / 60.0, "pv_kw": pv_w / 1000.0, "base_kw": float(self.base[k].sum()),
                "ev_kw": ev_w / 1000.0, "ev_phases": ev_phases, "boiler_kw": boiler_w / 1000.0, "hp_kw": hp_w / 1000.0,
                "grid_kw": sum(grid) / 1000.0, "phase_a_max": max(max(0.0, g) / VOLTAGE for g in grid),
                "transient_a_max": max(max(0.0, g) / VOLTAGE for g in grid_a),
                "sg_state": 3 if sg == (0, 1) else 2,
            })
            await asyncio.sleep(self.tick_s)
        stop.set()


async def _connect_modbus(host: str, port: int, tries: int = 120) -> AsyncModbusTcpClient:
    """Connect with retries so the containers can start in any order."""
    for _ in range(tries):
        c = AsyncModbusTcpClient(host, port=port)
        if await c.connect() and c.connected:
            return c
        c.close()
        await asyncio.sleep(1.0)
    raise ConnectionError(f"modbus server {host}:{port} not reachable")


async def _connect_ocpp(url: str, cp_id: str, ev_spec, tries: int = 120):
    for _ in range(tries):
        try:
            return await connect_charge_point(url, cp_id, ev_spec)
        except OSError:
            await asyncio.sleep(1.0)
    raise ConnectionError(f"OCPP central system {url} not reachable")


async def run_charger(ocpp_url: str, modbus_host: str, modbus_port: int, sc: Scenario, stop: asyncio.Event,
                      cp_id: str = "CP_1", site: Site | None = None) -> dict:
    """Charge point + EV simulator: OCPP client plus a Modbus client that reports its power to the plant."""
    site = site or Site()
    cp, conn = await _connect_ocpp(ocpp_url, cp_id, site.ev)
    cp.limit_a = 0.0  # fail-safe: no charging until the energy manager has set a limit
    runner = asyncio.create_task(cp.start())
    mb = await _connect_modbus(modbus_host, modbus_port)
    await cp.boot()
    last_clock = -1
    plugged = False
    delivered_last = 0.0
    try:
        while not stop.is_set():
            rr = await mb.read_holding_registers(S.SIM, count=2)
            if rr.isError():
                await asyncio.sleep(0.01)
                continue
            clock = S.u32_from(rr.registers)
            if clock == last_clock:
                await asyncio.sleep(0.005)
                continue
            last_clock = clock
            step = (clock // STEP_MIN) % STEPS_PER_DAY
            tod = clock % 1440
            if not plugged and sc.arrive_min <= tod < sc.depart_min:
                await cp.plug_in(EvModel(need_kwh=sc.need_kwh, max_a=site.ev.max_a))
                plugged = True
            if plugged and tod >= sc.depart_min:
                await cp.unplug()
                plugged = False
            # pass A: report power at the current limit
            await mb.write_registers(S.SIM_EV_W, [*S.u32_words(int(cp.present_power_kw() * 1000)), cp.phases])
            await mb.write_registers(SIM_EV_STEP, [step & 0xFFFF])
            # wait for the controller's decision, then settle
            for _ in range(5000):
                rr = await mb.read_holding_registers(SIM_CTRL_STEP, count=1)
                if not rr.isError() and rr.registers[0] == (step & 0xFFFF):
                    break
                await asyncio.sleep(0.002)
            cp.advance(STEP_MIN)
            delivered_last = cp.vehicle.delivered_kwh if cp.vehicle else delivered_last
            await cp.send_meter_values()
            await mb.write_registers(S.SIM_EV_W, [*S.u32_words(int(cp.power_w)), cp.phases if cp.power_w > 0 else 3])
            await mb.write_registers(SIM_EV_STEP2, [step & 0xFFFF])
            if step >= STEPS_PER_DAY - 1:  # virtual day finished
                break
    finally:
        mb.close()
        with contextlib.suppress(Exception):
            await conn.close()
        runner.cancel()
    return {"delivered_kwh": round(delivered_last, 2), "profiles": cp.profiles_received}


@dataclass
class ControllerRun:
    rows: list[dict] = field(default_factory=list)
    max_phase_a: float = 0.0
    transient_peak_a: float = 0.0
    steps_over_limit: int = 0
    delivered_kwh: float = 0.0
    profiles_sent: int = 0


async def run_controller(central: CentralSystem, modbus_host: str, modbus_port: int, sc: Scenario,
                         stop: asyncio.Event, cp_id: str = "CP_1", site: Site | None = None) -> ControllerRun:
    site = site or Site()
    ctl = SurplusController(site.ctl, site.ev, site.grid, boiler_kw=site.boiler.power_kw)
    mb = await _connect_modbus(modbus_host, modbus_port)
    out = ControllerRun()
    last_step = -1
    start_wh: float | None = None
    boiler_on = False
    sg = 2
    try:
        while not stop.is_set():
            rr = await mb.read_holding_registers(SIM_METER_STEP, count=1)
            if rr.isError() or rr.registers[0] == last_step:
                await asyncio.sleep(0.005)
                continue
            snap = await S.read_snapshot(mb)
            last_step = (snap.clock_min // STEP_MIN) % STEPS_PER_DAY
            tod = snap.clock_min % 1440
            st = central.states.get(cp_id)
            cp = central.points.get(cp_id)
            # provisional meter reading (before this decision takes effect), positive = import
            a_trans = max(max(0.0, a) for a in snap.grid_phase_a)
            out.transient_peak_a = max(out.transient_peak_a, a_trans)
            ev_ph = [0.0, 0.0, 0.0]
            if st is not None and st.transaction_id is not None:
                for i, ph in enumerate(("L1", "L2", "L3")):
                    ev_ph[i] = st.current_a.get(ph, 0.0) * VOLTAGE / 1000.0
            if st is not None and st.transaction_id is not None and start_wh is None:
                start_wh = st.energy_wh
            if st is not None and st.transaction_id is None:
                start_wh = None
            hp_kw = HP_BOOST_KW if sg == 3 else 0.0
            base = tuple(
                max(0.0, snap.grid_phase_w[p] / 1000.0 + snap.pv_w / 3000.0 - ev_ph[p]
                    - (site.boiler.power_kw if (boiler_on and p == 0) else 0.0) - hp_kw / 3.0)
                for p in range(3)
            )
            connected = st is not None and st.transaction_id is not None
            delivered = ((st.energy_wh - (start_wh or 0.0)) / 1000.0) if connected else 0.0
            m = Meas(
                pv_kw=snap.pv_w / 1000.0, base_ph_kw=base, hp_kw=hp_kw, tank_c=55.0, heating_season=False,
                ev_connected=connected, ev_remaining_kwh=max(0.0, sc.need_kwh - delivered),
                ev_hours_left=max(0.0, (sc.depart_min - tod) / 60.0), ev_session=1 if connected else -1,
            )
            sp = ctl.decide(m)
            if cp is not None and connected:
                if st.last_limit != (sp.ev_amps, sp.ev_phases):
                    if await cp.set_limit(sp.ev_amps, sp.ev_phases):
                        out.profiles_sent += 1
            boiler_on, sg = sp.boiler_on, sp.sg_state
            await mb.write_registers(S.SIM_BOILER_CMD, [int(boiler_on)])
            await mb.write_registers(S.SIM_SG_K1, [0, 1 if sg == 3 else 0])
            await mb.write_registers(SIM_CTRL_STEP, [last_step & 0xFFFF])
            # settled meter reading for this step (one atomic read: step marker + settled phase currents)
            a_meter = 0.0
            for _ in range(5000):
                rr = await mb.read_holding_registers(SIM_METER_FINAL, count=4)
                if not rr.isError() and rr.registers[0] == (last_step & 0xFFFF):
                    a_meter = max(S.unscaled(w, S.A_SF) for w in rr.registers[1:4])
                    break
                await asyncio.sleep(0.002)
            out.max_phase_a = max(out.max_phase_a, a_meter)
            if a_meter > site.grid.limit_a:
                out.steps_over_limit += 1
            out.rows.append({"step": last_step, "ev_amps": sp.ev_amps, "ev_phases": sp.ev_phases,
                             "boiler_on": boiler_on, "sg": sg, "limited": sp.limited, "meter_a_max": a_meter,
                             "transient_a_max": a_trans, "delivered_kwh": delivered})
            out.delivered_kwh = max(out.delivered_kwh, delivered)
            if last_step >= STEPS_PER_DAY - 1:  # virtual day finished
                break
    finally:
        mb.close()
    return out


async def run_live(tick_s: float = 0.05, scenario: Scenario | None = None, site: Site | None = None,
                   modbus_port: int = 5020, ocpp_port: int = 9000) -> dict:
    """Run plant, charger and controller in one process on localhost and return a summary + per-step rows."""
    site = site or Site()
    sc = scenario or Scenario(day=best_summer_day(site))
    regs = S.PlantRegisters()
    server = S.make_server(regs, port=modbus_port)
    server_task = asyncio.create_task(server.serve_forever())
    central = CentralSystem(port=ocpp_port)
    await central.start()
    await asyncio.sleep(0.3)
    stop = asyncio.Event()
    plant = Plant(regs, sc, site, tick_s)
    charger = asyncio.create_task(run_charger(f"ws://127.0.0.1:{central.port}", "127.0.0.1", modbus_port, sc, stop, site=site))
    await central.wait_for("CP_1", 10)
    ctrl = asyncio.create_task(run_controller(central, "127.0.0.1", modbus_port, sc, stop, site=site))
    await asyncio.sleep(0.1)
    try:
        await plant.run(stop)
        await asyncio.sleep(0.3)
        cres = await asyncio.wait_for(ctrl, 5)
        chres = await asyncio.wait_for(charger, 5)
    finally:
        stop.set()
        for t in (ctrl, charger):
            if not t.done():
                t.cancel()
        await central.stop()
        await server.shutdown()
        server_task.cancel()
    rows = plant.rows
    # peak current the house would have seen with the EV at its charger maximum (no limiter), for comparison
    ev_a = site.ev.max_a
    base_peak = max(
        max(max(0.0, plant.base[k, p] * 1000.0 + ev_a * VOLTAGE * 1.0 - plant.pv[k] * 1000.0 / 3.0) / VOLTAGE
            for p in range(3))
        for k in range(STEPS_PER_DAY)
        if sc.arrive_min <= k * STEP_MIN < sc.depart_min
    )
    ev_kwh = sum(r["ev_kw"] for r in rows) * DT_H
    return {
        "day_index": sc.day,
        "steps": len(rows),
        "unlimited_peak_a": round(base_peak, 2),
        "settled_peak_a": round(cres.max_phase_a, 2),
        "transient_peak_a_before_reaction": round(cres.transient_peak_a, 2),
        "steps_over_limit": cres.steps_over_limit,
        "steps_limited_by_controller": sum(1 for r in cres.rows if r["limited"]),
        "ev_kwh": round(ev_kwh, 2),
        "ev_need_kwh": sc.need_kwh,
        "ocpp_profiles_sent": cres.profiles_sent,
        "profiles_accepted_by_cp": len(chres["profiles"]),
        "rows": rows,
        "ctl_rows": cres.rows,
    }
