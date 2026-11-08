"""Physical device models used by the simulation (heat pump + building, hot-water tank, battery, SG-Ready mapping)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import DT_H, BatterySpec, BoilerSpec, HeatPumpSpec

# SG-Ready style signal: two dry contacts (K1, K2) select one of four operating states.
SG_STATES = {
    1: (1, 0),  # utility lock-out (not used by this controller)
    2: (0, 0),  # normal operation
    3: (0, 1),  # recommended: raise setpoint, use surplus
    4: (1, 1),  # forced: maximum run (not used by this controller)
}


def sg_contacts(state: int) -> tuple[int, int]:
    if state not in SG_STATES:
        raise ValueError(f"SG-Ready state must be 1..4, got {state}")
    return SG_STATES[state]


def cop(temp_out_c: float) -> float:
    """Air-source COP, simple linear fit, clipped (generic, not a product curve)."""
    return min(5.0, max(2.2, 3.4 + 0.08 * temp_out_c))


@dataclass
class HeatPump:
    spec: HeatPumpSpec
    t_in: float = 20.5
    on: bool = False

    def thermostat(self, t_out: float, sg_state: int) -> bool:
        """Update the on/off state from indoor temperature; SG state 3 raises the setpoint."""
        s = self.spec
        if t_out >= s.heating_limit_c:
            self.on = False
            return self.on
        sp = s.setpoint_c + (s.boost_c if sg_state == 3 else 0.0)
        if self.t_in < sp - s.band_c:
            self.on = True
        elif self.t_in > sp + s.band_c:
            self.on = False
        return self.on

    def electric_kw(self, t_out: float) -> float:
        return self.spec.thermal_kw / cop(t_out) if self.on else 0.0

    def step(self, t_out: float) -> None:
        q = self.spec.thermal_kw if self.on else 0.0
        loss = self.spec.ua_kw_per_k * (self.t_in - t_out)
        self.t_in += DT_H * (q - loss) / self.spec.capacity_kwh_per_k


@dataclass
class Boiler:
    spec: BoilerSpec
    t_c: float = 55.0

    def step(self, on: bool, draw_kwh: float, ambient_c: float = 20.0) -> float:
        """Advance one step, return electric kW drawn."""
        p = self.spec.power_kw if on else 0.0
        e = p * DT_H - draw_kwh - self.spec.loss_kw * DT_H * max(0.0, (self.t_c - ambient_c) / 35.0)
        self.t_c += e / self.spec.capacity_kwh_per_k
        self.t_c = min(self.t_c, self.spec.max_c + 2.0)
        return p


@dataclass
class Battery:
    spec: BatterySpec
    soc: float = 0.3

    def charge(self, kw: float) -> float:
        """Try to charge with kw from the DC/AC bus; return what was accepted (kW)."""
        s = self.spec
        room = (1.0 - self.soc) * s.capacity_kwh / (DT_H * s.efficiency)
        p = max(0.0, min(kw, s.power_kw, room))
        self.soc += p * DT_H * s.efficiency / s.capacity_kwh
        return p

    def discharge(self, kw: float) -> float:
        """Try to supply kw to the bus; return what was delivered (kW)."""
        s = self.spec
        avail = (self.soc - s.min_soc) * s.capacity_kwh * s.efficiency / DT_H
        p = max(0.0, min(kw, s.power_kw, avail))
        self.soc -= p * DT_H / s.efficiency / s.capacity_kwh
        return p
