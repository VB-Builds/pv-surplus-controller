"""Simulated OCPP 1.6J charge point with one connector and a simulated EV.

The protocol side is real (BootNotification, StatusNotification, Authorize, StartTransaction, MeterValues,
StopTransaction, and it obeys SetChargingProfile); the vehicle and the electrical side are simulated."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import websockets
from ocpp.routing import on
from ocpp.v16 import ChargePoint as _ChargePoint
from ocpp.v16 import call, call_result
from ocpp.v16.datatypes import MeterValue, SampledValue
from ocpp.v16.enums import (
    Action,
    ChargePointErrorCode,
    ChargePointStatus,
    ChargingProfileStatus,
    Measurand,
    Phase,
    UnitOfMeasure,
)
from websockets.asyncio.client import connect

from .config import VOLTAGE, EvSpec

log = logging.getLogger("pvsc.ocpp_cp")
SUBPROTOCOL = "ocpp1.6"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class EvModel:
    """A vehicle that wants `need_kwh` and accepts what the charge point offers within its onboard limits."""

    need_kwh: float
    delivered_kwh: float = 0.0
    max_a: float = 16.0
    phases_supported: int = 3

    @property
    def done(self) -> bool:
        return self.delivered_kwh >= self.need_kwh - 1e-9


class SimChargePoint(_ChargePoint):
    def __init__(self, cp_id: str, connection, ev_spec: EvSpec | None = None):
        super().__init__(cp_id, connection)
        self.ev_spec = ev_spec or EvSpec()
        self.limit_a = self.ev_spec.max_a  # charge point default until a profile arrives
        self.phases = 3
        self.vehicle: EvModel | None = None
        self.transaction_id: int | None = None
        self.energy_wh = 0.0
        self.power_w = 0.0
        self.profiles_received: list[tuple[float, int]] = []
        self.rejected = 0

    # ---- central system -> charge point
    @on(Action.set_charging_profile)
    async def on_set_charging_profile(self, connector_id, cs_charging_profiles, **kwargs):
        try:
            period = cs_charging_profiles["charging_schedule"]["charging_schedule_period"][0]
            unit = cs_charging_profiles["charging_schedule"]["charging_rate_unit"]
            limit = float(period["limit"])
            phases = int(period.get("number_phases", self.phases))
        except (KeyError, IndexError, TypeError, ValueError):
            self.rejected += 1
            return call_result.SetChargingProfile(status=ChargingProfileStatus.rejected)
        tx = cs_charging_profiles.get("transaction_id")
        if str(unit) not in ("A", "ChargingRateUnitType.amps") or phases not in (1, 3) or limit < 0 or limit > self.ev_spec.max_a:
            self.rejected += 1
            return call_result.SetChargingProfile(status=ChargingProfileStatus.rejected)
        if tx is not None and tx != self.transaction_id:
            self.rejected += 1
            return call_result.SetChargingProfile(status=ChargingProfileStatus.rejected)
        self.limit_a, self.phases = limit, phases
        self.profiles_received.append((limit, phases))
        return call_result.SetChargingProfile(status=ChargingProfileStatus.accepted)

    # ---- physics (no I/O)
    def present_power_kw(self) -> float:
        """Power the vehicle draws right now given the active limit."""
        if self.vehicle is None or self.vehicle.done or self.transaction_id is None:
            return 0.0
        if self.limit_a < self.ev_spec.min_a:  # below the minimum pilot current the EVSE suspends
            return 0.0
        a = min(self.limit_a, self.vehicle.max_a)
        return self.ev_spec.power_kw(a, self.phases)

    def advance(self, minutes: float) -> float:
        """Integrate energy over `minutes`; returns kWh delivered."""
        p = self.present_power_kw()
        e = min(p * minutes / 60.0, (self.vehicle.need_kwh - self.vehicle.delivered_kwh) if self.vehicle else 0.0)
        if self.vehicle is not None and e > 0:
            self.vehicle.delivered_kwh += e
        self.energy_wh += e * 1000.0
        self.power_w = e * 60.0 / minutes * 1000.0 if (e > 0 and minutes > 0) else 0.0
        return e

    # ---- charge point -> central system
    async def boot(self) -> None:
        await self.call(call.BootNotification(charge_point_model="SimCP-1", charge_point_vendor="pvsc-sim"))
        await self.status(ChargePointStatus.available)

    async def status(self, st: ChargePointStatus) -> None:
        await self.call(call.StatusNotification(connector_id=1, error_code=ChargePointErrorCode.no_error, status=st))

    async def plug_in(self, vehicle: EvModel, id_tag: str = "demo-tag") -> None:
        self.vehicle = vehicle
        await self.status(ChargePointStatus.preparing)
        await self.call(call.Authorize(id_tag=id_tag))
        resp = await self.call(
            call.StartTransaction(connector_id=1, id_tag=id_tag, meter_start=int(self.energy_wh), timestamp=_now())
        )
        self.transaction_id = resp.transaction_id
        await self.status(ChargePointStatus.charging)

    async def unplug(self) -> None:
        tx = self.transaction_id
        self.transaction_id = None
        self.power_w = 0.0
        if tx is not None:
            await self.call(call.StopTransaction(meter_stop=int(self.energy_wh), timestamp=_now(), transaction_id=tx))
        self.vehicle = None
        await self.status(ChargePointStatus.available)

    async def send_meter_values(self) -> None:
        if self.transaction_id is None:
            return
        per_phase_a = (self.power_w / VOLTAGE / self.phases) if self.phases else 0.0
        sv = [
            SampledValue(value=f"{self.power_w:.0f}", measurand=Measurand.power_active_import, unit=UnitOfMeasure.w),
            SampledValue(value=f"{self.energy_wh:.0f}", measurand=Measurand.energy_active_import_register,
                         unit=UnitOfMeasure.wh),
        ]
        for i, ph in enumerate((Phase.l1, Phase.l2, Phase.l3)):
            a = per_phase_a if (self.phases == 3 or i == 0) else 0.0
            sv.append(SampledValue(value=f"{a:.2f}", measurand=Measurand.current_import, phase=ph, unit=UnitOfMeasure.a))
        await self.call(call.MeterValues(connector_id=1, transaction_id=self.transaction_id,
                                          meter_value=[MeterValue(timestamp=_now(), sampled_value=sv)]))


async def connect_charge_point(url: str, cp_id: str, ev_spec: EvSpec | None = None) -> tuple[SimChargePoint, object]:
    """Open the websocket (subprotocol ocpp1.6) and return (charge point, connection). Caller runs cp.start()."""
    conn = await connect(f"{url.rstrip('/')}/{cp_id}", subprotocols=[SUBPROTOCOL])
    return SimChargePoint(cp_id, conn, ev_spec), conn


__all__ = ["EvModel", "SimChargePoint", "connect_charge_point", "websockets"]
