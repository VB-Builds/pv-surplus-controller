"""OCPP 1.6J central system (the energy manager's side): accepts charge points over websocket, tracks their state and
sets charging limits with SetChargingProfile (smart charging, TxProfile, amps per phase, number of phases)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import websockets
from ocpp.routing import on
from ocpp.v16 import ChargePoint as _ChargePoint
from ocpp.v16 import call, call_result
from ocpp.v16.datatypes import ChargingProfile, ChargingSchedule, ChargingSchedulePeriod, IdTagInfo
from ocpp.v16.enums import (
    Action,
    AuthorizationStatus,
    ChargingProfileKindType,
    ChargingProfilePurposeType,
    ChargingProfileStatus,
    ChargingRateUnitType,
    RegistrationStatus,
)
from websockets.asyncio.server import serve

log = logging.getLogger("pvsc.ocpp_cs")
SUBPROTOCOL = "ocpp1.6"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_tx_profile(profile_id: int, transaction_id: int, amps: float, phases: int) -> ChargingProfile:
    """Absolute TxProfile with a single open-ended period: `amps` per phase on `phases` phases."""
    if phases not in (1, 3):
        raise ValueError("phases must be 1 or 3")
    if amps < 0:
        raise ValueError("amps must be >= 0")
    period = ChargingSchedulePeriod(start_period=0, limit=float(amps), number_phases=phases)
    sched = ChargingSchedule(charging_rate_unit=ChargingRateUnitType.amps, charging_schedule_period=[period])
    return ChargingProfile(
        charging_profile_id=profile_id,
        stack_level=1,
        charging_profile_purpose=ChargingProfilePurposeType.tx_profile,
        charging_profile_kind=ChargingProfileKindType.absolute,
        charging_schedule=sched,
        transaction_id=transaction_id,
    )


@dataclass
class CpState:
    """What the central system knows about one charge point."""

    cp_id: str
    vendor: str = ""
    model: str = ""
    status: str = "Unknown"
    transaction_id: int | None = None
    power_w: float = 0.0
    energy_wh: float = 0.0
    current_a: dict[str, float] = field(default_factory=dict)
    last_limit: tuple[float, int] | None = None
    profiles_sent: int = 0
    meter_values_seen: int = 0
    tx_counter: int = 0


class CentralChargePoint(_ChargePoint):
    def __init__(self, cp_id, connection, state: CpState):
        super().__init__(cp_id, connection)
        self.state = state
        self._profile_id = 0

    @on(Action.boot_notification)
    async def on_boot(self, charge_point_vendor, charge_point_model, **kwargs):
        self.state.vendor, self.state.model = charge_point_vendor, charge_point_model
        return call_result.BootNotification(current_time=now_iso(), interval=30, status=RegistrationStatus.accepted)

    @on(Action.heartbeat)
    async def on_heartbeat(self, **kwargs):
        return call_result.Heartbeat(current_time=now_iso())

    @on(Action.authorize)
    async def on_authorize(self, id_tag, **kwargs):
        return call_result.Authorize(id_tag_info=IdTagInfo(status=AuthorizationStatus.accepted))

    @on(Action.status_notification)
    async def on_status(self, connector_id, error_code, status, **kwargs):
        if connector_id >= 1:
            self.state.status = str(getattr(status, "value", status))
        return call_result.StatusNotification()

    @on(Action.start_transaction)
    async def on_start(self, connector_id, id_tag, meter_start, timestamp, **kwargs):
        self.state.tx_counter += 1
        self.state.transaction_id = self.state.tx_counter
        self.state.energy_wh = float(meter_start)
        return call_result.StartTransaction(
            transaction_id=self.state.transaction_id, id_tag_info=IdTagInfo(status=AuthorizationStatus.accepted)
        )

    @on(Action.stop_transaction)
    async def on_stop(self, meter_stop, timestamp, transaction_id, **kwargs):
        self.state.energy_wh = float(meter_stop)
        self.state.transaction_id = None
        self.state.power_w = 0.0
        self.state.last_limit = None
        return call_result.StopTransaction(id_tag_info=IdTagInfo(status=AuthorizationStatus.accepted))

    @on(Action.meter_values)
    async def on_meter_values(self, connector_id, meter_value, **kwargs):
        for mv in meter_value:
            for sv in mv["sampled_value"]:
                meas = sv.get("measurand", "Energy.Active.Import.Register")
                val = float(sv["value"])
                if meas == "Power.Active.Import":
                    self.state.power_w = val * (1000.0 if sv.get("unit") == "kW" else 1.0)
                elif meas == "Energy.Active.Import.Register":
                    self.state.energy_wh = val * (1000.0 if sv.get("unit") == "kWh" else 1.0)
                elif meas == "Current.Import":
                    self.state.current_a[sv.get("phase", "total")] = val
        self.state.meter_values_seen += 1
        return call_result.MeterValues()

    async def set_limit(self, amps: float, phases: int) -> bool:
        """Send a SetChargingProfile for the running transaction. Returns True if the charge point accepted."""
        tx = self.state.transaction_id
        if tx is None:
            return False
        self._profile_id += 1
        profile = build_tx_profile(self._profile_id, tx, amps, phases)
        resp = await self.call(call.SetChargingProfile(connector_id=1, cs_charging_profiles=profile))
        ok = resp.status == ChargingProfileStatus.accepted
        if ok:
            self.state.last_limit = (float(amps), phases)
            self.state.profiles_sent += 1
        return ok


class CentralSystem:
    """Websocket server hosting any number of charge points."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host, self.port = host, port
        self.points: dict[str, CentralChargePoint] = {}
        self.states: dict[str, CpState] = {}
        self._server = None

    async def _handler(self, conn):
        cp_id = conn.request.path.strip("/").split("/")[-1] or "unknown"
        if conn.subprotocol != SUBPROTOCOL:
            await conn.close(code=1002, reason="ocpp1.6 subprotocol required")
            return
        state = self.states.setdefault(cp_id, CpState(cp_id))
        cp = CentralChargePoint(cp_id, conn, state)
        self.points[cp_id] = cp
        try:
            await cp.start()
        except websockets.exceptions.ConnectionClosed:
            log.info("charge point %s disconnected", cp_id)
        finally:
            self.points.pop(cp_id, None)

    async def start(self) -> None:
        self._server = await serve(
            self._handler, self.host, self.port, subprotocols=[SUBPROTOCOL]
        )
        if self.port == 0:
            self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def wait_for(self, cp_id: str, timeout: float = 10.0) -> CentralChargePoint:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if cp_id in self.points:
                return self.points[cp_id]
            await asyncio.sleep(0.05)
        raise TimeoutError(f"charge point {cp_id} did not connect")
