import asyncio
import socket

import pytest
from pymodbus.client import AsyncModbusTcpClient

from pvsc import sunspec as S
from pvsc.ocpp_cp import EvModel, connect_charge_point
from pvsc.ocpp_cs import CentralSystem, build_tx_profile


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ sunspec encoding


def test_s16_roundtrip():
    for v in (-32768, -1, 0, 1, 32767):
        assert S.from_s16(S.to_s16(v)) == v


def test_s16_overflow_raises():
    with pytest.raises(ValueError):
        S.to_s16(40000)


def test_scale_factor_roundtrip():
    assert S.scaled(12.34, -2) == 1234
    assert S.unscaled(1234, -2) == pytest.approx(12.34)
    assert S.unscaled(S.scaled(-5.5, -1), -1) == pytest.approx(-5.5)


def test_u32_words():
    assert S.u32_words(0x12345678) == (0x1234, 0x5678)
    assert S.u32_from(S.u32_words(4_000_000_000)) == 4_000_000_000
    with pytest.raises(ValueError):
        S.u32_words(-1)


def test_register_map_headers():
    regs = S.PlantRegisters()
    assert regs.get_words(S.BASE, 2) == [0x5375, 0x6E53]  # "SunS"
    assert regs.get_words(S.INV, 2) == [103, 50]
    assert regs.get_words(S.MET, 2) == [203, 105]


def test_update_and_decode_snapshot():
    regs = S.PlantRegisters()
    regs.update(5000, (1000, -200, 300), dt_h=0.5, clock_min=725)
    snap = S.decode_snapshot(regs.get_words(S.INV, 42), regs.get_words(S.MET, 58), regs.get_words(S.SIM, 10))
    assert snap.pv_w == 5000 and snap.grid_w == 1100
    assert snap.grid_phase_w == (1000, -200, 300)
    assert snap.grid_phase_a[1] == pytest.approx(-200 / 230, abs=0.01)
    assert snap.clock_min == 725 and snap.pv_wh == 2500


def test_decode_rejects_wrong_model_ids():
    with pytest.raises(ValueError):
        S.decode_snapshot([1] * 42, [203] * 58, [0] * 10)


def test_energy_counters_accumulate():
    regs = S.PlantRegisters()
    regs.update(0, (2000, 0, 0), dt_h=1.0)
    regs.update(0, (-1000, 0, 0), dt_h=1.0)
    m = S.MET + 2
    assert S.u32_from(regs.get_words(m + 46, 2)) == 2000  # imported Wh
    assert S.u32_from(regs.get_words(m + 38, 2)) == 1000  # exported Wh


def test_modbus_tcp_roundtrip_and_side_channel_write():
    async def main():
        regs = S.PlantRegisters()
        port = free_port()
        srv = S.make_server(regs, port=port)
        task = asyncio.create_task(srv.serve_forever())
        await asyncio.sleep(0.4)
        regs.update(4200, (500, 500, 500), clock_min=60)
        c = AsyncModbusTcpClient("127.0.0.1", port=port)
        await c.connect()
        snap = await S.read_snapshot(c)
        wr = await c.write_registers(S.SIM_EV_W, [0, 3680])
        snap2 = await S.read_snapshot(c)
        bad = await c.read_holding_registers(100, count=2)  # outside the map
        c.close()
        await srv.shutdown()
        task.cancel()
        return snap, wr.isError(), snap2, bad.isError()

    snap, werr, snap2, bad = asyncio.run(main())
    assert snap.pv_w == 4200 and snap.grid_w == 1500 and snap.clock_min == 60
    assert not werr and snap2.ev_w == 3680
    assert bad


# ------------------------------------------------------------------ OCPP


def test_tx_profile_fields():
    p = build_tx_profile(3, 7, 10.0, 1)
    assert p.transaction_id == 7
    period = p.charging_schedule.charging_schedule_period[0]
    assert period.limit == 10.0 and period.number_phases == 1


def test_tx_profile_validation():
    with pytest.raises(ValueError):
        build_tx_profile(1, 1, 10, 2)
    with pytest.raises(ValueError):
        build_tx_profile(1, 1, -1, 3)


def test_ocpp_session_smart_charging_roundtrip():
    async def main():
        cs = CentralSystem(port=0)
        await cs.start()
        cp, conn = await connect_charge_point(f"ws://127.0.0.1:{cs.port}", "CPX")
        runner = asyncio.create_task(cp.start())
        await cp.boot()
        central_cp = await cs.wait_for("CPX")
        before = await central_cp.set_limit(10, 3)  # no transaction yet
        await cp.plug_in(EvModel(need_kwh=3.0))
        ok = await central_cp.set_limit(10, 1)
        p_1p = cp.present_power_kw()
        ok3 = await central_cp.set_limit(16, 3)
        p_3p = cp.present_power_kw()
        too_high = await central_cp.set_limit(40, 3)
        await central_cp.set_limit(0, 3)
        p_zero = cp.present_power_kw()
        await central_cp.set_limit(16, 3)
        cp.advance(60)  # one hour at 11 kW but the EV only needs 3 kWh
        await cp.send_meter_values()
        state = cs.states["CPX"]
        await cp.unplug()
        final = cs.states["CPX"].transaction_id
        await conn.close()
        runner.cancel()
        await cs.stop()
        return before, ok, p_1p, ok3, p_3p, too_high, p_zero, state, final, cp

    before, ok, p_1p, ok3, p_3p, too_high, p_zero, state, final, cp = asyncio.run(main())
    assert before is False
    assert ok and p_1p == pytest.approx(2.3, abs=0.01)
    assert ok3 and p_3p == pytest.approx(11.04, abs=0.01)
    assert too_high is False and cp.rejected == 1
    assert p_zero == 0.0
    assert state.vendor == "pvsc-sim" and state.energy_wh == pytest.approx(3000, abs=1)
    assert state.meter_values_seen == 1 and final is None
    assert (10.0, 1) in cp.profiles_received


def test_charge_point_below_min_current_suspends():
    async def main():
        cs = CentralSystem(port=0)
        await cs.start()
        cp, conn = await connect_charge_point(f"ws://127.0.0.1:{cs.port}", "CPY")
        runner = asyncio.create_task(cp.start())
        await cp.boot()
        central_cp = await cs.wait_for("CPY")
        await cp.plug_in(EvModel(need_kwh=10.0))
        await central_cp.set_limit(4, 1)
        p = cp.present_power_kw()
        await conn.close()
        runner.cancel()
        await cs.stop()
        return p

    assert asyncio.run(main()) == 0.0
