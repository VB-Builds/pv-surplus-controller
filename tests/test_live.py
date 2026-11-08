import asyncio
import socket

from pvsc.live import Scenario, limit_scenario, run_live


def _ports():
    ports = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
    return ports


def test_live_loop_sunny_day_delivers_energy_over_real_protocols():
    mp, op = _ports()
    sc = Scenario(day=210, need_kwh=10.0)
    res = asyncio.run(run_live(tick_s=0.002, scenario=sc, modbus_port=mp, ocpp_port=op))
    assert res["steps"] == 288
    assert res["steps_over_limit"] == 0
    assert res["ocpp_profiles_sent"] > 0
    assert res["profiles_accepted_by_cp"] == res["ocpp_profiles_sent"]
    assert abs(res["ev_kwh"] - 10.0) < 0.3


def test_live_loop_limit_scenario_stays_under_limit():
    mp, op = _ports()
    res = asyncio.run(run_live(tick_s=0.002, scenario=limit_scenario(), modbus_port=mp, ocpp_port=op))
    assert res["steps_over_limit"] == 0
    assert res["settled_peak_a"] <= 24.0 + 0.5
    assert res["transient_peak_a_before_reaction"] >= res["settled_peak_a"]
    assert res["steps_limited_by_controller"] > 0
