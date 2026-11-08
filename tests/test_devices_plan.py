import numpy as np
import pytest

from pvsc.config import BatterySpec, BoilerSpec, HeatPumpSpec
from pvsc.devices import Battery, Boiler, HeatPump, cop
from pvsc.plan import Plan, plan_ev, slot_lengths


def test_cop_clipped():
    assert cop(-40) == 2.2 and cop(60) == 5.0
    assert cop(0) == pytest.approx(3.4)


def test_heat_pump_thermostat_hysteresis():
    hp = HeatPump(HeatPumpSpec(), t_in=19.0)
    assert hp.thermostat(0.0, 2)
    hp.t_in = 20.5
    assert hp.thermostat(0.0, 2)  # inside the band: keeps state
    hp.t_in = 21.0
    assert not hp.thermostat(0.0, 2)


def test_heat_pump_boost_raises_setpoint():
    hp = HeatPump(HeatPumpSpec(), t_in=20.9)
    assert not hp.thermostat(0.0, 2)  # normal setpoint 20.5 + 0.3 band: stays off
    assert hp.thermostat(0.0, 3)  # boosted setpoint 21.5: on below 21.2


def test_heat_pump_off_in_summer():
    hp = HeatPump(HeatPumpSpec(), t_in=15.0)
    assert not hp.thermostat(20.0, 3)


def test_building_cools_without_heating():
    hp = HeatPump(HeatPumpSpec(), t_in=20.0)
    for _ in range(12 * 12):
        hp.step(0.0)
    assert hp.t_in < 20.0


def test_building_heats_when_on():
    hp = HeatPump(HeatPumpSpec(), t_in=18.0)
    hp.on = True
    hp.step(5.0)
    assert hp.t_in > 18.0


def test_battery_respects_power_and_soc_limits():
    b = Battery(BatterySpec(capacity_kwh=10, power_kw=5, efficiency=1.0, min_soc=0.1), soc=0.5)
    assert b.charge(20.0) == 5.0
    b.soc = 0.9999
    assert b.charge(5.0) < 0.02
    b.soc = 0.1
    assert b.discharge(5.0) == 0.0


def test_battery_energy_conservation_with_losses():
    spec = BatterySpec(efficiency=0.9)
    b = Battery(spec, soc=0.2)
    e0 = b.soc * spec.capacity_kwh
    p = b.charge(2.0)
    stored = b.soc * spec.capacity_kwh - e0
    assert stored == pytest.approx(p / 12 * 0.9)


def test_boiler_heats_and_cools():
    b = Boiler(BoilerSpec(), t_c=50.0)
    for _ in range(12):
        b.step(True, 0.0)
    assert b.t_c > 50.0
    t = b.t_c
    for _ in range(12):
        b.step(False, 0.2)
    assert b.t_c < t


def test_slot_lengths_sum_and_partial_first():
    d = slot_lengths(now_step=6, end_step=6 + 30)  # 2.5 h starting at 00:30
    assert d.sum() == pytest.approx(2.5)
    assert d[0] == pytest.approx(0.5)


def test_plan_meets_need_when_feasible():
    dt = np.ones(10)
    p = plan_ev(20.0, np.zeros(10), np.full(10, 0.5), dt, 0.24, 0.08, 11.0, 17.0)
    assert p.shortfall_kwh == pytest.approx(0.0, abs=1e-6)
    assert (p.kw * dt).sum() == pytest.approx(20.0, abs=1e-6)


def test_plan_reports_shortfall_when_infeasible():
    p = plan_ev(50.0, np.zeros(2), np.zeros(2), np.ones(2), 0.24, 0.08, 11.0, 17.0)
    assert p.shortfall_kwh == pytest.approx(50 - 22, abs=1e-5)


def test_plan_prefers_sun_over_grid():
    pv = np.array([0, 0, 6.0, 6.0, 0, 0])
    p = plan_ev(10.0, pv, np.full(6, 0.5), np.ones(6), 0.24, 0.08, 11.0, 17.0)
    assert p.kw[2] + p.kw[3] == pytest.approx(10.0, abs=1e-5)
    assert p.kw[0] == pytest.approx(0.0, abs=1e-6)


def test_plan_prefers_cheap_night_tariff():
    price = np.array([0.27, 0.27, 0.19, 0.19])
    p = plan_ev(8.0, np.zeros(4), np.zeros(4), np.ones(4), price, 0.08, 11.0, 17.0)
    assert p.kw[:2].sum() == pytest.approx(0.0, abs=1e-6)
    assert p.cost_chf == pytest.approx(8.0 * 0.19, rel=1e-4)


def test_plan_respects_import_limit():
    p = plan_ev(30.0, np.zeros(3), np.full(3, 8.0), np.ones(3), 0.24, 0.08, 11.0, 12.0)
    assert (p.kw <= 4.0 + 1e-6).all()  # 12 kW import limit minus 8 kW base load


def test_plan_zero_need():
    p = plan_ev(0.0, np.zeros(3), np.zeros(3), np.ones(3), 0.24, 0.08, 11.0, 17.0)
    assert p.kw.sum() == 0 and p.shortfall_kwh == 0


def test_plan_mismatched_arrays():
    with pytest.raises(ValueError):
        plan_ev(5.0, np.zeros(3), np.zeros(2), np.ones(3), 0.24, 0.08, 11.0, 17.0)


def test_plan_kw_at_maps_steps():
    p = Plan(kw=np.array([1.0, 2.0]), dt_h=np.array([0.5, 1.0]), shortfall_kwh=0, cost_chf=0, start_step=100)
    assert p.kw_at(100) == 1.0 and p.kw_at(105) == 1.0 and p.kw_at(106) == 2.0 and p.kw_at(500) == 0.0
