import pytest

from pvsc.config import ControllerConfig, EvSpec, GridConnection
from pvsc.control import Meas, SurplusController, dumb_setpoint, phase_headroom_a
from pvsc.devices import sg_contacts


def ctl(**kw):
    return SurplusController(ControllerConfig(**kw), EvSpec(), GridConnection())


def meas(pv=0.0, base=(0.3, 0.3, 0.3), **kw):
    d = dict(pv_kw=pv, base_ph_kw=base, ev_connected=True, ev_remaining_kwh=20.0, ev_hours_left=20.0, ev_session=1)
    d.update(kw)
    return Meas(**d)


def test_no_start_below_threshold():
    sp = ctl().decide(meas(pv=0.9 + 1.0))  # surplus 1.0 kW < 1.5
    assert sp.ev_amps == 0


def test_start_at_min_current_with_small_surplus():
    sp = ctl().decide(meas(pv=0.9 + 1.6))
    assert sp.ev_amps == 6 and sp.ev_phases == 1


def test_amps_follow_surplus_single_phase():
    sp = ctl().decide(meas(pv=0.9 + 2.5))
    assert sp.ev_phases == 1 and sp.ev_amps == int(2500 // 230)


def test_hysteresis_keeps_running_between_stop_and_start():
    c = ctl()
    c.decide(meas(pv=0.9 + 2.0))
    for _ in range(5):
        sp = c.decide(meas(pv=0.9 + 1.3))  # between stop (1.1) and start (1.5)
        assert sp.ev_amps > 0


def test_stops_after_delay_when_surplus_gone():
    c = ctl(stop_delay_steps=2, min_run_steps=1)
    c.decide(meas(pv=0.9 + 2.0))
    s1 = c.decide(meas(pv=0.9))
    s2 = c.decide(meas(pv=0.9))
    assert s1.ev_amps > 0  # still within the stop delay
    assert s2.ev_amps == 0


def test_phase_up_at_high_surplus():
    c = ctl()
    c.decide(meas(pv=0.9 + 2.0))  # 1p
    c.ev_phase_age = 99
    sp = c.decide(meas(pv=0.9 + 6.0))
    assert sp.ev_phases == 3 and sp.ev_amps >= 6


def test_phase_dwell_blocks_immediate_flip():
    c = ctl(phase_dwell_steps=3)
    c.decide(meas(pv=0.9 + 6.0))  # start in 3p
    assert c.ev_phases == 3
    c.ev_phase_age = 0
    sp = c.decide(meas(pv=0.9 + 2.0))
    assert sp.ev_phases == 3  # dwell not elapsed: stay


def test_phase_down_after_dwell():
    c = ctl(phase_dwell_steps=2)
    c.decide(meas(pv=0.9 + 6.0))
    sp = None
    for _ in range(4):
        sp = c.decide(meas(pv=0.9 + 2.0))
    assert sp.ev_phases == 1


def test_deadline_forces_max_charge_without_sun():
    sp = ctl().decide(meas(pv=0.0, ev_remaining_kwh=20.0, ev_hours_left=2.0))
    assert sp.forced and sp.ev_phases == 3 and sp.ev_amps == 16


def test_forced_latches_for_the_session():
    c = ctl()
    c.decide(meas(pv=0.0, ev_remaining_kwh=20.0, ev_hours_left=2.0))
    sp = c.decide(meas(pv=0.0, ev_remaining_kwh=15.0, ev_hours_left=3.0))
    assert sp.forced


def test_new_session_resets_latch():
    c = ctl()
    c.decide(meas(pv=0.0, ev_remaining_kwh=20.0, ev_hours_left=2.0, ev_session=1))
    sp = c.decide(meas(pv=0.0, ev_remaining_kwh=5.0, ev_hours_left=20.0, ev_session=2))
    assert not sp.forced and sp.ev_amps == 0


def test_limit_reduces_current_below_connection_limit():
    sp = ctl().decide(meas(pv=0.0, base=(3.22, 3.22, 3.22), ev_remaining_kwh=20.0, ev_hours_left=1.0))
    head = 25 - 1 - 3.22 * 1000 / 230
    assert sp.limited and 6 <= sp.ev_amps <= head + 1e-9


def test_limit_never_exceeded_over_sweep():
    g = GridConnection()
    for base_kw in [0.2, 1.0, 2.0, 3.0, 3.2]:
        for pv in [0.0, 2.0, 6.0]:
            c = SurplusController(ControllerConfig(), EvSpec(), g)
            sp = c.decide(meas(pv=pv, base=(base_kw,) * 3, ev_remaining_kwh=30.0, ev_hours_left=1.0))
            net_a = max(0.0, base_kw - pv / 3) * 1000 / 230
            assert net_a + sp.ev_amps <= g.limit_a - g.margin_a + 1e-6 or sp.ev_amps == 0


def test_falls_back_to_single_phase_when_l2_l3_busy():
    sp = ctl().decide(meas(pv=0.0, base=(0.2, 4.4, 4.4), ev_remaining_kwh=20.0, ev_hours_left=1.0))
    assert sp.ev_phases == 1 and sp.ev_amps >= 6


def test_ev_off_when_no_headroom():
    sp = ctl().decide(meas(pv=0.0, base=(4.6, 4.6, 4.6), ev_remaining_kwh=20.0, ev_hours_left=1.0))
    assert sp.ev_amps == 0


def test_headroom_helper():
    g = GridConnection()
    assert phase_headroom_a(g, 0.0) == pytest.approx(24.0)
    assert phase_headroom_a(g, -5.0) == pytest.approx(24.0)  # export does not add headroom
    assert phase_headroom_a(g, 2.3) == pytest.approx(14.0)


def test_boiler_comfort_floor_forces_on():
    sp = ctl().decide(meas(pv=0.0, tank_c=45.0, ev_connected=False))
    assert sp.boiler_on


def test_boiler_on_with_surplus_and_holds_min_run():
    c = ctl(boiler_min_run_steps=3)
    assert c.decide(meas(pv=0.9 + 3.5, ev_connected=False, tank_c=55.0)).boiler_on
    assert c.decide(meas(pv=0.9, ev_connected=False, tank_c=55.0)).boiler_on  # min run not elapsed


def test_boiler_off_at_max_temperature():
    assert not ctl().decide(meas(pv=10.0, ev_connected=False, tank_c=63.0)).boiler_on


def test_boiler_shed_when_l1_over_limit():
    sp = ctl().decide(meas(pv=0.0, base=(3.2, 0.3, 0.3), tank_c=45.0, ev_connected=False))
    # 3.2 kW + 3 kW boiler on L1 = 27 A > 24 A: the boiler is shed
    assert not sp.boiler_on and sp.limited


def test_sg_boost_in_heating_season_with_surplus():
    sp = ctl().decide(meas(pv=0.9 + 1.5, ev_connected=False, heating_season=True, hp_room_to_boost=True))
    assert sp.sg_state == 3


def test_sg_normal_outside_heating_season():
    sp = ctl().decide(meas(pv=0.9 + 3.0, ev_connected=False, heating_season=False, hp_room_to_boost=True))
    assert sp.sg_state == 2


def test_no_ev_no_current():
    assert ctl().decide(meas(pv=8.0, ev_connected=False)).ev_amps == 0


def test_dumb_setpoint():
    m = meas()
    assert dumb_setpoint(m, 12.0, 55.0, EvSpec()).ev_amps == 16
    assert dumb_setpoint(m, 23.0, 40.0, EvSpec()).boiler_on
    assert not dumb_setpoint(m, 12.0, 40.0, EvSpec()).boiler_on


def test_sg_contacts_mapping():
    assert sg_contacts(2) == (0, 0) and sg_contacts(3) == (0, 1)
    with pytest.raises(ValueError):
        sg_contacts(7)
