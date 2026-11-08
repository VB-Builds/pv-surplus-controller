import numpy as np
import pytest

from pvsc import data as D
from pvsc.config import STEPS_PER_DAY, Site
from pvsc.metrics import MONTH_DAYS_2024, compute, month_slice
from pvsc.sim import Inputs, build_inputs, run


def synthetic(days=2, base_kw=0.3, pv_peak=6.0, need=15.0, arrive_h=8, depart_h=40, t_out=18.0):
    n = days * STEPS_PER_DAY
    hod = (np.arange(n) / 12.0) % 24
    pv = np.clip(pv_peak * np.sin(np.pi * (hod - 6) / 12), 0, None) * ((hod > 6) & (hod < 18))
    base = np.full((n, 3), base_kw / 3)
    sessions = [D.Session(arrive_h * 12, depart_h * 12, need)]
    return Inputs(pv=pv, pv_fc=pv.copy(), t_out=np.full(n, t_out), base_ph=base, dhw=np.full(n, 4.5 / STEPS_PER_DAY),
                  sessions=sessions)


@pytest.mark.parametrize("strategy", ["dumb", "rule", "optimised"])
def test_energy_balance_closes(strategy):
    inp = synthetic()
    r = run(strategy, inp, Site())
    load = r.base_kw + r.ev_kw + r.hp_kw + r.boiler_kw
    assert np.allclose(r.grid_kw, load + r.bat_kw - r.pv, atol=1e-9)


@pytest.mark.parametrize("strategy", ["dumb", "rule", "optimised"])
def test_ev_need_delivered(strategy):
    r = run(strategy, synthetic(), Site())
    s = r.sessions[0]
    assert s["delivered"] == pytest.approx(s["need"], abs=0.2)


def test_rule_uses_more_pv_than_dumb_on_sunny_day():
    site = Site()
    d = compute(run("dumb", synthetic(), site), site)
    r = compute(run("rule", synthetic(), site), site)
    assert r["self_consumption_pct"] > d["self_consumption_pct"] + 5
    assert r["cost_chf"] < d["cost_chf"]


def test_rule_never_exceeds_limit_but_dumb_does():
    site = Site()
    heavy = synthetic(base_kw=0.3)
    heavy.base_ph[:] = 3.2  # 13.9 A on every phase all day (uncontrollable)
    heavy.pv[:] = 0.0
    d = compute(run("dumb", heavy, site), site)
    r = compute(run("rule", heavy, site), site)
    assert d["steps_over_limit"] > 0 and d["max_phase_a"] > 25
    assert r["steps_over_limit"] == 0 and r["max_phase_a"] <= site.grid.limit_a - site.grid.margin_a + 1e-6


def test_rule_delivers_by_departure_even_without_sun():
    site = Site()
    inp = synthetic(pv_peak=0.0, need=20.0, depart_h=18)
    r = run("rule", inp, site)
    assert r.sessions[0]["delivered"] == pytest.approx(20.0, abs=0.2)


def test_optimised_plans_and_delivers():
    site = Site()
    r = run("optimised", synthetic(), site)
    assert r.plans > 0
    assert r.sessions[0]["delivered"] == pytest.approx(15.0, abs=0.2)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        run("magic", synthetic(), Site())


def test_dumb_charges_immediately_at_plug_in():
    r = run("dumb", synthetic(), Site())
    first = int(np.argmax(r.ev_kw > 0))
    assert first == 8 * 12
    assert r.ev_kw.max() == pytest.approx(11.04, abs=0.01)


def test_switch_counting_dumb_boiler_timer():
    r = run("dumb", synthetic(need=0.0), Site())
    assert r.sw_steps[:, 2].sum() >= 2  # night timer switches the boiler on and off


def test_metrics_cost_formula_and_bounds():
    site = Site()
    r = run("rule", synthetic(), site)
    m = compute(r, site)
    imp = np.clip(r.grid_kw, 0, None).sum() / 12
    exp = np.clip(-r.grid_kw, 0, None).sum() / 12
    assert m["cost_chf"] == pytest.approx(imp * site.tariff.buy - exp * site.tariff.feed_in, abs=0.01)
    assert 0 <= m["self_consumption_pct"] <= 100


def test_month_slices_cover_leap_year():
    assert sum(MONTH_DAYS_2024) == 366
    assert month_slice(12).stop == 366 * STEPS_PER_DAY
    assert month_slice(2).stop - month_slice(2).start == 29 * STEPS_PER_DAY


def test_build_inputs_shapes_and_short_run():
    inp = build_inputs("mixed", Site(), n_days=3)
    assert inp.n == 3 * STEPS_PER_DAY and inp.base_ph.shape == (inp.n, 3)
    r = run("rule", inp, Site())
    assert len(r.grid_kw) == inp.n
