import numpy as np
import pytest

from pvsc import data as D
from pvsc.config import EvSpec, PvSpec, Tariff


def test_pv_zero_at_night():
    assert D.pv_power_kw(np.zeros(5), np.full(5, 10.0), PvSpec()).sum() == 0.0


def test_pv_known_value():
    spec = PvSpec(kwp=8.0, system_factor=0.8, inverter_kw=10.0)
    # cell temperature = air + 0.03 * G; at G=1000 and air=-5 C the cell is exactly 25 C, so no derating
    p = D.pv_power_kw(np.array([1000.0]), np.array([-5.0]), spec)[0]
    assert p == pytest.approx(8.0 * 0.8, rel=1e-9)


def test_pv_clipped_by_inverter():
    p = D.pv_power_kw(np.array([2000.0]), np.array([0.0]), PvSpec(kwp=20.0, inverter_kw=5.0))[0]
    assert p == 5.0


def test_pv_heat_derates():
    cold = D.pv_power_kw(np.array([800.0]), np.array([0.0]), PvSpec())[0]
    hot = D.pv_power_kw(np.array([800.0]), np.array([35.0]), PvSpec())[0]
    assert hot < cold


def test_pv_nan_irradiance_is_zero():
    assert D.pv_power_kw(np.array([np.nan, 500.0]), np.array([10.0, 10.0]), PvSpec())[0] == 0.0


def test_hourly_to_steps_length_and_energy():
    h = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    s = D.hourly_to_steps(h)
    assert len(s) == len(h) * 12
    assert s.min() >= 0
    assert s.sum() / 12 == pytest.approx(h.sum(), rel=0.15)


def test_hourly_to_steps_centres_on_half_hour():
    h = np.zeros(6)
    h[3] = 3.0  # mean over 02:00-03:00
    s = D.hourly_to_steps(h)
    assert int(np.argmax(s)) == int(2.5 * 12)


def test_load_irradiance_rejects_wrong_length(tmp_path):
    (tmp_path / "zurich_2024_observed.csv").write_text("time,gti_wm2,temp_c\n2024-01-01T00:00,0,1\n")
    with pytest.raises(ValueError, match="expected"):
        D.load_irradiance("observed", tmp_path)


def test_real_data_files_present_and_sane():
    df = D.load_irradiance("observed")
    assert df.gti_wm2.min() >= 0 and df.gti_wm2.max() < 1400
    assert 800 < df.gti_wm2.sum() / 1000 < 1800  # kWh/m2/a on a tilted plane in Switzerland


def test_base_load_deterministic_and_capped():
    a = D.gen_base_load(3, n_days=5)
    b = D.gen_base_load(3, n_days=5)
    assert np.array_equal(a, b)
    assert a.shape == (5 * 288, 3)
    assert a.max() <= D.PHASE_CAP_KW + 1e-9 and a.min() >= 0


def test_base_load_seed_changes_output():
    assert not np.array_equal(D.gen_base_load(1, n_days=3), D.gen_base_load(2, n_days=3))


def test_base_load_annual_energy_plausible():
    e = D.gen_base_load(7).sum() / 12
    assert 3000 < e < 4500


def test_dhw_energy_per_day():
    d = D.gen_dhw_draw(n_days=30, kwh_per_day=4.5)
    per_day = d.reshape(30, 288).sum(axis=1)
    assert per_day.mean() == pytest.approx(4.5, rel=0.1)


def test_sessions_ordered_and_positive():
    for sc in D.SCENARIOS:
        ss = D.gen_ev_sessions(sc, EvSpec())
        assert ss, sc
        assert all(s.end > s.start and s.need_kwh > 0 for s in ss)
        assert all(a.end <= b.start + 1 for a, b in zip(ss, ss[1:], strict=False))


def test_commuter_has_more_sessions_than_home():
    c = len(D.gen_ev_sessions("commuter", EvSpec()))
    h = len(D.gen_ev_sessions("home", EvSpec()))
    assert c > h


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        D.gen_ev_sessions("nope", EvSpec())


def test_tariff_flat_and_htnt():
    assert Tariff().buy_at(10) == Tariff().buy
    t = Tariff(buy=0.27, nt_buy=0.19)
    assert t.buy_at(12 * 12) == 0.27  # Monday 12:00
    assert t.buy_at(3 * 12) == 0.19  # Monday 03:00
    assert t.buy_at(5 * 288 + 12 * 12) == 0.19  # Saturday noon


def test_buy_series_matches_tariff():
    s = D.buy_series(Tariff(buy=0.3, nt_buy=0.2), 288)
    assert s[0] == 0.2 and s[12 * 12] == 0.3
