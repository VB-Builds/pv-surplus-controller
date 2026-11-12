"""Run all simulations, write results/ (json, csv, markdown) and the figures + LinkedIn text in docs/."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DT_H, STEPS_PER_DAY, Site, Tariff  # noqa: E402
from .data import SCENARIOS  # noqa: E402
from .metrics import SEASONS, compute, month_slice, steps_of  # noqa: E402
from .sim import STRATEGIES, Result, build_inputs, run  # noqa: E402

# tariff variants: flat = ElCom 2024 H4 (operator 565) + assumed feed-in; htnt = ILLUSTRATIVE two-rate example (assumed prices)
TARIFFS = {
    "flat": Tariff(),
    "htnt": Tariff(buy=0.27, nt_buy=0.19),
}
PAL = {"bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
       "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA"}
SOURCE = "Irradiance: Open-Meteo.com (CC BY 4.0), Zurich 2024. House, EV, tariffs: simulated / assumed."


def _job(args: tuple) -> tuple[tuple[str, str, str], Result]:
    scenario, tariff_name, strategy, n_days = args
    site = Site(tariff=TARIFFS[tariff_name])
    inp = build_inputs(scenario, site, n_days=n_days)
    return (scenario, tariff_name, strategy), run(strategy, inp, site)


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "axes.facecolor": PAL["bg"], "figure.facecolor": PAL["bg"],
        "savefig.facecolor": PAL["bg"], "text.color": PAL["text"], "axes.labelcolor": PAL["muted"],
        "xtick.color": PAL["muted"], "ytick.color": PAL["muted"], "axes.edgecolor": PAL["grid"],
        "grid.color": PAL["grid"], "axes.grid": True, "grid.linewidth": 0.8,
    })


def pick_days(results: dict, site: Site) -> tuple[int, int]:
    """Showcase window = day d-1 (car arrives in the evening) + day d (sunny, car at home): dumb charging fills the
    car at night, the controller waits for the sun. worst = day with the highest phase current under dumb charging."""
    dumb, rule = results["dumb"], results["rule"]
    n_days = dumb.n // STEPS_PER_DAY
    score = np.full(n_days, -1.0)
    for d in range(151, min(n_days, 245)):
        s0 = d * STEPS_PER_DAY
        day = slice(s0, s0 + STEPS_PER_DAY)
        night = slice(s0 - 6 * 12, s0 + 7 * 12)  # 18:00 previous evening .. 07:00
        pv_day = rule.pv[day].sum() * DT_H
        dumb_night = dumb.ev_kw[night].sum() * DT_H
        dumb_day = dumb.ev_kw[day].sum() * DT_H
        if pv_day > 40 and dumb_night > 8 and dumb_day < 1.0:
            score[d] = np.minimum(rule.ev_kw[day], rule.pv[day]).sum() * DT_H
    sun_day = int(np.argmax(score))
    peaks = dumb.phase_a.reshape(-1, STEPS_PER_DAY, 3).max(axis=(1, 2))
    return sun_day, int(np.argmax(peaks))


def _window(d: int) -> slice:
    """36 hours: from noon the day before to midnight after the sunny day."""
    return slice((d - 1) * STEPS_PER_DAY + 12 * 12, (d + 1) * STEPS_PER_DAY)


def _two_day_axis(ax, d: int, fs: int = 10) -> None:
    ax.set_xlim(0, 36)
    ax.set_xticks(range(0, 37, 6))
    ax.set_xticklabels([f"{(h + 12) % 24:02d}" for h in range(0, 37, 6)])
    ax.axvline(12, color=PAL["grid"], lw=1.5)
    d0 = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d - 1)).strftime("%d %b")
    d1 = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).strftime("%d %b %Y")
    ax.set_xlabel(f"hour of day, {d0} noon to {d1} midnight", fontsize=fs)


def _stack(ax, r: Result, d: int, title: str, ymax: float) -> None:
    s = _window(d)
    t = np.arange(36 * 12) / 12.0
    other = r.base_kw[s] + r.hp_kw[s] + r.boiler_kw[s]
    ax.fill_between(t, 0, r.pv[s], color=PAL["amber"], alpha=0.35, lw=0, label="PV production")
    ax.stackplot(t, other, r.ev_kw[s], colors=[PAL["grey"], PAL["accent"]], alpha=0.9,
                 labels=["house, heat pump, boiler", "EV charging"])
    ax.plot(t, r.pv[s], color=PAL["amber"], lw=1.6)
    ax.set_ylim(0, ymax)
    _two_day_axis(ax, d)
    ax.set_title(title, loc="left", fontsize=13, color=PAL["text"], fontweight="bold")
    ax.set_ylabel("kW")


def make_figure(res: dict, site: Site, monthly: pd.DataFrame, days: tuple[int, int], path: Path) -> None:
    _style()
    sun, worst = days
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.855, bottom=0.075, hspace=0.32, wspace=0.17)
    ymax = 14.0
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    _stack(ax1, res["dumb"], sun, "Dumb charging: EV fills up at plug-in, at night", ymax)
    _stack(ax2, res["rule"], sun, "Surplus controller: EV waits for the sun", ymax)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.3, 1.0), fontsize=11, frameon=False, labelcolor=PAL["text"], ncol=1)
    ax3 = fig.add_subplot(gs[0, 1])
    wdate = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=worst)).strftime("%d %b %Y")
    sl = slice(worst * STEPS_PER_DAY, (worst + 1) * STEPS_PER_DAY)
    t = np.arange(STEPS_PER_DAY) / 12.0
    ax3.plot(t, res["dumb"].phase_a[sl].max(axis=1), color=PAL["red"], lw=1.8, label="dumb charging")
    ax3.plot(t, res["rule"].phase_a[sl].max(axis=1), color=PAL["green"], lw=1.8, label="surplus controller")
    ax3.axhline(site.grid.limit_a, color=PAL["text"], ls="--", lw=1.2)
    ax3.text(0.2, site.grid.limit_a + 0.6, f"{site.grid.limit_a:.0f} A connection limit per phase", color=PAL["text"], fontsize=10)
    ax3.set_title(f"Highest phase current, worst dumb day {wdate}", loc="left", fontsize=13, fontweight="bold")
    ax3.set_ylabel("A"), ax3.set_xlim(0, 24), ax3.set_xticks(range(0, 25, 4))
    ax3.legend(loc="center", bbox_to_anchor=(0.5, 0.42), fontsize=11, frameon=False, labelcolor=PAL["text"])
    ax4 = fig.add_subplot(gs[1, 1])
    months = np.arange(1, 13)
    w = 0.27
    for k, (strat, col) in enumerate((("dumb", PAL["red"]), ("rule", PAL["green"]), ("optimised", PAL["accent"]))):
        ax4.bar(months + (k - 1) * w, monthly[strat], width=w, color=col, label=strat)
    ax4.set_xticks(months)
    ax4.set_xticklabels(list("JFMAMJJASOND"))
    ax4.set_ylabel("%")
    ax4.set_title("PV self-consumption per month (mixed EV scenario)", loc="left", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper center", fontsize=11, frameon=False, labelcolor=PAL["text"], ncol=3)
    ax4.set_ylim(0, 118)
    ax4.set_yticks(range(0, 101, 20))
    fig.text(0.055, 0.945, "pv-surplus-controller: rooftop-PV surplus into EV, boiler, heat pump and battery",
             fontsize=19, fontweight="bold", color=PAL["text"])
    fig.text(0.055, 0.905, "8 kWp, 10 kWh battery, 25 A per phase house connection. " + SOURCE, fontsize=11, color=PAL["muted"])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def make_post_figure(res: dict, site: Site, monthly: pd.DataFrame, sun: int, path: Path) -> None:
    _style()
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=PAL["bg"])
    gs = fig.add_gridspec(2, 1, left=0.095, right=0.97, top=0.77, bottom=0.095, hspace=0.62, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0])
    s = _window(sun)
    t = np.arange(36 * 12) / 12.0
    ax.fill_between(t, 0, res["rule"].pv[s], color=PAL["amber"], alpha=0.3, lw=0)
    ax.plot(t, res["rule"].pv[s], color=PAL["amber"], lw=2.2, label="PV production")
    ax.plot(t, res["dumb"].ev_kw[s], color=PAL["red"], lw=3, label="EV, dumb")
    ax.plot(t, res["rule"].ev_kw[s], color=PAL["green"], lw=3, label="EV, controller")
    _two_day_axis(ax, sun, fs=12)
    ax.set_ylabel("kW", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 13.5)
    ax.legend(loc="upper right", fontsize=12, frameon=False, labelcolor=PAL["text"], ncol=1)
    ax.set_title("A summer evening and the next day: when the car charges", loc="left", fontsize=14, color=PAL["text"])
    ax2 = fig.add_subplot(gs[1])
    months = np.arange(1, 13)
    ax2.bar(months - 0.2, monthly["dumb"], width=0.4, color=PAL["grey"], label="dumb charging")
    ax2.bar(months + 0.2, monthly["rule"], width=0.4, color=PAL["green"], label="surplus controller")
    ax2.set_xticks(months)
    ax2.set_xticklabels(list("JFMAMJJASOND"))
    ax2.set_ylim(0, 125)
    ax2.set_yticks(range(0, 101, 25))
    ax2.set_ylabel("%", fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.legend(loc="upper center", fontsize=12, frameon=False, labelcolor=PAL["text"], ncol=2)
    ax2.set_title("PV self-consumption per month", loc="left", fontsize=14, color=PAL["text"])
    fig.text(0.04, 0.925, "EV charging follows the sun", fontsize=19, fontweight="bold", color=PAL["text"])
    fig.text(0.04, 0.885, "Open-Meteo irradiance, Zurich 2024. Simulated house, EV, devices.", fontsize=12, color=PAL["muted"])
    fig.savefig(path, dpi=100, facecolor=PAL["bg"])
    plt.close(fig)


def linkedin(all_m: dict, live: dict | None) -> dict:
    m = all_m["mixed|flat"]
    d, r = m["dumb"]["year"], m["rule"]["year"]
    saved = d["cost_chf"] - r["cost_chf"]
    post = (
        "Rooftop solar often goes to the grid at a low price while the EV charges from the grid at night.\n\n"
        "I built an open-source home energy manager that routes PV surplus into an EV charger (OCPP 1.6J smart charging), "
        "a boiler and a heat pump (SG-Ready style signal), with a 25 A per phase house connection limit. "
        "It reads a simulated inverter and meter over Modbus TCP and sets charging limits through real OCPP messages to a "
        "simulated charge point.\n\n"
        "Simulated over one year (Zurich 2024 irradiance from Open-Meteo, 8 kWp, 10 kWh battery, a household with an EV "
        "and a heat pump): "
        f"PV self-consumption {d['self_consumption_pct']:.0f} % with dumb charging, "
        f"{r['self_consumption_pct']:.0f} % with the controller. "
        f"Cost saved against dumb charging: CHF {saved:.0f} per year at the ElCom 2024 Zurich tariff and an assumed 8 Rp feed-in. "
        f"The dumb baseline crossed 25 A on {d['steps_over_limit']} five-minute steps, the controller on {r['steps_over_limit']}. "
        f"Every EV session still reached its target by departure ({r['ev_delivered_by_departure_pct']:.0f} %).\n\n"
        "Not verified: no real hardware, a simulated house and driving pattern, an assumed feed-in price, and the "
        "limit holds only if the loop reacts faster than the load steps.\n\n"
        "What would you check first before trusting a controller like this on a real house?\n\n"
        "Code: https://github.com/VB-Builds/pv-surplus-controller"
    )
    return {
        "claim": "Open-source energy manager: sends rooftop-PV surplus to the EV, boiler and heat pump.",
        "data_label": "open data: Open-Meteo irradiance, Zurich 2024 | simulated house and devices",
        "stats": [
            [f"{r['self_consumption_pct']:.0f} %", f"PV self-consumption (dumb: {d['self_consumption_pct']:.0f} %)", "green"],
            [f"CHF {saved:.0f}", "saved per year vs dumb charging", "accent"],
            [f"{r['steps_over_limit']}", f"steps above 25 A (dumb: {d['steps_over_limit']})", "amber"],
        ],
        "limit": "Simulated house, EV and devices; feed-in price assumed; no real hardware tested yet.",
        "alt": ("Dark chart. Top: a summer evening and the next day. Solar production is an amber curve; EV charging "
                "power is a red spike at plug-in for dumb charging and a green block around midday for the controller. "
                "Bottom: monthly PV self-consumption bars, higher with the controller in every month."),
        "hashtags": ["#IoT", "#Photovoltaik", "#Eigenverbrauch", "#EVCharging", "#OCPP", "#Energiemanagement"],
        "post": post,
    }


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def run_all(results_dir: Path, docs_dir: Path, quick: bool = False) -> dict:
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    n_days = 31 if quick else 366
    jobs = [(sc, "flat", st, n_days) for sc in (("mixed",) if quick else SCENARIOS) for st in STRATEGIES]
    if not quick:
        jobs += [("mixed", "htnt", st, n_days) for st in STRATEGIES]
    print(f"running {len(jobs)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        out = dict(ex.map(_job, jobs))
    if not quick:
        import pickle

        (results_dir / "cache").mkdir(exist_ok=True)
        (results_dir / "cache" / "mixed_flat.pkl").write_bytes(pickle.dumps({st: out[("mixed", "flat", st)] for st in STRATEGIES}))
    all_m: dict[str, dict] = {}
    for (sc, tn, st), r in out.items():
        site = Site(tariff=TARIFFS[tn])
        entry = {"year": compute(r, site)}
        if not quick:
            entry["months"] = {m: compute(r, site, month_slice(m)) for m in range(1, 13)}
            entry["seasons"] = {n: compute(r, site, steps_of(ms)) for n, ms in SEASONS.items()}
        all_m.setdefault(f"{sc}|{tn}", {})[st] = entry
    # feed-in sensitivity (re-pricing the same energy flows)
    sens = {}
    if not quick:
        for fi in (0.04, 0.08, 0.12):
            site = Site(tariff=replace(TARIFFS["flat"], feed_in=fi))
            c = {st: compute(out[("mixed", "flat", st)], site)["cost_chf"] for st in STRATEGIES}
            sens[f"{fi * 100:.0f} Rp/kWh"] = {st: round(c["dumb"] - c[st], 2) for st in ("rule", "optimised")}
    (results_dir / "metrics.json").write_text(json.dumps({"metrics": all_m, "feed_in_sensitivity_chf_saved": sens}, indent=1))
    lines = ["# Results (generated by `python -m pvsc demo`)", ""]
    for key, m in all_m.items():
        df = pd.DataFrame({st: m[st]["year"] for st in STRATEGIES})
        df.loc["chf_saved_vs_dumb"] = [round(df.loc["cost_chf", "dumb"] - df.loc["cost_chf", s], 2) for s in df.columns]
        lines += [f"## {key} (year)", "", _md_table(df), ""]
    if sens:
        lines += ["## Feed-in sensitivity (mixed, flat purchase tariff): CHF saved per year vs dumb", "", json.dumps(sens), ""]
    (results_dir / "summary.md").write_text("\n".join(lines))
    # figures
    key = ("mixed", "flat")
    res = {st: out[(*key, st)] for st in STRATEGIES}
    site = Site()
    monthly = pd.DataFrame({st: [all_m["mixed|flat"][st]["months"][m]["self_consumption_pct"] for m in range(1, 13)]
                            for st in STRATEGIES}) if not quick else pd.DataFrame({st: [0.0] * 12 for st in STRATEGIES})
    sun, worst = pick_days(res, site)
    if not quick:
        make_figure(res, site, monthly, (sun, worst), docs_dir / "figure.png")
        make_post_figure(res, site, monthly, sun, docs_dir / "post-figure.png")
        (docs_dir / "linkedin.json").write_text(json.dumps(linkedin(all_m, None), indent=2, ensure_ascii=False))
        pd.DataFrame({f"{st}_{k}": getattr(res[st], k) for st in STRATEGIES for k in ("grid_kw", "ev_kw")}).iloc[::3].to_csv(
            results_dir / "timeseries_mixed_flat_15min.csv", index=False, float_format="%.3f")
    print((results_dir / "summary.md").read_text())
    return all_m


def render(results_dir: Path, docs_dir: Path) -> None:
    """Regenerate figures and linkedin.json from results/metrics.json and the cached mixed/flat run."""
    import pickle

    all_m = json.loads((results_dir / "metrics.json").read_text())["metrics"]
    res = pickle.loads((results_dir / "cache" / "mixed_flat.pkl").read_bytes())
    site = Site()
    monthly = pd.DataFrame({st: [all_m["mixed|flat"][st]["months"][str(m)]["self_consumption_pct"] for m in range(1, 13)]
                            for st in STRATEGIES})
    sun, worst = pick_days(res, site)
    make_figure(res, site, monthly, (sun, worst), docs_dir / "figure.png")
    make_post_figure(res, site, monthly, sun, docs_dir / "post-figure.png")
    (docs_dir / "linkedin.json").write_text(json.dumps(linkedin(all_m, None), indent=2, ensure_ascii=False))
