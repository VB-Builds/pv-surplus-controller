"""Command line: `python -m pvsc <command>`.

demo         year simulation (3 strategies x 3 EV scenarios + tariff variants), writes results/ and docs/ figures
demo-live    closed loop over OCPP + Modbus TCP with simulated devices (one process, one virtual day)
inverter     docker role: simulated inverter/meter/house (Modbus TCP server, owns the virtual clock)
charger      docker role: simulated OCPP charge point + EV
controller   docker role: central system + surplus controller
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _hostport(s: str, default_port: int) -> tuple[str, int]:
    host, _, port = s.partition(":")
    return host, int(port) if port else default_port


def cmd_demo(args: argparse.Namespace) -> int:
    from .report import run_all

    run_all(ROOT / "results", ROOT / "docs", quick=args.quick)
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from .report import render

    render(ROOT / "results", ROOT / "docs")
    return 0


def cmd_demo_live(args: argparse.Namespace) -> int:
    from .live import limit_scenario, run_live

    tick = max(0.002, args.seconds / 288.0)
    sc = limit_scenario() if args.limit_demo else None
    res = asyncio.run(run_live(tick_s=tick, scenario=sc, modbus_port=args.modbus_port, ocpp_port=args.ocpp_port))
    rows = res.pop("rows")
    res.pop("ctl_rows")
    print(json.dumps(res, indent=2))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / ("live_limit_demo.csv" if args.limit_demo else "live_demo.csv"), index=False)
    ok = res["steps_over_limit"] == 0 and res["ev_kwh"] > 0 and res["ocpp_profiles_sent"] > 0
    print("live demo:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_inverter(args: argparse.Namespace) -> int:
    from . import sunspec as S
    from .live import Plant, Scenario, best_summer_day

    async def main() -> None:
        regs = S.PlantRegisters()
        srv = S.make_server(regs, host="0.0.0.0", port=args.port)  # noqa: S104 - container role
        task = asyncio.create_task(srv.serve_forever())
        await asyncio.sleep(args.start_delay)
        stop = asyncio.Event()
        plant = Plant(regs, Scenario(day=best_summer_day()), tick_s=args.tick)
        await plant.run(stop)
        import pandas as pd

        (ROOT / "results").mkdir(exist_ok=True)
        pd.DataFrame(plant.rows).to_csv(ROOT / "results" / "live_compose.csv", index=False)
        print("virtual day finished; wrote results/live_compose.csv")
        await srv.shutdown()
        task.cancel()

    asyncio.run(main())
    return 0


def cmd_charger(args: argparse.Namespace) -> int:
    from .live import Scenario, best_summer_day, run_charger

    host, port = _hostport(args.modbus, 5020)

    async def main() -> None:
        res = await run_charger(args.ocpp, host, port, Scenario(day=best_summer_day()), asyncio.Event())
        print(json.dumps({"delivered_kwh": res["delivered_kwh"], "profiles_received": len(res["profiles"])}))

    asyncio.run(main())
    return 0


def cmd_controller(args: argparse.Namespace) -> int:
    from .live import Scenario, best_summer_day, run_controller
    from .ocpp_cs import CentralSystem

    host, port = _hostport(args.modbus, 5020)

    async def main() -> None:
        central = CentralSystem(host="0.0.0.0", port=args.ocpp_port)  # noqa: S104 - container role
        await central.start()
        await central.wait_for("CP_1", 120)
        stop = asyncio.Event()
        res = await asyncio.wait_for(run_controller(central, host, port, Scenario(day=best_summer_day()), stop), 3600)
        print(json.dumps({"max_phase_a": round(res.max_phase_a, 2), "steps_over_limit": res.steps_over_limit,
                          "delivered_kwh": round(res.delivered_kwh, 2), "profiles_sent": res.profiles_sent}))
        await central.stop()

    asyncio.run(main())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pvsc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="year simulation + figures")
    d.add_argument("--quick", action="store_true", help="January and July only (fast check)")
    d.set_defaults(fn=cmd_demo)
    fg = sub.add_parser("figures", help="re-render figures from the cached demo run")
    fg.set_defaults(fn=cmd_figures)
    dl = sub.add_parser("demo-live", help="closed loop over OCPP + Modbus TCP (simulated devices)")
    dl.add_argument("--seconds", type=float, default=15.0, help="wall time for one virtual day")
    dl.add_argument("--limit-demo", action="store_true", help="winter evening scenario that stresses the 25 A limit")
    dl.add_argument("--modbus-port", type=int, default=5020)
    dl.add_argument("--ocpp-port", type=int, default=9000)
    dl.set_defaults(fn=cmd_demo_live)
    inv = sub.add_parser("inverter", help="container role: simulated inverter/meter/house")
    inv.add_argument("--port", type=int, default=5020)
    inv.add_argument("--tick", type=float, default=0.2)
    inv.add_argument("--start-delay", type=float, default=8.0)
    inv.set_defaults(fn=cmd_inverter)
    ch = sub.add_parser("charger", help="container role: simulated OCPP charge point + EV")
    ch.add_argument("--ocpp", default="ws://controller:9000")
    ch.add_argument("--modbus", default="inverter:5020")
    ch.set_defaults(fn=cmd_charger)
    co = sub.add_parser("controller", help="container role: central system + surplus controller")
    co.add_argument("--ocpp-port", type=int, default=9000)
    co.add_argument("--modbus", default="inverter:5020")
    co.set_defaults(fn=cmd_controller)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
