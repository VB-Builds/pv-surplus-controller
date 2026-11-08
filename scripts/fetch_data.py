"""Download hourly irradiance for Zurich (2024) from Open-Meteo and cache it as CSV.

Observed: Open-Meteo Historical Weather API (reanalysis based).
Forecast: Open-Meteo Historical Forecast API (archived model forecasts, used as the day-ahead forecast).
Licence of the data: CC BY 4.0 (https://open-meteo.com/en/license). Attribution: "Weather data by Open-Meteo.com".
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

LAT, LON = 47.3769, 8.5417  # Zurich
TILT, AZIMUTH = 30, 0  # south-facing 30 deg roof (Open-Meteo: azimuth 0 = south)
VARS = "global_tilted_irradiance,temperature_2m"
APIS = {
    "observed": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}


def fetch(api: str, start: str, end: str) -> pd.DataFrame:
    url = (
        f"{api}?latitude={LAT}&longitude={LON}&start_date={start}&end_date={end}"
        f"&hourly={VARS}&tilt={TILT}&azimuth={AZIMUTH}&timezone=Europe%2FZurich"
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            h = data["hourly"]
            return pd.DataFrame(
                {"time": pd.to_datetime(h["time"]), "gti_wm2": h["global_tilted_irradiance"], "temp_c": h["temperature_2m"]}
            )
        except Exception as exc:  # noqa: BLE001
            print("retry", attempt, exc, file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"failed {url}")


def main(year: int = 2024) -> None:
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    for name, api in APIS.items():
        parts = []
        for m in range(1, 13):
            s = pd.Timestamp(year, m, 1)
            e = s + pd.offsets.MonthEnd(0)
            parts.append(fetch(api, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out / f"zurich_{year}_{name}.csv", index=False)
        print(name, len(df), int(df.gti_wm2.isna().sum()), "NaN")


if __name__ == "__main__":
    main()
