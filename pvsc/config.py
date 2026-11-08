"""Static configuration: site, tariff, grid connection, devices. All values are plain numbers in SI-ish units
(kW, kWh, A, V, degC, CHF)."""
from __future__ import annotations

from dataclasses import dataclass, field

DT_H = 5 / 60  # simulation step: 5 minutes
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 288
VOLTAGE = 230.0  # V line-to-neutral


@dataclass(frozen=True)
class Tariff:
    """Swiss example tariff, CHF/kWh, excluding VAT.

    buy: ElCom electricity price 2024, category H4 (4500 kWh/a household), the Zurich city utility (operator id 565 in the dataset),
         standard product: 24.054 Rp/kWh total (energy 7.384 + grid 12.27 + levies 4.4).
         Source: ElCom via LINDAS / opendata.swiss "Strompreis pro Stromnetzbetreiber".
    feed_in: ASSUMED round example value of 8 Rp/kWh. Feed-in tariffs differ per utility and are not
         published by ElCom; replace with your utility's current Rueckliefertarif.
    """

    buy: float = 0.24054
    feed_in: float = 0.08
    nt_buy: float | None = None  # optional low-tariff price; None = flat tariff

    def buy_at(self, step: int) -> float:
        """Purchase price at a 5-minute step. With nt_buy set: high tariff Mon-Fri 06-22 h, low tariff otherwise."""
        if self.nt_buy is None:
            return self.buy
        weekday = (step // STEPS_PER_DAY) % 7
        hod = (step % STEPS_PER_DAY) / STEPS_PER_HOUR
        return self.buy if (weekday < 5 and 6.0 <= hod < 22.0) else self.nt_buy


@dataclass(frozen=True)
class GridConnection:
    limit_a: float = 25.0  # per phase, house connection fuse
    margin_a: float = 1.0  # controller keeps this far below the limit


@dataclass(frozen=True)
class EvSpec:
    capacity_kwh: float = 60.0
    min_a: float = 6.0
    max_a: float = 16.0
    target_soc: float = 0.80
    efficiency: float = 0.93  # AC side -> battery
    consumption_kwh_100km: float = 18.0

    def power_kw(self, amps: float, phases: int) -> float:
        return amps * VOLTAGE * phases / 1000.0

    @property
    def min_kw_1p(self) -> float:
        return self.power_kw(self.min_a, 1)

    @property
    def max_kw_1p(self) -> float:
        return self.power_kw(self.max_a, 1)

    @property
    def min_kw_3p(self) -> float:
        return self.power_kw(self.min_a, 3)

    @property
    def max_kw_3p(self) -> float:
        return self.power_kw(self.max_a, 3)


@dataclass(frozen=True)
class PvSpec:
    kwp: float = 8.0
    tilt_deg: float = 30.0
    azimuth_deg: float = 0.0  # south
    system_factor: float = 0.80  # soiling, wiring, inverter efficiency
    temp_coeff: float = -0.0035  # per K
    inverter_kw: float = 8.0


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float = 10.0
    power_kw: float = 5.0
    efficiency: float = 0.95  # each way
    min_soc: float = 0.10


@dataclass(frozen=True)
class HeatPumpSpec:
    """Air-source heat pump with SG-Ready style input, thermostat on indoor temperature."""

    thermal_kw: float = 6.0
    ua_kw_per_k: float = 0.16
    capacity_kwh_per_k: float = 8.0
    setpoint_c: float = 20.5
    band_c: float = 0.3
    boost_c: float = 1.0  # SG state 3 raises the setpoint by this much
    heating_limit_c: float = 15.0


@dataclass(frozen=True)
class BoilerSpec:
    """Domestic hot water tank, 300 l, one resistive element on L1."""

    power_kw: float = 3.0
    capacity_kwh_per_k: float = 0.35
    loss_kw: float = 0.05
    min_c: float = 48.0  # comfort floor: controller forces heating below this
    normal_c: float = 55.0
    max_c: float = 62.0
    draw_kwh_per_day: float = 4.5


@dataclass(frozen=True)
class ControllerConfig:
    start_kw: float = 1.5  # surplus needed to start EV charging
    stop_kw: float = 1.1  # EV stops when surplus stays below this
    stop_delay_steps: int = 2  # 10 min
    min_run_steps: int = 2  # 10 min between start and stop
    phase_up_kw: float = 4.6  # 1p -> 3p when surplus above this
    phase_down_kw: float = 4.2  # 3p -> 1p when surplus below this
    phase_dwell_steps: int = 3  # 15 min between phase changes
    boiler_on_kw: float = 3.0
    boiler_off_kw: float = 1.0
    boiler_min_run_steps: int = 6
    boiler_min_c: float = 48.0
    boiler_max_c: float = 62.0
    sg_on_kw: float = 1.2
    sg_off_kw: float = 0.4
    sg_min_steps: int = 6
    deadline_margin_h: float = 0.5
    priority: tuple[str, ...] = ("ev", "boiler", "sg")


@dataclass(frozen=True)
class Site:
    tariff: Tariff = field(default_factory=Tariff)
    grid: GridConnection = field(default_factory=GridConnection)
    ev: EvSpec = field(default_factory=EvSpec)
    pv: PvSpec = field(default_factory=PvSpec)
    battery: BatterySpec = field(default_factory=BatterySpec)
    hp: HeatPumpSpec = field(default_factory=HeatPumpSpec)
    boiler: BoilerSpec = field(default_factory=BoilerSpec)
    ctl: ControllerConfig = field(default_factory=ControllerConfig)
    latitude: float = 47.3769
    longitude: float = 8.5417
