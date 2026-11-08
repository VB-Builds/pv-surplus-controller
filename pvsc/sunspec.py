"""SunSpec-style Modbus TCP register map for a simulated 3-phase PV inverter and grid meter.

Layout (holding registers, 16 bit, big endian), a subset of the public SunSpec information model:
    40000-40001  "SunS" marker
    40002        inverter model id 103 (three phase), 40003 length 50
    40004+       model 103 points at SunSpec offsets: A=2 AphA=3 AphB=4 AphC=5 A_SF=6, PhVphA=10..12 V_SF=13,
                 W=14 W_SF=15, Hz=16 Hz_SF=17, WH=24-25 (acc32) WH_SF=26, St=38
    40054        meter model id 203 (three phase wye), length 105; points at offsets: A=2 AphA..C=3..5 A_SF=6,
                 PhVphA..C=8..10 V_SF=15, Hz=16 Hz_SF=17, W=18 WphA..C=19..21 W_SF=22,
                 TotWhExp=38-39 TotWhImp=46-47 TotWh_SF=54
    49900+       simulator side channel (NOT SunSpec): virtual clock, EV power, relay commands

Sign convention of this project: meter power and current are positive for import from the grid.
This is not a certified SunSpec implementation; it is a generic, documented subset for simulation and tests.
"""
from __future__ import annotations

from dataclasses import dataclass

from pymodbus.client import AsyncModbusTcpClient
from pymodbus.constants import ExcCodes
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

BASE = 40000
INV = BASE + 2  # model 103 header
MET = BASE + 54  # model 203 header
SIM = 49900
N_REGS = 10000

A_SF, V_SF, HZ_SF, W_SF = -2, -1, -2, 0
SIM_CLOCK = SIM  # u32: virtual minutes since 2024-01-01 00:00
SIM_EV_W = SIM + 2  # u32 total EV power in W (written by the charge point simulator)
SIM_EV_PHASES = SIM + 4
SIM_BOILER_CMD = SIM + 5  # 1 = boiler relay closed (written by the controller)
SIM_SG_K1 = SIM + 6
SIM_SG_K2 = SIM + 7
SIM_HP_W = SIM + 8  # heat pump electric draw, W (plant side, informational)
SIM_BOILER_W = SIM + 9


def to_s16(v: int) -> int:
    v = int(round(v))
    if not -32768 <= v <= 32767:
        raise ValueError(f"value {v} does not fit int16")
    return v & 0xFFFF


def from_s16(raw: int) -> int:
    return raw - 0x10000 if raw >= 0x8000 else raw


def scaled(value: float, sf: int) -> int:
    """Encode value with SunSpec scale factor sf (real = raw * 10**sf)."""
    return to_s16(value / (10.0 ** sf))


def unscaled(raw: int, sf: int) -> float:
    return from_s16(raw) * (10.0 ** sf)


def u32_words(v: int) -> tuple[int, int]:
    v = int(v)
    if not 0 <= v <= 0xFFFFFFFF:
        raise ValueError("u32 out of range")
    return (v >> 16) & 0xFFFF, v & 0xFFFF


def u32_from(words) -> int:
    return (int(words[0]) << 16) | int(words[1])


@dataclass
class Snapshot:
    """Decoded reading: what the controller sees over Modbus."""

    pv_w: float
    pv_wh: int
    grid_w: float  # + import
    grid_phase_w: tuple[float, float, float]
    grid_phase_a: tuple[float, float, float]  # import current per phase (>= 0 magnitude is not enforced)
    volts: tuple[float, float, float]
    clock_min: int
    ev_w: float
    status: int


class PlantRegisters:
    """Server side data model; updated by the plant simulation."""

    def __init__(self) -> None:
        self.mem = [0] * N_REGS  # authoritative copy, index = address - BASE
        self.set_words(BASE, [0x5375, 0x6E53])
        self.set_words(INV, [103, 50])
        self.set_words(INV + 2 + 6, [to_s16(A_SF)])
        self.set_words(INV + 2 + 13, [to_s16(V_SF)])
        self.set_words(INV + 2 + 15, [to_s16(W_SF)])
        self.set_words(INV + 2 + 17, [to_s16(HZ_SF)])
        self.set_words(MET, [203, 105])
        self.set_words(MET + 2 + 6, [to_s16(A_SF)])
        self.set_words(MET + 2 + 15, [to_s16(V_SF)])
        self.set_words(MET + 2 + 17, [to_s16(HZ_SF)])
        self.set_words(MET + 2 + 22, [to_s16(W_SF)])
        self.device = SimDevice(
            0, simdata=[SimData(BASE, values=[0] * N_REGS, datatype=DataType.REGISTERS)], action=self._action
        )
        self._pv_wh = 0.0
        self._imp_wh = 0.0
        self._exp_wh = 0.0

    async def _action(self, function_code, start_address, address, count, current_registers, set_values):
        """Serve reads from self.mem; remember writes (clients may write the simulator side channel)."""
        off = address - start_address
        if set_values is not None:
            for k, v in enumerate(set_values):
                if 0 <= address + k - BASE < N_REGS:
                    self.mem[address + k - BASE] = int(v) & 0xFFFF
            return None
        if address < BASE or address + count > BASE + N_REGS:
            return ExcCodes.ILLEGAL_ADDRESS
        current_registers[off: off + count] = self.mem[address - BASE: address - BASE + count]
        return None

    def set_words(self, address: int, words: list[int]) -> None:
        self.mem[address - BASE: address - BASE + len(words)] = [int(w) & 0xFFFF for w in words]

    def get_words(self, address: int, count: int = 1) -> list[int]:
        return self.mem[address - BASE: address - BASE + count]

    def update(self, pv_w: float, grid_phase_w: tuple[float, float, float], volts=(230.0, 230.0, 230.0),
               dt_h: float = 0.0, clock_min: int | None = None, hz: float = 50.0) -> None:
        """Write inverter + meter points. grid_phase_w positive = import."""
        pv_w = max(0.0, pv_w)
        pv_a = pv_w / 3.0 / volts[0]
        self._pv_wh += pv_w * dt_h
        grid_w = sum(grid_phase_w)
        self._imp_wh += max(0.0, grid_w) * dt_h
        self._exp_wh += max(0.0, -grid_w) * dt_h
        i = INV + 2
        self.set_words(i + 2, [scaled(pv_a * 3, A_SF)])
        self.set_words(i + 3, [scaled(pv_a, A_SF)] * 3)
        self.set_words(i + 10, [scaled(v, V_SF) for v in volts])
        self.set_words(i + 14, [scaled(pv_w, W_SF)])
        self.set_words(i + 16, [scaled(hz, HZ_SF)])
        self.set_words(i + 24, list(u32_words(int(self._pv_wh))))
        self.set_words(i + 38, [4 if pv_w > 0 else 2])  # St: 4 = MPPT, 2 = sleeping
        m = MET + 2
        amps = [w / v for w, v in zip(grid_phase_w, volts, strict=True)]
        self.set_words(m + 2, [scaled(sum(abs(a) for a in amps), A_SF)])
        self.set_words(m + 3, [scaled(a, A_SF) for a in amps])
        self.set_words(m + 8, [scaled(v, V_SF) for v in volts])
        self.set_words(m + 16, [scaled(hz, HZ_SF)])
        self.set_words(m + 18, [scaled(grid_w, W_SF)])
        self.set_words(m + 19, [scaled(w, W_SF) for w in grid_phase_w])
        self.set_words(m + 38, list(u32_words(int(self._exp_wh))))
        self.set_words(m + 46, list(u32_words(int(self._imp_wh))))
        if clock_min is not None:
            self.set_words(SIM_CLOCK, list(u32_words(clock_min)))


def make_server(regs: PlantRegisters, host: str = "127.0.0.1", port: int = 5020) -> ModbusTcpServer:
    return ModbusTcpServer(regs.device, address=(host, port))


async def read_snapshot(client: AsyncModbusTcpClient) -> Snapshot:
    """Read inverter, meter and simulator side channel with three requests."""
    inv = await client.read_holding_registers(INV, count=42)
    met = await client.read_holding_registers(MET, count=58)
    sim = await client.read_holding_registers(SIM, count=10)
    for rr in (inv, met, sim):
        if rr.isError():
            raise ConnectionError(f"modbus read failed: {rr}")
    r_i, r_m, r_s = inv.registers, met.registers, sim.registers
    if r_i[0] != 103 or r_m[0] != 203:
        raise ValueError(f"unexpected model ids {r_i[0]}/{r_m[0]}")
    return decode_snapshot(r_i, r_m, r_s)


def decode_snapshot(r_i: list[int], r_m: list[int], r_s: list[int]) -> Snapshot:
    a_sf, v_sf, w_sf = (from_s16(r_i[2 + 6]), from_s16(r_i[2 + 13]), from_s16(r_i[2 + 15]))
    ma_sf, mv_sf, mw_sf = (from_s16(r_m[2 + 6]), from_s16(r_m[2 + 15]), from_s16(r_m[2 + 22]))
    del a_sf
    return Snapshot(
        pv_w=unscaled(r_i[2 + 14], w_sf),
        pv_wh=u32_from(r_i[2 + 24: 2 + 26]),
        grid_w=unscaled(r_m[2 + 18], mw_sf),
        grid_phase_w=tuple(unscaled(r_m[2 + 19 + p], mw_sf) for p in range(3)),
        grid_phase_a=tuple(unscaled(r_m[2 + 3 + p], ma_sf) for p in range(3)),
        volts=tuple(unscaled(r_m[2 + 8 + p], mv_sf) for p in range(3)),
        clock_min=u32_from(r_s[0:2]),
        ev_w=float(u32_from(r_s[2:4])),
        status=r_i[2 + 38],
    ) if v_sf == V_SF else (_ for _ in ()).throw(ValueError("unexpected scale factors"))
