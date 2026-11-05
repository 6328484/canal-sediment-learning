"""
Pump cascade (Amu-Zang I + II lumped): energy, efficiency and sediment-induced wear.

Public data used (ADB PCR 30500-013 / RRP): AZ-I design 100 m3/s at 33 m lift; AZ-II 100 m3/s at 44 m lift
(77 m in series); AZ-I measured efficiency 92 % after rehabilitation vs >= 86 % specified;
potential efficiency decline ~5 % / year at AZ-I/II "if maintenance continues at present levels".
ASSUMED: aggregate efficiency curve, friction coefficient, wear law (linear in the cumulative sediment dose),
wear coefficient calibrated so that the reference season at reference turbidity costs 5 % of efficiency.
"""
import numpy as np
from canal import G

RHO = 1000.0


class PumpParams:
    def __init__(self, **kw):
        self.H_total = 77.0          # m at design flow (33 + 44), ADB PCR
        self.Q_design = 100.0        # m3/s
        self.kf = 6.0 / 100.0 ** 2   # friction head: 6 m at design flow (assumed)
        self.eta_bep0 = 0.88         # aggregate best-efficiency (assumed, between 86 % spec and 92 % measured)
        self.Q_bep = 90.0            # m3/s (assumed)
        self.kq = 0.25               # efficiency droop away from BEP (assumed)
        self.Q_max = 105.0
        self.Y_head_ref = 5.0        # reference canal head level (m above head bed)
        self.price = 0.03            # $/kWh (ADB RRP 2003 economic price; present tariff unknown)
        self.k_wear = 0.0            # efficiency loss per kg of sediment passing (set by calibrate_wear)
        for k, v in kw.items():
            setattr(self, k, v)
        self.H_static = self.H_total - self.kf * self.Q_design ** 2

    def calibrate_wear(self, dose_ref_kg, annual_loss=0.05):
        """k_wear such that dose_ref_kg (one reference season) produces annual_loss relative efficiency loss."""
        self.k_wear = annual_loss * self.eta_bep0 / dose_ref_kg


class PumpStation:
    def __init__(self, pp: PumpParams):
        self.pp = pp
        self.dose = 0.0       # cumulative sediment mass passed (kg)
        self.energy_kwh = 0.0
        self.water_m3 = 0.0

    def eta(self, Q, dose=None):
        pp = self.pp
        d = self.dose if dose is None else dose
        shape = 1.0 - pp.kq * (Q / pp.Q_bep - 1.0) ** 2
        return max(pp.eta_bep0 * shape - pp.k_wear * d, 0.3)

    def head(self, Q, Y_head):
        pp = self.pp
        return pp.H_static + (Y_head - pp.Y_head_ref) + pp.kf * Q ** 2

    def power_kw(self, Q, Y_head, dose=None):
        return RHO * G * Q * self.head(Q, Y_head) / (1000.0 * self.eta(Q, dose))

    def step(self, Q, Y_head, c_total, dt):
        P = self.power_kw(Q, Y_head)
        self.energy_kwh += P * dt / 3600.0
        self.dose += c_total * Q * dt
        self.water_m3 += Q * dt
        return P
