"""
Bed-material sediment transport and bed evolution coupled to the Saint-Venant plant
(Saint-Venant-Exner with an advected suspended-load equation, 1-D, first-order exchange).

  d(V C)/dt = Qin*C_up - Qin*C - b*ws*dx*(C - C_eq)          (suspended bed-material load, per segment)
  (1-p) rho_s b dx d(dz)/dt = b*ws*dx*(C - C_eq)             (Exner: bed aggradation dz, m)
  C_eq = mult * Engelund-Hansen total-load capacity           (mult = unknown capacity error, 1 = EH)

Wash load (fraction 1-f_b of the incoming turbidity) is not capacity-limited: it never deposits in the
model; it only matters for pump wear.  d50 of the bed-material fraction and f_b are assumptions
(see docs/parameters.md).  Erosion is limited to the existing deposit (original bed non-erodible).
"""
import numpy as np
from canal import trapezoid, G

RHO_S = 2650.0
RHO = 1000.0
S_REL = RHO_S / RHO
NU = 1.0e-6
POROSITY = 0.4


def settling_velocity(d):
    """Stokes settling velocity (m/s), valid for d < ~0.1 mm."""
    return (S_REL - 1) * G * d * d / (18 * NU)


def eh_capacity(v, R, h_m, Sf, d, shift=0.0, theta_ref=2.5):
    """Engelund-Hansen total-load capacity as mass concentration (kg/m3).

    Phi = 0.1 theta^(5/2) / f,  f = 2 g R Sf / v^2,  theta = R Sf /((s-1) d),  Phi = q_t / sqrt((s-1) g d^3)
    C = rho_s q_t / (v h_m)    (h_m = A/T, mean depth)
    shift != 0 changes the exponent of the Shields parameter, theta^(5/2+shift) theta_ref^(-shift): a structural
    error of the capacity law that leaves the capacity unchanged at theta = theta_ref (plant only; twins use 0).
    """
    v = np.maximum(v, 1e-3)
    Sf = np.maximum(Sf, 1e-9)
    theta = R * Sf / ((S_REL - 1) * d)
    f = 2 * G * R * Sf / v ** 2
    if shift:
        phi = 0.1 * theta ** (2.5 + shift) * theta_ref ** (-shift) / f
    else:
        phi = 0.1 * theta ** 2.5 / f
    qt = phi * np.sqrt((S_REL - 1) * G * d ** 3)
    return RHO_S * qt / (v * h_m)


class SedimentParams:
    def __init__(self, **kw):
        self.d50 = 5.0e-5            # m, bed-material fraction (E3S used d = 0.05 mm)
        self.f_bed = 0.12            # fraction of turbidity that is bed-material load (assumption)
        self.trap = 0.27             # trapping in approach basin (E3S: 1.470 -> 1.067 kg/m3 = 27 %)
        self.mult = 1.0              # TRUE capacity multiplier vs Engelund-Hansen (plant)
        self.kappa_n = 0.5           # roughness growth with aggradation: n = n0 (1 + kappa_n dz/h_ref)
        self.exp_fit = False         # exponentially fitted exchange (exact steady relaxation over a segment)
        self.win = None              # window (m) of the reach-mean aggradation used in capacity_loss (None: worst node)
        for k, v in kw.items():
            setattr(self, k, v)


class Sediment:
    """Sediment state attached to one PumpedCanal (plant or digital twin)."""

    def __init__(self, canal, sp: SedimentParams, mult=None):
        self.canal = canal
        self.sp = sp
        self.mult = sp.mult if mult is None else mult
        self.ws = settling_velocity(sp.d50)
        c = canal
        self.p, self.q = c.seg[:, 0], c.seg[:, 1]
        self.ns = len(self.p)
        self.C = np.zeros(self.ns)            # suspended concentration per segment, kg/m3
        self.dz = np.zeros(self.ns)           # aggradation per segment, m
        self.mass_in = 0.0
        self.mass_dep = 0.0                   # net deposited mass, kg
        self.mass_out = 0.0                   # leaving through offtakes / tail / seepage
        self.cum_dose = 0.0

    def capacity(self, mult=None):
        c = self.canal
        p, q = self.p, self.q
        h = 0.5 * (c.h[p] + c.h[q])
        Q = 0.5 * (c.Q[p] + c.Q[q])
        A, T, P = trapezoid(h, c.par.b, c.par.m)
        R = A / P
        v = Q / A
        n = 0.5 * (c.n_rough[p] + c.n_rough[q])
        Sf = n ** 2 * v ** 2 / R ** (4 / 3)
        return (self.mult if mult is None else mult) * eh_capacity(v, R, A / T, Sf, self.sp.d50,
                                                                  shift=getattr(self.sp, "theta_shift", 0.0))

    def step(self, dt, c_in):
        """Advance suspended load and bed by dt (s).  c_in: bed-material concentration entering the canal (kg/m3).
        Updates the canal bed (zb, roughness, depth at constant water level)."""
        c = self.canal
        par = c.par
        p, q = self.p, self.q
        h = 0.5 * (c.h[p] + c.h[q])
        Qin = np.maximum(c.Q[p], 0.5)
        Qout = np.maximum(c.Q[q], 0.0)
        A, T, P = trapezoid(h, par.b, par.m)
        dx = c.dx
        V = A * dx
        Ceq = self.capacity()
        kex0 = par.b * self.ws * dx                     # m3/s
        fitted = bool(getattr(self.sp, "exp_fit", False))
        if fitted:                                      # k_eff = Q (exp(k/Q) - 1): steady relaxation exact in a segment
            Qr = Qin
            if getattr(self.sp, "exp_fit", False) == 2:  # log-mean discharge: exact for linearly varying Q in the segment
                Qo = np.maximum(Qout, 0.5)
                dQ = Qin - Qo
                big = np.abs(dQ) > 1e-6 * Qin
                Qr = Qin.copy()
                Qr[big] = dQ[big] / np.log(Qin[big] / Qo[big])
            kex = Qin * np.expm1(np.minimum(kex0 / Qr, 30.0))
        else:
            kex = np.full(self.ns, kex0)
        Cn = np.empty(self.ns)
        dep = np.empty(self.ns)
        Cup = c_in
        rho_b = RHO_S * (1 - POROSITY)
        for j in range(self.ns):
            den = V[j] / dt + Qin[j] + kex[j]
            num = V[j] * self.C[j] / dt + Qin[j] * Cup + kex[j] * Ceq[j]
            Cj = num / den
            if fitted:                                  # deposit from the segment balance (robust for large k_eff)
                mass = (Qin[j] * (Cup - Cj) - V[j] * (Cj - self.C[j]) / dt) * dt
            else:
                mass = kex[j] * (Cj - Ceq[j]) * dt      # kg deposited (+) / eroded (-)
            avail = rho_b * self.dz[j] * par.b * dx
            if mass < -avail:                            # erosion limited by deposit
                mass = -avail
                Cj = (V[j] * self.C[j] / dt + Qin[j] * Cup - mass / dt) / (V[j] / dt + Qin[j])
            Cn[j] = Cj
            dep[j] = mass
            Cup = Cj
        # budget
        self.mass_in += c.Q[0] * c_in * dt
        susp_change = np.sum(V * (Cn - self.C))
        self.mass_dep += float(np.sum(dep))
        withdrawn = np.sum((Qin - Qout) * Cn) * dt + Qout[-1] * Cn[-1] * dt
        self.mass_out += withdrawn
        self._last_budget = (c.Q[0] * c_in * dt, susp_change, float(np.sum(dep)), withdrawn)
        self.C = Cn
        ddz = dep / (rho_b * par.b * dx)
        self.dz += ddz
        self._apply_bed(ddz)
        return dep

    def node_dz(self):
        """Aggradation interpolated to canal nodes (structure nodes copy the neighbour)."""
        c = self.canal
        dzn = np.zeros(c.M)
        cnt = np.zeros(c.M)
        np.add.at(dzn, self.p, self.dz)
        np.add.at(dzn, self.q, self.dz)
        np.add.at(cnt, self.p, 1.0)
        np.add.at(cnt, self.q, 1.0)
        dzn = dzn / np.maximum(cnt, 1)
        # structure pairs: use the upstream-side value on both sides
        for pu, pd in c.struct:
            dzn[pd] = dzn[pu]
        return dzn

    def _apply_bed(self, ddz_seg):
        c = self.canal
        old = c.zb.copy()
        c.set_bed(self.node_dz(), kappa_n=self.sp.kappa_n)
        c.h = np.maximum(c.h - (c.zb - old), 0.05)       # keep water level unchanged

    def dz_window(self, W):
        """Segment aggradation averaged over a sliding window of fixed length W (m) kept inside the canal."""
        c = self.canal
        dx, L = c.dx, c.par.L
        ns = self.ns
        edges = np.arange(ns + 1) * dx
        F = np.concatenate([[0.0], np.cumsum(self.dz) * dx])
        W = min(float(W), L)
        xm = (np.arange(ns) + 0.5) * dx
        a = np.clip(xm - 0.5 * W, 0.0, L - W)
        return (np.interp(a + W, edges, F) - np.interp(a, edges, F)) / W

    def capacity_loss(self, Q_ref=120.0, freeboard=0.5, window="default"):
        """Fractional loss of design capacity: the largest steady uniform-flow discharge whose level stays
        below (bank crest - freeboard), worst segment, relative to the clean canal."""
        c = self.canal
        par = c.par
        hmax_clean = par.h_bank - freeboard

        def qcap(dz, n):
            h = max(hmax_clean - dz, 0.1)
            A, T, P = trapezoid(h, par.b, par.m)
            R = A / P
            return A * R ** (2 / 3) * np.sqrt(par.S0) / n
        q0 = qcap(0.0, par.n0)
        W = getattr(self.sp, "win", None) if window == "default" else window
        if W:
            dzs = self.dz_window(W)
            qs = [qcap(d, par.n0 * (1.0 + self.sp.kappa_n * min(max(d / par.h_ref, 0.0), 1.0))) for d in dzs]
            return 1.0 - min(qs) / q0, min(qs)
        dzn = self.node_dz()
        qs = [qcap(dzn[i], c.n_rough[i]) for i in range(c.M)]
        return 1.0 - min(qs) / q0, min(qs)
