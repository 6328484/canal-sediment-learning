"""Steady gradually varied flow (standard-step) capacity of the aggraded canal.

capacity(): largest steady discharge for which the water surface stays below (bank crest - freeboard) everywhere,
with a normal-depth boundary at the tail and the check structures fully open (no head loss).
The aggradation profile is taken as piecewise constant on the plant segments and sampled on a fixed fine grid, so the
result does not depend on the plant mesh.
"""
import math
import numpy as np

G = 9.80665


def _geom(h, b, m):
    A = (b + m * h) * h
    P = b + 2.0 * h * math.sqrt(1.0 + m * m)
    return A, P


def _sf(h, Q, n, b, m):
    A, P = _geom(h, b, m)
    R = A / P
    return (n * Q / (A * R ** (2.0 / 3.0))) ** 2


def normal_depth(Q, n, S, b, m):
    lo, hi = 0.05, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        A, P = _geom(mid, b, m)
        q = A * (A / P) ** (2.0 / 3.0) * math.sqrt(S) / n
        if q < Q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def profile(Q, dzx, nx, zb0x, dx, b, m, S0):
    """Depths h_i on the fine grid (index 0 = head).  dzx, nx, zb0x: aggradation, roughness, original bed at fine nodes."""
    M = len(dzx)
    zb = zb0x + dzx
    h = np.empty(M)
    h[-1] = normal_depth(Q, nx[-1], S0, b, m)
    for i in range(M - 2, -1, -1):
        A1, _ = _geom(h[i + 1], b, m)
        E1 = zb[i + 1] + h[i + 1] + Q * Q / (2 * G * A1 * A1)
        sf1 = _sf(h[i + 1], Q, nx[i + 1], b, m)
        lo, hi = 0.05, 40.0
        for _ in range(48):                       # bisection: E(h) is increasing for subcritical flow
            mid = 0.5 * (lo + hi)
            A0, _ = _geom(mid, b, m)
            f = zb[i] + mid + Q * Q / (2 * G * A0 * A0) + 0.5 * dx * (_sf(mid, Q, nx[i], b, m) + sf1) - E1 - 0.0
            # energy at i (upstream) equals energy at i+1 plus friction loss; f is E_i + h_f - E_{i+1}... see below
            lo, hi = (mid, hi) if f < 0 else (lo, mid)
        h[i] = 0.5 * (lo + hi)
    return h


def _profile(Q, zb, nx, dx, b, m, S0):
    """Standard step from the tail: E_i = E_{i+1} + 0.5 dx (Sf_i + Sf_{i+1}), with E = zb + h + v^2/2g."""
    M = len(zb)
    h = np.empty(M)
    h[-1] = normal_depth(Q, nx[-1], S0, b, m)
    for i in range(M - 2, -1, -1):
        A1, _ = _geom(h[i + 1], b, m)
        E1 = zb[i + 1] + h[i + 1] + Q * Q / (2 * G * A1 * A1)
        sf1 = _sf(h[i + 1], Q, nx[i + 1], b, m)
        lo, hi = 0.05, 40.0
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            A0, _ = _geom(mid, b, m)
            Ei = zb[i] + mid + Q * Q / (2 * G * A0 * A0)
            f = Ei - E1 - 0.5 * dx * (_sf(mid, Q, nx[i], b, m) + sf1)
            if f < 0:
                lo = mid
            else:
                hi = mid
        h[i] = 0.5 * (lo + hi)
    return h


def fine_grid(par, dz_seg, kappa_n, dxf=100.0):
    """Sample the piecewise-constant segment aggradation on a fine uniform grid."""
    Nf = int(round(par.L / dxf))
    x = np.linspace(0.0, par.L, Nf + 1)
    ns = len(dz_seg)
    idx = np.minimum((x / (par.L / ns)).astype(int), ns - 1)
    dz = np.asarray(dz_seg)[idx]
    n = par.n0 * (1.0 + kappa_n * np.clip(dz / par.h_ref, 0.0, 1.0))
    zb0 = -par.S0 * x
    return x, dz, n, zb0, par.L / Nf


def margin(Q, par, x, dz, n, zb0, dx, freeboard):
    zb = zb0 + dz
    h = _profile(Q, zb, n, dx, par.b, par.m, par.S0)
    return float(np.max(zb + h - (zb0 + par.h_bank - freeboard)))


def capacity(par, dz_seg, kappa_n=0.5, freeboard=0.5, dxf=100.0, tol=1e-5):
    x, dz, n, zb0, dx = fine_grid(par, dz_seg, kappa_n, dxf)
    lo, hi = 20.0, 400.0
    while hi - lo > tol * lo:
        mid = 0.5 * (lo + hi)
        if margin(mid, par, x, dz, n, zb0, dx, freeboard) <= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def capacity_loss(par, dz_seg, kappa_n=0.5, freeboard=0.5, dxf=100.0, q0=None):
    if q0 is None:
        q0 = capacity(par, np.zeros(len(dz_seg)), kappa_n, freeboard, dxf)
    q = capacity(par, dz_seg, kappa_n, freeboard, dxf)
    return 1.0 - q / q0, q, q0
