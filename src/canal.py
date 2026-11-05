"""
Amu-Zang-type pumped main canal: 1-D Saint-Venant model (Preissmann scheme) with gated check
structures, tail gate, offtakes, seepage and an evolving bed.

PARAMETERISATION (public data, see docs/parameters.md):
  length 56.6 km, trapezoid b = 17 m, m = 1.7, S0 = 6e-5, n0 = 0.030 (derived from 75 m3/s, 5 m, 0.6 m/s)
  [Samiev et al. 2024, E3S 508, 07006]; capacity 120 m3/s.
ILLUSTRATIVE (NOT from any passport): positions of check structures, offtakes and their shares,
  bank height, gate widths.  Replace with passport data when available.

Equations (ordinary segment between nodes p, q):
  dA/dt + dQ/dx + q_lat = 0
  dQ/dt + d(Q^2/A)/dx + g A dY/dx + g A Sf = 0,    Y = zb + h
Preissmann theta-scheme, Newton iteration, analytic Jacobian, dense solve (about 120 unknowns).
"""
import numpy as np
from scipy.linalg import solve_banded

G = 9.81


class CanalParams:
    def __init__(self, **kw):
        self.L = 56600.0            # m
        self.N = 57                 # segments (dx ~ 993 m)
        self.b = 17.0               # bottom width, m
        self.m = 1.7                # side slope (H:V)
        self.S0 = 6e-5              # bed slope
        self.n0 = 0.030             # Manning n (clean bed)
        self.h_bank = 7.0           # bank height above ORIGINAL bed, m (illustrative)
        self.h_ref = 5.0            # reference (target) depth at pool ends, m
        self.seep = 3.0e-6          # seepage rate, m/s (0.26 m/day)
        self.theta = 0.65
        self.dt = 1800.0            # s
        # check structures: positions (km) of gates between pools (illustrative)
        self.check_km = [14.2, 28.3, 42.5]
        self.gate_W = 30.0          # total gate width, m
        self.gate_amax = 5.0        # m
        self.mu_model = 0.65        # discharge coefficient assumed by controllers
        self.mu_plant = 0.65 * 1.04  # actual (4 % higher): controller/plant mismatch
        self.tail_W = 10.0
        self.tail_amax = 4.0
        # offtakes (illustrative): (km, share of total offtake demand, name)
        self.offtakes = [(10.0, 0.04, "AZ local 1"),
                         (22.0, 0.05, "AZ local 2"),
                         (35.0, 0.50, "Zang-type branch"),
                         (44.0, 0.12, "AZ local 3"),
                         (50.0, 0.24, "Babatag-type branch"),
                         (56.6, 0.05, "tail users")]   # the tail share is delivered through the tail gate
        self.sill_above_bed = 1.0   # offtake sill height above original bed, m
        self.off_head_min = 3.0     # minimum head over the sill for full scheduled discharge, m (illustrative)
        for k, v in kw.items():
            setattr(self, k, v)


def trapezoid(h, b, m):
    A = (b + m * h) * h
    T = b + 2 * m * h
    P = b + 2 * h * np.sqrt(1 + m * m)
    return A, T, P


class PumpedCanal:
    """Plant model.  State: h (depth above current bed), Q at all nodes (structure nodes duplicated)."""

    def __init__(self, par: CanalParams = None):
        self.par = par = par or CanalParams()
        N = par.N
        dx = par.L / N
        self.dx = dx
        x_nodes = [0.0]
        self.struct_after = []          # node index p (upstream side) for each check structure
        chk_idx = sorted(set(int(round(k * 1000 / dx)) for k in par.check_km))
        node_x = []
        seg_pairs = []
        struct_pairs = []
        cur = 0
        node_x.append(0.0)
        for i in range(1, N + 1):
            node_x.append(i * dx)
            seg_pairs.append((len(node_x) - 2, len(node_x) - 1))
            if i in chk_idx and i < N:
                # duplicate node for downstream side of the structure
                node_x.append(i * dx)
                struct_pairs.append((len(node_x) - 2, len(node_x) - 1))
        self.x = np.array(node_x)
        self.M = len(self.x)            # number of nodes
        self.seg = np.array(seg_pairs)
        self.struct = np.array(struct_pairs)
        self.n_struct = len(struct_pairs)
        self.zb0 = -par.S0 * self.x     # original bed elevation (m), head bed = 0
        self.zb = self.zb0.copy()       # current bed elevation (changes with sediment)
        self.n_rough = np.full(self.M, par.n0)
        # pool index for every node (0..n_struct)
        pool = np.zeros(self.M, int)
        for k, (p, q) in enumerate(struct_pairs):
            pool[q:] = k + 1
        self.pool = pool
        self.pool_end_node = np.array([p for p, q in struct_pairs] + [self.M - 1])
        self.pool_start_node = np.array([0] + [q for p, q in struct_pairs])
        # offtake -> segment index (distributed over that segment); tail share goes through tail gate
        self.off_seg, self.off_share, self.off_name, self.off_node = [], [], [], []
        for km, sh, nm in par.offtakes:
            if nm == "tail users":
                continue
            xs = km * 1000.0
            # choose segment whose mid-point is closest and which is not a structure pair
            mids = 0.5 * (self.x[self.seg[:, 0]] + self.x[self.seg[:, 1]])
            j = int(np.argmin(np.abs(mids - xs)))
            self.off_seg.append(j)
            self.off_share.append(sh)
            self.off_name.append(nm)
            self.off_node.append(self.seg[j, 1])
        self.off_seg = np.array(self.off_seg)
        self.off_share = np.array(self.off_share)
        self.off_node = np.array(self.off_node)
        self.tail_share = [sh for km, sh, nm in par.offtakes if nm == "tail users"][0]
        # offtake pool (pool index of the offtake node)
        self.off_pool = pool[self.off_node]
        self.sill = self.zb0[self.off_node] + par.sill_above_bed
        self.head_ref = par.h_ref - par.sill_above_bed
        self.t = 0.0
        self.h = np.full(self.M, par.h_ref)
        self.Q = np.full(self.M, 60.0)
        self.gate_a = np.full(self.n_struct, 2.0)
        self.tail_a = 0.2
        self.last_off_q = np.zeros(len(self.off_seg))
        self.last_seep = 0.0
        self.last_iter = 0
        self.mass_err = 0.0

    # ----------------------------------------------------------------- helpers
    def set_bed(self, dz_nodes, kappa_n=0.5):
        """Aggradation dz (m, >=0) per node; updates bed elevation, roughness."""
        self.zb = self.zb0 + dz_nodes
        self.n_rough = self.par.n0 * (1.0 + kappa_n * np.clip(dz_nodes / self.par.h_ref, 0.0, 1.0))

    def level(self, nodes=None):
        Y = self.zb + self.h
        return Y if nodes is None else Y[nodes]

    def pool_end_levels(self):
        return (self.zb + self.h)[self.pool_end_node]

    def pool_end_depth(self):
        return self.h[self.pool_end_node]

    def _gate_flow(self, a, hu, dY, mu):
        eps = 1e-3
        ae = np.minimum(a, hu)
        f = np.sqrt(2 * G) * dY / (dY * dY + eps * eps) ** 0.25
        return mu * ae * self.par.gate_W * f

    def gate_opening_for_flow(self, k, Qcmd, mu=None):
        """Opening a that passes Qcmd at the present head difference (controller side, uses mu_model)."""
        mu = mu or self.par.mu_model
        p, q = self.struct[k]
        dY = (self.zb[p] + self.h[p]) - (self.zb[q] + self.h[q])
        dY = max(dY, 0.02)
        eps = 1e-3
        f = np.sqrt(2 * G) * dY / (dY * dY + eps * eps) ** 0.25
        a = Qcmd / (mu * self.par.gate_W * f)
        return float(np.clip(a, 0.0, self.par.gate_amax))

    def tail_opening_for_flow(self, Qcmd, mu=None):
        mu = mu or self.par.mu_model
        hM = max(self.h[-1], 0.1)
        a = Qcmd / (mu * self.par.tail_W * np.sqrt(2 * G * hM))
        return float(np.clip(a, 0.0, self.par.tail_amax))

    def gate_flows(self):
        out = []
        for k, (p, q) in enumerate(self.struct):
            dY = (self.zb[p] + self.h[p]) - (self.zb[q] + self.h[q])
            out.append(float(self._gate_flow(self.gate_a[k], self.h[p], dY, self.par.mu_plant)))
        return np.array(out)

    def tail_flow(self):
        hM = max(self.h[-1], 0.0)
        ae = min(self.tail_a, hM)
        return float(self.par.mu_plant * ae * self.par.tail_W * np.sqrt(2 * G * hM))

    # ----------------------------------------------------------------- residual / jacobian
    def _eval(self, hn, Qn, h0, Q0, dt, Qin, a_g, a_t, qlat_seg):
        par = self.par
        th = par.theta
        b, m = par.b, par.m
        M = self.M
        dp = 2 * np.sqrt(1 + m * m)
        res = np.zeros(2 * M)
        J = np.zeros((2 * M, 2 * M))
        p, q = self.seg[:, 0], self.seg[:, 1]
        dx = self.dx
        nr = self.n_rough

        def props(h, Q):
            A, T, P = trapezoid(h, b, m)
            R = A / P
            Sf = nr ** 2 * Q * np.abs(Q) / (A ** 2 * R ** (4 / 3))
            dR = (T - R * dp) / P
            dSf_dQ = 2 * nr ** 2 * np.abs(Q) / (A ** 2 * R ** (4 / 3))
            dSf_dh = Sf * (-2 * T / A - (4 / 3) * dR / R)
            F = Q * Q / A
            dF_dQ = 2 * Q / A
            dF_dh = -Q * Q * T / A ** 2
            return A, T, P, R, Sf, dSf_dQ, dSf_dh, F, dF_dQ, dF_dh

        An, Tn, Pn, Rn, Sfn, dSfQn, dSfhn, Fn, dFQn, dFhn = props(hn, Qn)
        A0, T0, P0, R0, Sf0, _, _, F0, _, _ = props(h0, Q0)
        Yn = self.zb + hn
        Y0 = self.zb + h0

        # --- continuity
        Cres = ((An[p] + An[q] - A0[p] - A0[q]) / (2 * dt)
                + (th * (Qn[q] - Qn[p]) + (1 - th) * (Q0[q] - Q0[p])) / dx + qlat_seg)
        # --- momentum
        Abar_n = 0.5 * (An[p] + An[q])
        Abar_0 = 0.5 * (A0[p] + A0[q])
        dY_n = Yn[q] - Yn[p]
        dY_0 = Y0[q] - Y0[p]
        Sfb_n = 0.5 * (Sfn[p] + Sfn[q])
        Sfb_0 = 0.5 * (Sf0[p] + Sf0[q])
        Phi_n = (Fn[q] - Fn[p]) / dx + G * Abar_n * dY_n / dx + G * Abar_n * Sfb_n
        Phi_0 = (F0[q] - F0[p]) / dx + G * Abar_0 * dY_0 / dx + G * Abar_0 * Sfb_0
        Mres = (Qn[p] + Qn[q] - Q0[p] - Q0[q]) / (2 * dt) + th * Phi_n + (1 - th) * Phi_0

        ns = len(p)
        rc = 2 * np.arange(ns)            # row index of continuity eq for segment
        rm = rc + 1
        # we place segment equations in rows 1 .. 2*ns  (row 0 is upstream BC)
        rc = rc + 1
        rm = rm + 1
        res[rc] = Cres
        res[rm] = Mres
        ih_p, iQ_p, ih_q, iQ_q = 2 * p, 2 * p + 1, 2 * q, 2 * q + 1
        # continuity jacobian
        J[rc, ih_p] = Tn[p] / (2 * dt)
        J[rc, iQ_p] = -th / dx
        J[rc, ih_q] = Tn[q] / (2 * dt)
        J[rc, iQ_q] = th / dx
        # momentum jacobian
        dPhi_hp = (-dFhn[p] / dx + G * 0.5 * Tn[p] * dY_n / dx - G * Abar_n / dx
                   + G * 0.5 * Tn[p] * Sfb_n + G * Abar_n * 0.5 * dSfhn[p])
        dPhi_Qp = -dFQn[p] / dx + G * Abar_n * 0.5 * dSfQn[p]
        dPhi_hq = (dFhn[q] / dx + G * 0.5 * Tn[q] * dY_n / dx + G * Abar_n / dx
                   + G * 0.5 * Tn[q] * Sfb_n + G * Abar_n * 0.5 * dSfhn[q])
        dPhi_Qq = dFQn[q] / dx + G * Abar_n * 0.5 * dSfQn[q]
        J[rm, ih_p] = th * dPhi_hp
        J[rm, iQ_p] = 1 / (2 * dt) + th * dPhi_Qp
        J[rm, ih_q] = th * dPhi_hq
        J[rm, iQ_q] = 1 / (2 * dt) + th * dPhi_Qq

        # --- upstream BC: Q_0 = Qin
        res[0] = Qn[0] - Qin
        J[0, 1] = 1.0
        # --- structures
        r0 = 1 + 2 * ns
        eps = 1e-3
        mu = par.mu_plant
        W = par.gate_W
        for k, (pu, pd) in enumerate(self.struct):
            dY = Yn[pu] - Yn[pd]
            den = (dY * dY + eps * eps)
            f = np.sqrt(2 * G) * dY / den ** 0.25
            df = np.sqrt(2 * G) * (den ** -0.25 - 0.5 * dY * dY * den ** -1.25)
            a = a_g[k]
            if a < Yn[pu] - self.zb[pu]:        # gate in water
                ae, dae_dh = a, 0.0
            else:
                ae, dae_dh = hn[pu], 1.0
            Gq = mu * ae * W * f
            res[r0 + 2 * k] = Qn[pu] - Gq
            J[r0 + 2 * k, 2 * pu + 1] = 1.0
            J[r0 + 2 * k, 2 * pu] = -(mu * W * (dae_dh * f + ae * df))
            J[r0 + 2 * k, 2 * pd] = -(mu * ae * W * (-df))
            res[r0 + 2 * k + 1] = Qn[pd] - Qn[pu]
            J[r0 + 2 * k + 1, 2 * pd + 1] = 1.0
            J[r0 + 2 * k + 1, 2 * pu + 1] = -1.0
        # --- downstream BC (tail gate, free outflow)
        r_last = 2 * M - 1
        hM = max(hn[-1], 1e-3)
        if a_t < hM:
            ae, dae = a_t, 0.0
        else:
            ae, dae = hM, 1.0
        sq = np.sqrt(2 * G * hM)
        Qt = par.mu_plant * ae * par.tail_W * sq
        res[r_last] = Qn[-1] - Qt
        J[r_last, 2 * (M - 1) + 1] = 1.0
        J[r_last, 2 * (M - 1)] = -par.mu_plant * par.tail_W * (dae * sq + ae * G / sq)
        return res, J

    # ----------------------------------------------------------------- time stepping
    def _lateral(self, q_off, h_for_seep):
        """Lateral outflow per unit length on every segment (m3/s/m): seepage + offtakes."""
        par = self.par
        p, q = self.seg[:, 0], self.seg[:, 1]
        P = 0.5 * (trapezoid(h_for_seep[p], par.b, par.m)[2] + trapezoid(h_for_seep[q], par.b, par.m)[2])
        ql = par.seep * P
        ql = ql.copy()
        for j, qo in zip(self.off_seg, q_off):
            ql[j] += qo / self.dx
        return ql

    def offtake_flows(self, sched):
        """Actual offtake discharge (head-dependent), sched in m3/s per offtake."""
        Y = self.zb + self.h
        head = np.maximum(Y[self.off_node] - self.sill, 0.05)
        # operators keep the scheduled discharge as long as the command head is available (>= off_head_min)
        return sched * np.minimum(1.0, np.sqrt(head / self.par.off_head_min))

    def step(self, Qin, a_g, a_t, sched_off, dt=None):
        """Advance one time step.  Gate openings a_g (array), tail opening a_t, scheduled offtakes (m3/s)."""
        par = self.par
        dt = dt or par.dt
        self.gate_a = np.asarray(a_g, float).copy()
        self.tail_a = float(a_t)
        q_off = self.offtake_flows(np.asarray(sched_off, float))
        h0, Q0 = self.h.copy(), self.Q.copy()
        ok = self._advance(h0, Q0, dt, Qin, self.gate_a, self.tail_a, q_off)
        if not ok:                                    # halve the step
            h1, Q1 = h0.copy(), Q0.copy()
            self.h, self.Q = h0, Q0
            for _ in range(2):
                ok = self._advance(self.h.copy(), self.Q.copy(), dt / 2, Qin, self.gate_a, self.tail_a, q_off)
                if not ok:
                    raise RuntimeError("Newton failed at t=%.0f" % self.t)
        self.t += dt
        self.last_off_q = q_off
        return self.state()

    def _setup_band(self):
        """Row permutation that makes the Newton matrix banded (structural pattern of _eval) and gather indices."""
        M = self.M
        n = 2 * M
        p, q = self.seg[:, 0], self.seg[:, 1]
        ns = len(p)
        pat = np.zeros((n, n), bool)
        rc = 1 + 2 * np.arange(ns)
        rm = rc + 1
        for rows in (rc, rm):
            for cols in (2 * p, 2 * p + 1, 2 * q, 2 * q + 1):
                pat[rows, cols] = True
        pat[0, 1] = True
        r0 = 1 + 2 * ns
        for k, (pu, pd) in enumerate(self.struct):
            pat[r0 + 2 * k, [2 * pu + 1, 2 * pu, 2 * pd]] = True
            pat[r0 + 2 * k + 1, [2 * pd + 1, 2 * pu + 1]] = True
        pat[n - 1, [n - 1, n - 2]] = True
        key = np.array([np.nonzero(pat[i])[0].min() for i in range(n)])
        perm = np.argsort(key, kind="stable")             # permuted row i is original row perm[i]
        i_idx, j_idx = np.nonzero(pat[perm])
        kl = int(max(np.max(i_idx - j_idx), 0))
        ku = int(max(np.max(j_idx - i_idx), 0))
        R, C = [], []
        for d in range(-kl, ku + 1):
            r = np.arange(max(0, -d), min(n, n - d))
            R.append(r)
            C.append(r + d)
        R = np.concatenate(R)
        C = np.concatenate(C)
        self._band = (perm, kl, ku, perm[R], C, ku + R - C, n)

    def _solve(self, J, rhs):
        if getattr(self, "_band", None) is None:
            self._setup_band()
        perm, kl, ku, Rorig, C, tr, n = self._band
        ab = np.zeros((kl + ku + 1, n))
        ab[tr, C] = J[Rorig, C]
        return solve_banded((kl, ku), ab, rhs[perm], check_finite=False)

    def _advance(self, h0, Q0, dt, Qin, a_g, a_t, q_off):
        ql = self._lateral(q_off, h0)
        self.last_seep_flow = float(np.sum((ql - self._off_part(q_off)) * self.dx))
        hn, Qn = h0.copy(), Q0.copy()
        for it in range(25):
            res, J = self._eval(hn, Qn, h0, Q0, dt, Qin, a_g, a_t, ql)
            nr_ = np.max(np.abs(res))
            if nr_ < 1e-8:
                break
            dxv = self._solve(J, -res)
            dh = dxv[0::2]
            dQ = dxv[1::2]
            # damp to keep depth positive
            lam = 1.0
            while np.any(hn + lam * dh < 0.05) and lam > 1e-3:
                lam *= 0.5
            hn = hn + lam * dh
            Qn = Qn + lam * dQ
        else:
            return False
        self.last_iter = it
        self.h, self.Q = hn, Qn
        return True

    def _off_part(self, q_off):
        o = np.zeros(len(self.seg))
        for j, qo in zip(self.off_seg, q_off):
            o[j] += qo / self.dx
        return o

    # ----------------------------------------------------------------- diagnostics
    def volume(self):
        A, T, P = trapezoid(self.h, self.par.b, self.par.m)
        p, q = self.seg[:, 0], self.seg[:, 1]
        return float(np.sum(0.5 * (A[p] + A[q]) * self.dx))

    def state(self):
        return dict(t=self.t, Qin=self.Q[0], Qtail=self.tail_flow(), h_end=self.pool_end_depth().copy(),
                    Y_end=self.pool_end_levels().copy(), h_head=self.h[0], Y_head=self.zb[0] + self.h[0],
                    gate_q=self.gate_flows(), q_off=self.last_off_q.copy())

    def run_steady(self, Qin, a_g, a_t, sched_off, hours=48.0):
        n = int(hours * 3600 / self.par.dt)
        for _ in range(n):
            self.step(Qin, a_g, a_t, sched_off)
        return self.state()
