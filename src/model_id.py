"""Reduced (integrator) model identification for MPC: effective backwater storage area of each pool,
obtained from steady gradually-varied-flow profiles on the controller's (nominal or twin) geometry."""
import numpy as np
from scipy.integrate import solve_ivp
from canal import trapezoid, G


def pool_volume(canal, pool, Qfun, h_end, zb, n_rough):
    """Volume of pool for a steady GVF profile ending with depth h_end at the downstream end of the pool."""
    par = canal.par
    p0 = canal.pool_start_node[pool]
    p1 = canal.pool_end_node[pool]
    xs = canal.x[p0:p1 + 1]
    zbs = zb[p0:p1 + 1]
    ns = n_rough[p0:p1 + 1]
    b, m = par.b, par.m
    slope = -np.gradient(zbs, xs)

    def rhs(x, h):
        Q = Qfun(x)
        n = np.interp(x, xs, ns)
        Sb = np.interp(x, xs, slope)
        A, T, P = trapezoid(max(h[0], 0.3), b, m)
        R = A / P
        Sf = n ** 2 * Q * abs(Q) / (A ** 2 * R ** (4 / 3))
        Fr2 = min(Q * Q * T / (G * A ** 3), 0.8)
        return [(Sb - Sf) / (1 - Fr2)]
    sol = solve_ivp(rhs, [xs[-1], xs[0]], [h_end], t_eval=xs[::-1], rtol=1e-6, atol=1e-8)
    hh = np.maximum(sol.y[0][::-1], 0.3)
    A = trapezoid(hh, b, m)[0]
    return float(np.trapezoid(A, xs)), hh


def pool_areas(canal, Qprofile_nodes, zb, n_rough, h_end=None, delta=0.05):
    """Effective surface (backwater) area for every pool (m2) around the depth h_end (default h_ref)."""
    par = canal.par
    h_end = par.h_ref if h_end is None else h_end
    out = []
    for j in range(canal.n_struct + 1):
        p0 = canal.pool_start_node[j]
        p1 = canal.pool_end_node[j]
        xs = canal.x[p0:p1 + 1]
        Qs = Qprofile_nodes[p0:p1 + 1]
        Qf = lambda x, xs=xs, Qs=Qs: np.interp(x, xs, Qs)
        v1, _ = pool_volume(canal, j, Qf, h_end, zb, n_rough)
        v2, _ = pool_volume(canal, j, Qf, h_end + delta, zb, n_rough)
        out.append((v2 - v1) / delta)
    return np.array(out)
