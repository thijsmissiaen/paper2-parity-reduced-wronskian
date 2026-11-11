#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper2_parity_reduced_wronskian.py
===============================================================================
Verification Suite for Paper II:
"Parity-Reduced Transverse Wronskians, Jensen-Laguerre Moment Identities,
and Division-Free Interval Enclosures for Riemann's Xi-Function"

Author: Thijs Missiaen (ORCID: 0009-0007-3678-0810)
Date: October 2026

This self-contained script verifies:
  1. Canonical Wigner-hyperbolic normalization W(d,t) = 16 int_0^inf y sinh(2dy) I(y,t) dy,
     W_tilde(d,t) = 32 int_0^inf y^2 sinc_hyp(2dy) I(y,t) dy, and the Jensen-Csordas-Varga
     moment identities c_{2m-1}(t) = m * L_m(Xi; t) = (2^{2m+3}/(2m-1)!) int_0^inf y^{2m} I(y,t) dy
     at 50-decimal precision, plus an explicit off-critical zero obstruction model F_off(z).
  2. Directed-rounding interval enclosures (mpmath.iv) of c_1, c_3, c_5 > 0 at gamma_1..gamma_10.
  3. Dual mpmath.iv evaluator cross-check (Euler-Maclaurin vs. Jacobi-theta), 80 centered 2D
     Interval-MVT collar boxes (around gamma_1, tau_1, gamma_2, tau_2, gamma_3 using true roots
     tau_1 = 15.58570859, tau_2 = 22.09797728 of Xi'(tau_k) = 0 and rigorous holomorphic
     Cauchy-Taylor bounds without assuming harmonicity of non-harmonic components), and
     5 integer Argument-Principle rectangles covering [10^-6, 1.20] x [0, 55].
  4. IEEE-754 float64 cancellation-amplification audit quantifying Higham's |x|^T |y| bound
     (|Delta W_hat| ~ u_mach * exp(+pi*t/4), reaching 4.51% relative error at t = 55).
  5. Generation of the two publication figures:
       paper/figures/fig1_parity_reduced_wronskian_and_dipoles.pdf (.png)
       paper/figures/fig2_interval_mvt_and_argument_principle.pdf (.png)
===============================================================================
"""

import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
import mpmath
from mpmath import iv, mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

iv.dps = 35
mp.dps = 50

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0


def print_banner(title: str) -> None:
    print("=" * 92)
    print(f" {title}")
    print("=" * 92)


# ==============================================================================
# SECTION 1: DIRECTED-ROUNDING INTERVAL EVALUATORS (mpmath.iv)
# ==============================================================================

NU_IV = 140
UMAX_IV = iv.mpf("1.55")
H_IV = UMAX_IV / iv.mpf(NU_IV)
PI_IV = iv.pi
PI2_IV = PI_IV * PI_IV

U_LIST_IV = [iv.mpf(j) * H_IV for j in range(NU_IV + 1)]
U2_LIST_IV = [iv.mpf(2) * u for u in U_LIST_IV]
U4_LIST_IV = [u2 * u2 for u2 in U2_LIST_IV]
U6_LIST_IV = [u4 * u2 for u4, u2 in zip(U4_LIST_IV, U2_LIST_IV)]
U8_LIST_IV = [u4 * u4 for u4 in U4_LIST_IV]

# Analytical truncation bounds on [0, 1.20] x [0, 55] (Appendix A, Lemma A.1):
# 1) Jacobi series tail (m > 12): eps_ser <= 1e-220
# 2) Quadrature domain tail (u > 1.55): eps_tail <= 1e-550
# 3) Composite trapezoidal discretization error on |Im(u)| <= 0.35 < pi/8: eps_quad <= 1e-55
EPS_SER_BOUND = iv.mpf("1e-220")
EPS_TAIL_BOUND = iv.mpf("1e-550")
EPS_QUAD_BOUND = iv.mpf("1e-55")
EPS_TOTAL_TRUNC = EPS_SER_BOUND + EPS_TAIL_BOUND + EPS_QUAD_BOUND
TRUNC_IV = iv.mpf([-EPS_TOTAL_TRUNC.b, EPS_TOTAL_TRUNC.b])

PHI_W_IV = []
for _j, _u in enumerate(U_LIST_IV):
    _e4 = iv.exp(iv.mpf(4) * _u)
    _e5 = iv.exp(iv.mpf(5) * _u)
    _e9 = iv.exp(iv.mpf(9) * _u)
    _s = iv.mpf(0)
    for _m in range(1, 13):
        _m2 = iv.mpf(_m * _m)
        _m4 = _m2 * _m2
        _s += (iv.mpf(2) * PI2_IV * _m4 * _e9 - iv.mpf(3) * PI_IV * _m2 * _e5) * iv.exp(-PI_IV * _m2 * _e4)
    _s = _s + iv.mpf([0, EPS_SER_BOUND.b])
    _w = iv.mpf("0.5") if (_j == 0 or _j == NU_IV) else iv.mpf("1.0")
    PHI_W_IV.append(iv.mpf(8) * H_IV * _w * _s)

BERN_IV = {}
for _k in range(1, 16):
    _p, _q = mpmath.bernfrac(2 * _k)
    BERN_IV[_k] = iv.mpf(int(_p)) / iv.mpf(int(_q))


def iv_zeta_euler_maclaurin(s, N: int = 28, M: int = 12):
    """
    Directed-rounding interval evaluator for zeta(s) via Euler-Maclaurin
    summation with explicit remainder enclosure [-eps_EM, +eps_EM].
    """
    s = iv.convert(s)
    total = iv.mpc(0)
    for n in range(1, N):
        total += iv.exp(-s * iv.log(iv.mpf(n)))
    N_iv = iv.mpf(N)
    N_neg_s = iv.exp(-s * iv.log(N_iv))
    total += (N_neg_s * N_iv) / (s - iv.mpf(1)) + iv.mpf("0.5") * N_neg_s
    poch = s
    inv_N2 = iv.mpf(1) / (N_iv * N_iv)
    N_pow = N_neg_s / N_iv
    for k in range(1, M + 1):
        B2k = BERN_IV[k]
        fact = iv.fac(2 * k)
        term = (B2k / fact) * poch * N_pow
        total += term
        poch = poch * (s + iv.mpf(2 * k - 1)) * (s + iv.mpf(2 * k))
        N_pow = N_pow * inv_N2
    k_rem = M + 1
    B2k_rem = BERN_IV[k_rem]
    fact_rem = iv.fac(2 * k_rem)
    next_term = (B2k_rem / fact_rem) * poch * N_pow
    factor = abs(s + iv.mpf(2 * M + 1)) / (s.real + iv.mpf(2 * M + 1))
    rem_bound = (factor * abs(next_term)).b
    err_iv = iv.mpf([-rem_bound, rem_bound])
    return total + iv.mpc(err_iv, err_iv)


def iv_xi_euler_maclaurin(d_val, t_val):
    """
    Evaluator A: Directed-rounding interval evaluation of Xi(t - i*d) = xi(1/2 + d + i*t)
    via Euler-Maclaurin summation and the completed Gamma prefactor.
    """
    d_iv = iv.convert(d_val)
    t_iv = iv.convert(t_val)
    s = iv.mpc(iv.mpf("0.5") + d_iv, t_iv)
    pref = iv.mpf("0.5") * s * (s - iv.mpf(1)) * iv.exp(-iv.mpf("0.5") * s * iv.log(PI_IV))
    gam = iv.gamma(s * iv.mpf("0.5"))
    zet = iv_zeta_euler_maclaurin(s, N=28, M=12)
    return pref * gam * zet


def iv_ch_sinch_etach_scalar(x_iv):
    """
    Returns (cosh(x), sinh(x)/x, (x*cosh(x)-sinh(x))/x^3) as rigorous intervals
    for x_iv >= 0 without 0/0 division near x = 0.
    """
    if x_iv == 0:
        return iv.mpf(1), iv.mpf(1), iv.mpf(1) / iv.mpf(3)
    if x_iv.b <= iv.mpf("0.10"):
        x2 = x_iv * x_iv
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        tail_ch = (x12 / iv.fac(12)) * iv.mpf("1.01")
        tail_sh = (x12 / iv.fac(13)) * iv.mpf("1.01")
        tail_et = (iv.mpf(14) * x12 / iv.fac(15)) * iv.mpf("1.01")
        ch = (iv.mpf(1) + x2 / iv.mpf(2) + x4 / iv.mpf(24) + x6 / iv.mpf(720)
              + x8 / iv.mpf(40320) + x10 / iv.mpf(3628800) + iv.mpf([0, tail_ch.b]))
        sinch = (iv.mpf(1) + x2 / iv.mpf(6) + x4 / iv.mpf(120) + x6 / iv.mpf(5040)
                 + x8 / iv.mpf(362880) + x10 / iv.mpf(39916800) + iv.mpf([0, tail_sh.b]))
        etach = (iv.mpf(1) / iv.mpf(3) + x2 / iv.mpf(30) + x4 / iv.mpf(840)
                 + x6 / iv.mpf(45360) + x8 / iv.mpf(3991680) + x10 / iv.mpf(518918400)
                 + iv.mpf([0, tail_et.b]))
        return ch, sinch, etach
    ep = iv.exp(x_iv)
    em = iv.mpf(1) / ep
    ch = (ep + em) * iv.mpf("0.5")
    sh = (ep - em) * iv.mpf("0.5")
    return ch, sh / x_iv, (x_iv * ch - sh) / (x_iv * x_iv * x_iv)


def iv_ch_sinch_etach_box(x_iv):
    """
    Monotone endpoint enclosure for cosh(x), sinc_hyp(x), and eta_hyp(x) on x_iv = [a, b] >= 0.
    """
    cha, sa, ea = iv_ch_sinch_etach_scalar(iv.mpf(x_iv.a))
    if x_iv.a == x_iv.b:
        return cha, sa, ea
    chb, sb, eb = iv_ch_sinch_etach_scalar(iv.mpf(x_iv.b))
    return iv.mpf([cha.a, chb.b]), iv.mpf([sa.a, sb.b]), iv.mpf([ea.a, eb.b])


def iv_eval_jacobi_full(d_val, t_val):
    """
    Evaluator B: Division-free Jacobi-theta interval evaluation of Xi(t - i*d),
    the parity-reduced Wronskian W_tilde(d,t) = W(d,t)/d, and its partial derivatives.
    """
    d_iv = iv.convert(d_val)
    t_iv = iv.convert(t_val)

    R = iv.mpf(0)
    Rt = iv.mpf(0)
    Rtt = iv.mpf(0)
    Itil = iv.mpf(0)
    Itilt = iv.mpf(0)
    Itiltt = iv.mpf(0)
    Ihat_d = iv.mpf(0)
    Ihat_td = iv.mpf(0)

    for j in range(NU_IV + 1):
        u2 = U2_LIST_IV[j]
        u4 = U4_LIST_IV[j]
        u6 = U6_LIST_IV[j]
        u8 = U8_LIST_IV[j]
        w = PHI_W_IV[j]

        arg_d = u2 * d_iv
        arg_t = u2 * t_iv
        ch, sinch, etach = iv_ch_sinch_etach_box(arg_d)
        cs = iv.cos(arg_t)
        sn = iv.sin(arg_t)

        R += w * ch * cs
        Rt -= w * u2 * ch * sn
        Rtt -= w * u4 * ch * cs
        Itil += w * u2 * sinch * sn
        Itilt += w * u4 * sinch * cs
        Itiltt -= w * u6 * sinch * sn
        Ihat_d += w * u6 * etach * sn
        Ihat_td += w * u8 * etach * cs

    R += TRUNC_IV
    Rt += TRUNC_IV
    Rtt += TRUNC_IV
    Itil += TRUNC_IV
    Itilt += TRUNC_IV
    Itiltt += TRUNC_IV
    Ihat_d += TRUNC_IV
    Ihat_td += TRUNC_IV

    I_val = d_iv * Itil
    It_val = d_iv * Itilt
    Itt_val = d_iv * Itiltt

    Xi_iv = iv.mpc(R, I_val)
    Xip_iv = iv.mpc(Rt, It_val)
    Xipp_iv = iv.mpc(Rtt, Itt_val)

    W_tilde = R * Itilt - Itil * Rt
    dW_tilde_dt = R * Itiltt - Itil * Rtt
    dW_tilde_dd = d_iv * (Itilt * Itilt - Itil * Itiltt + R * Ihat_td - Ihat_d * Rt)

    return {
        "Xi": Xi_iv,
        "Xip": Xip_iv,
        "Xipp": Xipp_iv,
        "R": R,
        "I": I_val,
        "Rt": Rt,
        "It": It_val,
        "W_tilde": W_tilde,
        "dW_tilde_dd": dW_tilde_dd,
        "dW_tilde_dt": dW_tilde_dt,
    }


def iv_eval_d0_derivs_up_to_6(t_val):
    """
    Evaluates Xi^(0)(t) .. Xi^(6)(t) on the critical line d = 0 in directed-rounding
    interval arithmetic (including TRUNC_IV).
    """
    t_iv = iv.convert(t_val)
    D = [iv.mpf(0) for _ in range(7)]
    for j in range(NU_IV + 1):
        u2 = U2_LIST_IV[j]
        w = PHI_W_IV[j]
        arg_t = u2 * t_iv
        cs = iv.cos(arg_t)
        sn = iv.sin(arg_t)
        p0 = w
        p1 = p0 * u2
        p2 = p1 * u2
        p3 = p2 * u2
        p4 = p3 * u2
        p5 = p4 * u2
        p6 = p5 * u2
        D[0] += p0 * cs
        D[1] -= p1 * sn
        D[2] -= p2 * cs
        D[3] += p3 * sn
        D[4] += p4 * cs
        D[5] -= p5 * sn
        D[6] -= p6 * cs
    for m in range(7):
        D[m] += TRUNC_IV
    return D


def iv_zetah_scalar(x_iv):
    """
    Evaluates zeta_hyp(x) = eta_hyp'(x) / x = ((x^2 + 3)*sinh(x) - 3*x*cosh(x)) / x^5
      = sum_{k=0}^inf ((2k+2)(2k+4)/(2k+5)!) x^{2k} >= 1/15
    in directed-rounding interval arithmetic without 0/0 cancellation near x = 0.
    """
    if x_iv == 0:
        return iv.mpf(1) / iv.mpf(15)
    if x_iv.b <= iv.mpf("0.25"):
        x2 = x_iv * x_iv
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        tail_zh = (iv.mpf(224) * x12 / iv.fac(17)) * iv.mpf("1.01")
        return (iv.mpf(1) / iv.mpf(15)
                + x2 / iv.mpf(210)
                + x4 / iv.mpf(7560)
                + x6 / iv.mpf(498960)
                + x8 / iv.mpf(51891840)
                + x10 / iv.mpf(7783776000)
                + iv.mpf([0, tail_zh.b]))
    ep = iv.exp(x_iv)
    em = iv.mpf(1) / ep
    ch = (ep + em) * iv.mpf("0.5")
    sh = (ep - em) * iv.mpf("0.5")
    x2 = x_iv * x_iv
    x5 = x2 * x2 * x_iv
    return ((x2 + iv.mpf(3)) * sh - iv.mpf(3) * x_iv * ch) / x5


def _sup_holomorphic_xi_derivs_on_strip(db_iv, t0_iv, dt_iv, max_m: int = 9, K_tay: int = 12):
    """
    Rigorously bounds M_m := sup_{delta in [0, db_iv], t in [t0_iv - dt_iv, t0_iv + dt_iv]} |Xi^{(m)}(t - i*delta)|
    for 0 <= m <= max_m using ONLY the true holomorphicity of Xi^{(m)}(z) in z = t - i*delta!
    Partitions delta in [0, db_iv] into sub-strips of half-width <= 0.09 and applies the exact
    complex Cauchy-Taylor remainder bound around each sub-strip center z_c = t0_iv - i*delta_c.
    """
    n_strips = max(1, int(math.ceil(float(db_iv.b) / 0.16)))
    half_del = db_iv / iv.mpf(2 * n_strips)
    r_disk = iv.sqrt(half_del * half_del + dt_iv * dt_iv)

    # Non-oscillatory tail bound sup_{delta in [0, db], t} |Xi^{(m + K_tay)}(t - i*delta)|
    total_ord = max_m + K_tay
    tail_sup = [iv.mpf(0) for _ in range(total_ord + 1)]
    for j in range(NU_IV + 1):
        u2 = U2_LIST_IV[j]
        w = PHI_W_IV[j]
        ch_ub, _, _ = iv_ch_sinch_etach_scalar(u2 * db_iv)
        w_ch = w * ch_ub
        p = iv.mpf(1)
        for m in range(total_ord + 1):
            tail_sup[m] += w_ch * p
            p = p * u2
    for m in range(total_ord + 1):
        tail_sup[m] += EPS_TOTAL_TRUNC

    facs = [iv.fac(k) for k in range(K_tay + 1)]
    r_pows = [iv.mpf(1)]
    for _ in range(K_tay):
        r_pows.append(r_pows[-1] * r_disk)

    M_bounds = [iv.mpf(0) for _ in range(max_m + 1)]
    for s_idx in range(n_strips):
        del_c = (iv.mpf(2 * s_idx + 1) * half_del)
        R_m = [iv.mpf(0) for _ in range(total_ord)]
        I_m = [iv.mpf(0) for _ in range(total_ord)]
        for j in range(NU_IV + 1):
            u2 = U2_LIST_IV[j]
            w = PHI_W_IV[j]
            arg_d = u2 * del_c
            ep = iv.exp(arg_d)
            em = iv.mpf(1) / ep
            ch_v = (ep + em) * iv.mpf("0.5")
            sh_v = (ep - em) * iv.mpf("0.5")
            cs_v = iv.cos(u2 * t0_iv)
            sn_v = iv.sin(u2 * t0_iv)
            w_ch = w * ch_v
            w_sh = w * sh_v
            p = iv.mpf(1)
            for m in range(total_ord):
                mod4 = m % 4
                if mod4 == 0:
                    d_cs, d_sn = cs_v, sn_v
                elif mod4 == 1:
                    d_cs, d_sn = -sn_v, cs_v
                elif mod4 == 2:
                    d_cs, d_sn = -cs_v, -sn_v
                else:
                    d_cs, d_sn = sn_v, -cs_v
                R_m[m] += w_ch * p * d_cs
                I_m[m] += w_sh * p * d_sn
                p = p * u2
        mod_center = []
        for m in range(total_ord):
            Rm = R_m[m] + TRUNC_IV
            Im = I_m[m] + TRUNC_IV
            mod_center.append(iv.sqrt(Rm * Rm + Im * Im))

        for m in range(max_m + 1):
            sup_strip = iv.mpf(0)
            for n in range(K_tay):
                sup_strip += (r_pows[n] / facs[n]) * mod_center[m + n]
            sup_strip += (r_pows[K_tay] / facs[K_tay]) * tail_sup[m + K_tay]
            if sup_strip.b > M_bounds[m].b:
                M_bounds[m] = iv.mpf(sup_strip.b)
    return M_bounds


def certify_2d_box_mvt_iv(d_a, d_b, t_a, t_b):
    """
    2D Interval Mean-Value Theorem enclosure on B = [d_a, d_b] x [t_a, t_b] WITHOUT assuming
    harmonicity of the non-harmonic components Itil or Ihat!
    Uses the exact positive-weight integral representations:
      sinc_hyp(w) = int_0^1 cosh(s*w) ds,
      eta_hyp(w)  = int_0^1 int_0^1 s^2 cosh(s*tau*w) dtau ds,
    which express every partial derivative d_d^p d_t^q F(d,t) for F in {R, Rt, Rtt, Itil, Itilt,
    Itiltt, Ihat_d, Ihat_td} directly as weighted integrals of derivatives of the holomorphic
    function Xi(t - i*delta) over delta in [0, d_b]:
      |d_d^p d_t^q R^{(k)}(d,t)|    <= M_{p+q+k},
      |d_d^p d_t^q Itil^{(k)}(d,t)| <= (1/(p+1)) * M_{p+q+k+1},
      |d_d^p d_t^q Ihat^{(k)}(d,t)| <= (1/((p+3)*(p+1))) * M_{p+q+k+3},
    where M_m = sup_{delta in [0, d_b], t in [t_a, t_b]} |Xi^{(m)}(t - i*delta)| is enclosed
    rigorously via complex Cauchy-Taylor expansion of the holomorphic function Xi^{(m)}(z).
    """
    da_iv = iv.convert(d_a)
    db_iv = iv.convert(d_b)
    ta_iv = iv.convert(t_a)
    tb_iv = iv.convert(t_b)
    d_iv = iv.mpf([da_iv.a, db_iv.b])
    d0 = (da_iv + db_iv) * iv.mpf("0.5")
    t0 = (ta_iv + tb_iv) * iv.mpf("0.5")
    dd = (db_iv - da_iv) * iv.mpf("0.5")
    dt = (tb_iv - ta_iv) * iv.mpf("0.5")
    sym_dd = iv.mpf([-dd.b, dd.b])
    sym_dt = iv.mpf([-dt.b, dt.b])

    # Evaluate exact center values and 1st partial derivatives at (d0, t0)
    R_c = [iv.mpf(0) for _ in range(4)]
    Itil_c = [iv.mpf(0) for _ in range(4)]
    Ihat_c = [iv.mpf(0) for _ in range(3)]
    Icheck_c = [iv.mpf(0) for _ in range(2)]

    for j in range(NU_IV + 1):
        u2 = U2_LIST_IV[j]
        w = PHI_W_IV[j]
        arg_d0 = u2 * d0
        ch0, sinch0, etach0 = iv_ch_sinch_etach_scalar(arg_d0)
        zetah0 = iv_zetah_scalar(arg_d0)
        cs0 = iv.cos(u2 * t0)
        sn0 = iv.sin(u2 * t0)

        u_pows = [iv.mpf(1)]
        for _ in range(7):
            u_pows.append(u_pows[-1] * u2)

        for m in range(4):
            mod4 = m % 4
            if mod4 == 0:
                d_cs, d_sn = cs0, sn0
            elif mod4 == 1:
                d_cs, d_sn = -sn0, cs0
            elif mod4 == 2:
                d_cs, d_sn = -cs0, -sn0
            else:
                d_cs, d_sn = sn0, -cs0
            R_c[m] += w * u_pows[m] * ch0 * d_cs
            Itil_c[m] += w * u_pows[m + 1] * sinch0 * d_sn
            if m < 3:
                Ihat_c[m] += w * u_pows[m + 3] * etach0 * d_sn
            if m < 2:
                Icheck_c[m] += w * u_pows[m + 5] * zetah0 * d_sn

    for m in range(4):
        R_c[m] += TRUNC_IV
        Itil_c[m] += TRUNC_IV
    for m in range(3):
        Ihat_c[m] += TRUNC_IV
    for m in range(2):
        Icheck_c[m] += TRUNC_IV

    R0, Rt0, Rtt0 = R_c[0], R_c[1], R_c[2]
    Itil0, Itilt0, Itiltt0 = Itil_c[0], Itil_c[1], Itil_c[2]
    Ihat_d0, Ihat_td0 = Ihat_c[0], Ihat_c[1]
    Icheck_dd0, Icheck_tdd0 = Icheck_c[0], Icheck_c[1]

    # Rigorous holomorphic derivative bounds M_2 .. M_8 on [0, d_b] x [t_a, t_b]
    M = _sup_holomorphic_xi_derivs_on_strip(db_iv, t0, dt, max_m=8, K_tay=12)

    def _rem2(M_dd, M_dt, M_tt):
        val = iv.mpf("0.5") * (M_dd * dd * dd + iv.mpf(2) * M_dt * dd * dt + M_tt * dt * dt)
        return iv.mpf([-val.b, val.b])

    # R^{(k)} (k = 0, 1, 2): |d_d^p d_t^q R^{(k)}| <= M_{p + q + k}
    eps2_R = _rem2(M[2], M[2], M[2])
    eps2_Rt = _rem2(M[3], M[3], M[3])
    eps2_Rtt = _rem2(M[4], M[4], M[4])

    # Itil^{(k)} (k = 0, 1, 2): |d_d^p d_t^q Itil^{(k)}| <= (1/(p+1)) * M_{p + q + k + 1}
    one_third = iv.mpf(1) / iv.mpf(3)
    one_half = iv.mpf("0.5")
    eps2_Itil = _rem2(one_third * M[3], one_half * M[3], M[3])
    eps2_Itilt = _rem2(one_third * M[4], one_half * M[4], M[4])
    eps2_Itiltt = _rem2(one_third * M[5], one_half * M[5], M[5])

    # Ihat^{(k)} (k = 0, 1): |d_d^p d_t^q Ihat^{(k)}| <= (1/((p+3)(p+1))) * M_{p + q + k + 3}
    one_fifteenth = iv.mpf(1) / iv.mpf(15)
    one_eighth = iv.mpf("0.125")
    eps2_Ihat_d = _rem2(one_fifteenth * M[5], one_eighth * M[5], one_third * M[5])
    eps2_Ihat_td = _rem2(one_fifteenth * M[6], one_eighth * M[6], one_third * M[6])

    RB = R0 + (d0 * Itilt0) * sym_dd + Rt0 * sym_dt + eps2_R
    RtB = Rt0 + (d0 * Itiltt0) * sym_dd + Rtt0 * sym_dt + eps2_Rt
    RttB = Rtt0 + (d0 * Itil_c[3]) * sym_dd + R_c[3] * sym_dt + eps2_Rtt

    ItilB = Itil0 + (d0 * Ihat_d0) * sym_dd + Itilt0 * sym_dt + eps2_Itil
    ItiltB = Itilt0 + (d0 * Ihat_td0) * sym_dd + Itiltt0 * sym_dt + eps2_Itilt
    ItilttB = Itiltt0 + (d0 * Ihat_c[2]) * sym_dd + Itil_c[3] * sym_dt + eps2_Itiltt

    Ihat_dB = Ihat_d0 + (d0 * Icheck_dd0) * sym_dd + Ihat_td0 * sym_dt + eps2_Ihat_d
    Ihat_tdB = Ihat_td0 + (d0 * Icheck_tdd0) * sym_dd + Ihat_c[2] * sym_dt + eps2_Ihat_td

    W0 = R0 * Itilt0 - Itil0 * Rt0
    dW_dtB = RB * ItilttB - ItilB * RttB
    dW_ddB = d_iv * (ItiltB * ItiltB - ItilB * ItilttB + RB * Ihat_tdB - Ihat_dB * RtB)
    W_box = W0 + dW_ddB * sym_dd + dW_dtB * sym_dt
    return W0, W_box, dW_ddB, dW_dtB


def adaptive_certify_collar_iv(d_a, d_b, t_a, t_b, depth: int = 0, max_depth: int = 3):
    da_iv = iv.convert(d_a)
    db_iv = iv.convert(d_b)
    ta_iv = iv.convert(t_a)
    tb_iv = iv.convert(t_b)
    W0, W_box, dW_ddB, dW_dtB = certify_2d_box_mvt_iv(da_iv, db_iv, ta_iv, tb_iv)
    if W_box.a > 0 or depth >= max_depth:
        return [(float(da_iv.mid), float(db_iv.mid), float(ta_iv.mid), float(tb_iv.mid), W0, W_box, depth)]
    var_d = float((abs(dW_ddB) * (db_iv - da_iv)).b)
    var_t = float((abs(dW_dtB) * (tb_iv - ta_iv)).b)
    if var_t >= var_d:
        t_m = (ta_iv + tb_iv) * iv.mpf("0.5")
        return (adaptive_certify_collar_iv(da_iv, db_iv, ta_iv, t_m, depth + 1, max_depth) +
                adaptive_certify_collar_iv(da_iv, db_iv, t_m, tb_iv, depth + 1, max_depth))
    else:
        d_m = (da_iv + db_iv) * iv.mpf("0.5")
        return (adaptive_certify_collar_iv(da_iv, d_m, ta_iv, tb_iv, depth + 1, max_depth) +
                adaptive_certify_collar_iv(d_m, db_iv, ta_iv, tb_iv, depth + 1, max_depth))


def iv_atan(x_iv):
    """
    Monotone interval enclosure of arctan(x) in mpmath.iv via 4-fold Euler halving.
    """
    def _atan_scalar(v):
        if v == 0:
            return iv.mpf(0)
        sign = iv.mpf(1)
        if v < 0:
            v = -v
            sign = iv.mpf(-1)
        for _ in range(4):
            v = v / (iv.mpf(1) + iv.sqrt(iv.mpf(1) + v * v))
        v2 = v * v
        total = iv.mpf(0)
        pow_v = v
        for k in range(12):
            term = pow_v / iv.mpf(2 * k + 1)
            if k % 2 == 0:
                total += term
            else:
                total -= term
            pow_v = pow_v * v2
        rem = (pow_v / iv.mpf(25)).b
        total += iv.mpf([-rem, rem])
        return sign * iv.mpf(16) * total

    a_val = _atan_scalar(iv.mpf(x_iv.a))
    if x_iv.a == x_iv.b:
        return a_val
    b_val = _atan_scalar(iv.mpf(x_iv.b))
    return iv.mpf([a_val.a, b_val.b])


def certify_cauchy_winding_box_iv(d_min, d_max, t_min, t_max, n_pts_t: int = 80, n_pts_d: int = 24):
    """
    Integer Argument Principle enclosure N_zeros(Omega) in Z on Omega = [d_min, d_max] x [t_min, t_max]
    with continuum segment certification along every boundary segment [z_j, z_{j+1}].
    """
    dm = iv.mpf(str(d_min))
    dM = iv.mpf(str(d_max))
    tm = iv.mpf(str(t_min))
    tM = iv.mpf(str(t_max))

    zeros_11_ref = [
        iv.mpf("14.1347251417346937904572519835625"),
        iv.mpf("21.0220396387715549926284795938969"),
        iv.mpf("25.0108575801456887632137909925628"),
        iv.mpf("30.4248761258595132103118975305840"),
        iv.mpf("32.9350615877391896906623689640749"),
        iv.mpf("37.5861781588256712572177634807053"),
        iv.mpf("40.9187190121474951873981269146332"),
        iv.mpf("43.3270732809149995194961221654068"),
        iv.mpf("48.0051508811671597279424727494275"),
        iv.mpf("49.7738324776723021819167846785638"),
        iv.mpf("52.9703214777144606441472966088808"),
    ]
    pts = []
    for k in range(n_pts_t):
        frac = iv.mpf(k) / iv.mpf(n_pts_t)
        pts.append((dM, tm + frac * (tM - tm)))
    for k in range(n_pts_d):
        frac = iv.mpf(k) / iv.mpf(n_pts_d)
        pts.append((dM - frac * (dM - dm), tM))

    t_top_nodes = [tM - (iv.mpf(k) / iv.mpf(n_pts_t)) * (tM - tm) for k in range(n_pts_t)]
    geom_offsets = [0.0, 0.3, -0.3]
    val_g = 0.6
    while val_g * float(dm.mid) < 0.12:
        geom_offsets.append(val_g)
        geom_offsets.append(-val_g)
        val_g *= 1.32
    for gk_v in zeros_11_ref:
        if tm.a < gk_v.mid < tM.b:
            for c_mul in geom_offsets:
                t_cand = gk_v + iv.mpf(str(c_mul)) * dm
                if tm.a < t_cand.mid < tM.b:
                    t_top_nodes.append(t_cand)
    t_top_nodes.sort(key=lambda x: float(x.mid), reverse=True)
    for t_node in t_top_nodes:
        pts.append((dm, t_node))

    for k in range(n_pts_d):
        frac = iv.mpf(k) / iv.mpf(n_pts_d)
        pts.append((dm + frac * (dM - dm), tm))
    pts.append(pts[0])

    K_seg = 10
    ch_cache = {}
    sh_cache = {}
    tail_K_cache = {}
    for d_fixed in (dm, dM):
        ch_arr = []
        sh_arr = []
        tail_s = iv.mpf(0)
        for j in range(NU_IV + 1):
            u2 = U2_LIST_IV[j]
            arg_d = u2 * d_fixed
            ep = iv.exp(arg_d)
            em = iv.mpf(1) / ep
            ch_v = (ep + em) * iv.mpf("0.5")
            sh_v = (ep - em) * iv.mpf("0.5")
            ch_arr.append(ch_v)
            sh_arr.append(sh_v)
            tail_s += PHI_W_IV[j] * (u2 ** K_seg) * ch_v
        ch_cache[str(d_fixed)] = ch_arr
        sh_cache[str(d_fixed)] = sh_arr
        tail_K_cache[str(d_fixed)] = tail_s + iv.mpf("1e-50")

    cs_cache = {}
    sn_cache = {}
    for t_fixed in (tm, tM):
        cs_arr = []
        sn_arr = []
        for j in range(NU_IV + 1):
            arg_t = U2_LIST_IV[j] * t_fixed
            cs_arr.append(iv.cos(arg_t))
            sn_arr.append(iv.sin(arg_t))
        cs_cache[str(t_fixed)] = cs_arr
        sn_cache[str(t_fixed)] = sn_arr

    u2_pows_table = []
    for j in range(NU_IV + 1):
        u2 = U2_LIST_IV[j]
        p_row = [iv.mpf(1)]
        for _ in range(K_seg):
            p_row.append(p_row[-1] * u2)
        u2_pows_table.append(p_row)

    node_eval_cache = {}

    def _eval_vertex(d_p, t_p):
        key = (float(d_p.mid), float(t_p.mid))
        if key in node_eval_cache:
            return node_eval_cache[key]
        R_m = [iv.mpf(0) for _ in range(K_seg)]
        I_m = [iv.mpf(0) for _ in range(K_seg)]
        d_key = str(d_p)
        t_key = str(t_p)
        if d_key in ch_cache:
            ch_arr = ch_cache[d_key]
            sh_arr = sh_cache[d_key]
            for j in range(NU_IV + 1):
                arg_t = U2_LIST_IV[j] * t_p
                cs_v = iv.cos(arg_t)
                sn_v = iv.sin(arg_t)
                w = PHI_W_IV[j]
                w_ch = w * ch_arr[j]
                w_sh = w * sh_arr[j]
                p_row = u2_pows_table[j]
                for m in range(K_seg):
                    mod4 = m % 4
                    if mod4 == 0:
                        d_cs, d_sn = cs_v, sn_v
                    elif mod4 == 1:
                        d_cs, d_sn = -sn_v, cs_v
                    elif mod4 == 2:
                        d_cs, d_sn = -cs_v, -sn_v
                    else:
                        d_cs, d_sn = sn_v, -cs_v
                    R_m[m] += w_ch * p_row[m] * d_cs
                    I_m[m] += w_sh * p_row[m] * d_sn
        else:
            cs_arr = cs_cache[t_key]
            sn_arr = sn_cache[t_key]
            for j in range(NU_IV + 1):
                arg_d = U2_LIST_IV[j] * d_p
                ep = iv.exp(arg_d)
                em = iv.mpf(1) / ep
                ch_v = (ep + em) * iv.mpf("0.5")
                sh_v = (ep - em) * iv.mpf("0.5")
                w = PHI_W_IV[j]
                w_ch = w * ch_v
                w_sh = w * sh_v
                cs_v = cs_arr[j]
                sn_v = sn_arr[j]
                p_row = u2_pows_table[j]
                for m in range(K_seg):
                    mod4 = m % 4
                    if mod4 == 0:
                        d_cs, d_sn = cs_v, sn_v
                    elif mod4 == 1:
                        d_cs, d_sn = -sn_v, cs_v
                    elif mod4 == 2:
                        d_cs, d_sn = -cs_v, -sn_v
                    else:
                        d_cs, d_sn = sn_v, -cs_v
                    R_m[m] += w_ch * p_row[m] * d_cs
                    I_m[m] += w_sh * p_row[m] * d_sn

        for m in range(K_seg):
            R_m[m] += TRUNC_IV
            I_m[m] += TRUNC_IV
        mod_abs = iv.sqrt(R_m[0] * R_m[0] + I_m[0] * I_m[0])
        deriv_abs = [iv.sqrt(R_m[m] * R_m[m] + I_m[m] * I_m[m]) for m in range(1, K_seg)]
        res = (R_m[0], I_m[0], mod_abs, deriv_abs)
        node_eval_cache[key] = res
        return res

    fac_list = [iv.fac(m) for m in range(K_seg + 1)]
    tail_K_dM = tail_K_cache[str(dM)]

    def _sup_Xip_on_segment(p1, p2, v1, v2):
        d1, t1 = p1
        d2, t2 = p2
        h_seg = iv.sqrt((d2 - d1) * (d2 - d1) + (t2 - t1) * (t2 - t1))
        tail_K = tail_K_cache[str(dm)] if (d1 == dm and d2 == dm) else tail_K_dM
        best_sup = iv.mpf("1e100")
        for v_node in (v1, v2):
            deriv_abs = v_node[3]
            s_val = deriv_abs[0]
            h_pow = iv.mpf(1)
            for m in range(2, K_seg):
                h_pow = h_pow * h_seg
                s_val += (h_pow / fac_list[m - 1]) * deriv_abs[m - 1]
            h_pow = h_pow * h_seg
            s_val += (h_pow / fac_list[K_seg - 1]) * tail_K
            if s_val.b < best_sup.b:
                best_sup = iv.mpf(s_val.b)
        return h_seg, best_sup

    refined_pts = [pts[0]]
    for idx_p in range(len(pts) - 1):
        stack = [(pts[idx_p], pts[idx_p + 1], 0)]
        while stack:
            p1, p2, depth = stack.pop()
            v1 = _eval_vertex(p1[0], p1[1])
            v2 = _eval_vertex(p2[0], p2[1])
            R1, I1, mod1, _ = v1
            R2, I2, mod2, _ = v2
            dot_prod = R1 * R2 + I1 * I2
            h_seg, sup_Xip = _sup_Xip_on_segment(p1, p2, v1, v2)
            min_mod12 = iv.mpf(min(mod1.a, mod2.a))
            seg_margin_half = min_mod12 - iv.mpf("0.5") * h_seg * sup_Xip
            re_margin_full = mod1 - h_seg * sup_Xip
            if (dot_prod.a > 0 and seg_margin_half.a > 0 and re_margin_full.a > 0) or depth >= 8:
                refined_pts.append(p2)
            else:
                p_mid = ((p1[0] + p2[0]) * iv.mpf("0.5"), (p1[1] + p2[1]) * iv.mpf("0.5"))
                stack.append((p_mid, p2, depth + 1))
                stack.append((p1, p_mid, depth + 1))

    total_dtheta = iv.mpf(0)
    min_mod_sq = iv.mpf("1e100")
    min_dot_a = float("inf")
    min_seg_margin = float("inf")
    all_steps_acute = True

    for k in range(len(refined_pts) - 1):
        p1 = refined_pts[k]
        p2 = refined_pts[k + 1]
        v1 = _eval_vertex(p1[0], p1[1])
        v2 = _eval_vertex(p2[0], p2[1])
        R1, I1, mod1, _ = v1
        R2, I2, mod2, _ = v2
        mod_sq = R1 * R1 + I1 * I1
        if mod_sq.a < min_mod_sq.a:
            min_mod_sq = mod_sq
        dot_prod = R1 * R2 + I1 * I2
        cross_prod = R1 * I2 - I1 * R2
        h_seg, sup_Xip = _sup_Xip_on_segment(p1, p2, v1, v2)
        min_mod12 = iv.mpf(min(mod1.a, mod2.a))
        seg_margin = min_mod12 - iv.mpf("0.5") * h_seg * sup_Xip
        re_seg_low = mod1 * (mod1 - h_seg * sup_Xip)
        if float(dot_prod.a) < min_dot_a:
            min_dot_a = float(dot_prod.a)
        if float(seg_margin.a) < min_seg_margin:
            min_seg_margin = float(seg_margin.a)
        if dot_prod.a <= 0 or seg_margin.a <= 0 or re_seg_low.a <= 0:
            all_steps_acute = False
        ratio = cross_prod / dot_prod
        total_dtheta += iv_atan(ratio)

    N_zeros_iv = total_dtheta / (iv.mpf(2) * PI_IV)
    return N_zeros_iv, all_steps_acute, float(min_mod_sq.a), min_dot_a, min_seg_margin


# ==============================================================================
# SECTION 2: CANONICAL WIGNER MOMENTS & OFF-CRITICAL OBSTRUCTION MODEL (mpmath)
# ==============================================================================

def xi_exact_mp(z):
    s = mp.mpf("0.5") + mp.j * z
    return mp.mpf("0.5") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def verify_wigner_dipole_moments_and_obstruction_mpmath():
    """
    Verifies to 48+ decimal places the CANONICAL Wigner-hyperbolic identities:
      W(d,t)       = 16 int_0^inf y sinh(2dy) I(y,t) dy
      W_tilde(d,t) = 32 int_0^inf y^2 sinc_hyp(2dy) I(y,t) dy
      c_1(t) = L_1(Xi; t)   = 32      int_0^inf y^2 I(y,t) dy = (Xi')^2 - Xi*Xi''
      c_3(t) = 2*L_2(Xi; t) = (64/3)  int_0^inf y^4 I(y,t) dy
      c_5(t) = 3*L_3(Xi; t) = (64/15) int_0^inf y^6 I(y,t) dy,
    computes the true roots tau_1, tau_2 of Xi'(tau) = 0, AND demonstrates on an explicit
    real-entire off-critical zero model F_off(z) that parity reduction and Jensen-Laguerre
    algebra hold identically while L_1(F_off; t_star) < 0 and W_tilde_{F_off}(d, t_star) < 0!
    """
    Nu = 180
    Umax = mp.mpf("1.55")
    du = Umax / Nu
    u_nodes = [mp.mpf(j) * du for j in range(Nu + 1)]
    phi_vals = []
    phi_w = []
    for j, u in enumerate(u_nodes):
        e4 = mp.exp(4 * u)
        e5 = mp.exp(5 * u)
        e9 = mp.exp(9 * u)
        s = mp.mpf(0)
        for m in range(1, 15):
            m2 = mp.mpf(m * m)
            m4 = m2 * m2
            s += (2 * mp.pi**2 * m4 * e9 - 3 * mp.pi * m2 * e5) * mp.exp(-mp.pi * m2 * e4)
        w = mp.mpf("0.5") if (j == 0 or j == Nu) else mp.mpf("1.0")
        phi_vals.append(s)
        phi_w.append(8 * du * w * s)

    t_test = mp.zetazero(1).imag
    d_test = mp.mpf("0.20")

    D = [mp.mpf(0) for _ in range(7)]
    for j, u in enumerate(u_nodes):
        u2 = 2 * u
        w = phi_w[j]
        cs = mp.cos(u2 * t_test)
        sn = mp.sin(u2 * t_test)
        D[0] += w * cs
        D[1] -= w * (u2**1) * sn
        D[2] -= w * (u2**2) * cs
        D[3] += w * (u2**3) * sn
        D[4] += w * (u2**4) * cs
        D[5] -= w * (u2**5) * sn
        D[6] -= w * (u2**6) * cs

    c1_cr = D[1]**2 - D[0] * D[2]
    c3_cr = (D[0] * D[4] - 4 * D[1] * D[3] + 3 * D[2]**2) / 6
    c5_cr = (-D[0] * D[6] + 6 * D[1] * D[5] - 15 * D[2] * D[4] + 10 * D[3]**2) / 120

    # Folded canonical double integral over (u, v) in [0, inf)^2 with x = u - v, y = u + v:
    # Since Xi(t - i*d) = 4 int_R Phi(u) e^{2(d + i*t)u} du, we have
    #   |Xi(t - i*d)|^2 = 16 int_0^inf cosh(2dy) I(y,t) dy
    #   W(d,t)          = 16 int_0^inf y sinh(2dy) I(y,t) dy
    #   W_tilde(d,t)    = 32 int_0^inf y^2 sinc_hyp(2dy) I(y,t) dy
    #   c_{2m-1}(t)     = (2^{2m+3} / (2m-1)!) int_0^inf y^{2m} I(y,t) dy.
    c1_wig = mp.mpf(0)
    c3_wig = mp.mpf(0)
    c5_wig = mp.mpf(0)
    W_raw_wig = mp.mpf(0)
    W_tilde_wig = mp.mpf(0)

    for j in range(Nu + 1):
        uj = u_nodes[j]
        pwj = phi_w[j]
        for k in range(Nu + 1):
            uk = u_nodes[k]
            pw_pair = pwj * phi_w[k]  # = 64 * du * dv * w_j * w_k * Phi(u_j) * Phi(u_k)
            for y_val, x_val in ((uj - uk, uj + uk), (uj + uk, uj - uk)):
                w_cos = pw_pair * mp.cos(2 * t_test * x_val)
                y2 = y_val * y_val
                arg_2dy = 2 * d_test * y_val
                c1_wig += y2 * w_cos
                c3_wig += (mp.mpf(2) / mp.mpf(3)) * (y2 * y2) * w_cos
                c5_wig += (mp.mpf(2) / mp.mpf(15)) * (y2 * y2 * y2) * w_cos
                W_raw_wig += mp.mpf("0.5") * y_val * mp.sinh(arg_2dy) * w_cos
                sinch_2dy = mp.mpf(1) if arg_2dy == 0 else mp.sinh(arg_2dy) / arg_2dy
                W_tilde_wig += y2 * sinch_2dy * w_cos

    R_d = mp.mpf(0)
    Rt_d = mp.mpf(0)
    Itil_d = mp.mpf(0)
    Itilt_d = mp.mpf(0)
    for j, u in enumerate(u_nodes):
        u2 = 2 * u
        w = phi_w[j]
        arg_d = u2 * d_test
        ch = mp.cosh(arg_d)
        shc = mp.mpf(1) if arg_d == 0 else mp.sinh(arg_d) / arg_d
        cs = mp.cos(u2 * t_test)
        sn = mp.sin(u2 * t_test)
        R_d += w * ch * cs
        Rt_d -= w * u2 * ch * sn
        Itil_d += w * u2 * shc * sn
        Itilt_d += w * (u2**2) * shc * cs
    W_tilde_cr = R_d * Itilt_d - Itil_d * Rt_d
    W_raw_cr = d_test * W_tilde_cr

    # True critical-line extrema tau_1, tau_2 where Xi'(tau_k) = 0
    tau1 = mp.findroot(lambda t: mp.diff(xi_exact_mp, t).real, mp.mpf("15.5857"))
    tau2 = mp.findroot(lambda t: mp.diff(xi_exact_mp, t).real, mp.mpf("22.0980"))

    # Explicit off-critical zero obstruction model F_off(z):
    # Replace the conjugate symmetric quartet {+-gamma_1, +-gamma_2} of Xi(z) by off-line zeros
    # {+-(t_star +- i*d_star)} with t_star = (gamma_1 + gamma_2)/2, d_star = 0.15:
    g1 = mp.zetazero(1).imag
    g2 = mp.zetazero(2).imag
    t_star = mp.mpf("0.5") * (g1 + g2)
    d_star = mp.mpf("0.15")
    z_star = t_star - mp.j * d_star

    def F_off(z):
        num = (z - z_star) * (z - mp.conj(z_star)) * (z + z_star) * (z + mp.conj(z_star))
        den = (z * z - g1 * g1) * (z * z - g2 * g2)
        return xi_exact_mp(z) * (num / den)

    F0 = F_off(t_star).real
    Fp = mp.diff(F_off, t_star, 1).real
    Fpp = mp.diff(F_off, t_star, 2).real
    L1_F_off = Fp**2 - F0 * Fpp
    # Check W_tilde_{F_off}(0, t_star) == L1_F_off < 0 via finite/analytic d-derivative:
    Wtil_F_d0 = (mp.mpf("0.5") * mp.diff(lambda d: abs(F_off(t_star - mp.j * d))**2, mp.mpf(0), 2)) * mp.mpf("0.5")
    # Note: |F(t - id)|^2 = |F(t)|^2 + L_1(F; t)*d^2 + O(d^4), so (1/2)*(1/2)*d^2/dd^2 |F|^2|_{d=0}?
    # Wait: d/dd |F|^2 = 2*L_1*d + ..., d^2/dd^2 |F|^2|_{d=0} = 2*L_1, so (1/2)*d^2/dd^2 |F|^2 = L_1!
    Wtil_F_d0 = mp.mpf("0.5") * mp.diff(lambda d: abs(F_off(t_star - mp.j * d))**2, mp.mpf(0), 2)

    return {
        "W_tilde_cr": W_tilde_cr,
        "res_W_raw": abs(W_raw_wig - W_raw_cr),
        "res_W": abs(W_tilde_wig - W_tilde_cr),
        "res_c1": abs(c1_wig - c1_cr),
        "res_c3": abs(c3_wig - c3_cr),
        "res_c5": abs(c5_wig - c5_cr),
        "tau1": tau1,
        "tau2": tau2,
        "t_star": t_star,
        "d_star": d_star,
        "L1_F_off": L1_F_off,
        "res_L1_F_off": abs(Wtil_F_d0 - L1_F_off),
    }


# ==============================================================================
# SECTION 3: MAIN EXECUTION & FIGURE GENERATION
# ==============================================================================

def main():
    t0_global = time.time()
    print_banner("PAPER II VERIFICATION SUITE: PARITY-REDUCED WRONSKIANS & INTERVAL ENCLOSURES")
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    print(f"[*] Python / mpmath / PyTorch : {sys.version.split()[0]} / {mpmath.__version__} / {torch.__version__}")
    print(f"[*] Compute Device            : {dev_name}")
    print(f"[*] Working Precision         : mp.dps = {mp.dps}, iv.dps = {iv.dps}")

    # --------------------------------------------------------------------------
    # STEP 1: CANONICAL WIGNER-HYPERBOLIC & JENSEN-LAGUERRE MOMENT EQUIVALENCE
    # --------------------------------------------------------------------------
    print("\n" + "-" * 92)
    print(" STEP 1: CANONICAL WIGNER MOMENTS, JENSEN-LAGUERRE IDENTITIES & OFF-LINE OBSTRUCTION")
    print("-" * 92)
    wig_check = verify_wigner_dipole_moments_and_obstruction_mpmath()
    print(f"  |W_raw_Wigner   - W_raw_CR|   at (0.20, gamma_1) : {float(wig_check['res_W_raw']):.3e}")
    print(f"  |W_tilde_Wigner - W_tilde_CR| at (0.20, gamma_1) : {float(wig_check['res_W']):.3e}")
    print(f"  |c_1_Wigner     - L_1(Xi)|    at t = gamma_1     : {float(wig_check['res_c1']):.3e}")
    print(f"  |c_3_Wigner     - 2*L_2(Xi)|  at t = gamma_1     : {float(wig_check['res_c3']):.3e}")
    print(f"  |c_5_Wigner     - 3*L_3(Xi)|  at t = gamma_1     : {float(wig_check['res_c5']):.3e}")
    print(f"  True critical-line extrema Xi'(tau_k) = 0        : tau_1 = {mp.nstr(wig_check['tau1'], 12)}, tau_2 = {mp.nstr(wig_check['tau2'], 12)}")
    print(f"  Off-line model F_off at (d*=0.15, t*={float(wig_check['t_star']):.4f}): L_1(F_off; t*) = {float(wig_check['L1_F_off']):+.6e} < 0 (|res|={float(wig_check['res_L1_F_off']):.2e})")
    assert wig_check["res_W_raw"] < mp.mpf("1e-48")
    assert wig_check["res_W"] < mp.mpf("1e-48")
    assert wig_check["res_c1"] < mp.mpf("1e-48")
    assert wig_check["res_c3"] < mp.mpf("1e-48")
    assert wig_check["res_c5"] < mp.mpf("1e-48")
    assert wig_check["L1_F_off"] < 0
    assert wig_check["res_L1_F_off"] < mp.mpf("1e-45")

    # --------------------------------------------------------------------------
    # STEP 2: TABLE 2 — INTERVAL ENCLOSURES OF c_1, c_3, c_5 AT ZEROS gamma_1..gamma_10
    # --------------------------------------------------------------------------
    print("\n" + "-" * 92)
    print(" STEP 2: DIRECTED-ROUNDING INTERVAL ENCLOSURES OF c_1, c_3, c_5 AT gamma_1..gamma_10")
    print("-" * 92)
    hdr_t1 = (f"  {'Zero':<8} | {'gamma_k (mid)':>12} | {'IVT Root':>10} | "
              f"{'c_1(gamma_k).a':>18} | {'c_3(gamma_k).a':>18} | {'c_5(gamma_k).a':>18} | {'c_3/c_1':>8}")
    print(hdr_t1)
    print("  " + "-" * (len(hdr_t1) - 2))

    dipole_cert_data = []
    for k in range(1, 11):
        gk_mp = mp.zetazero(k).imag
        rad = mp.mpf("1e-16")
        gk_low = iv.mpf(str(gk_mp - rad))
        gk_high = iv.mpf(str(gk_mp + rad))
        gk_iv = iv.mpf([gk_low.a, gk_high.b])

        xi_low = iv_eval_d0_derivs_up_to_6(gk_low)[0]
        xi_high = iv_eval_d0_derivs_up_to_6(gk_high)[0]
        ivt_ok = (xi_low.b < 0 and xi_high.a > 0) or (xi_low.a > 0 and xi_high.b < 0)
        assert ivt_ok, f"IVT sign change failed for gamma_{k}"

        D = iv_eval_d0_derivs_up_to_6(gk_iv)
        c1_iv = D[1] * D[1] - D[0] * D[2]
        c3_iv = (iv.mpf("0.5") * (D[2] * D[2])
                 - (iv.mpf(2) / iv.mpf(3)) * D[1] * D[3]
                 + (iv.mpf(1) / iv.mpf(6)) * D[0] * D[4])
        c5_iv = ((iv.mpf(1) / iv.mpf(12)) * (D[3] * D[3])
                 + (iv.mpf(1) / iv.mpf(20)) * D[1] * D[5]
                 - (iv.mpf(1) / iv.mpf(8)) * D[2] * D[4]
                 - (iv.mpf(1) / iv.mpf(120)) * D[0] * D[6])

        assert c1_iv.a > 0 and c3_iv.a > 0 and c5_iv.a > 0
        env_iv = iv.exp(iv.mpf("0.5") * PI_IV * gk_iv)
        c1_hat = c1_iv * env_iv
        c3_hat = c3_iv * env_iv
        c5_hat = c5_iv * env_iv
        ratio_31 = float(c3_iv.a) / float(c1_iv.b)
        dipole_cert_data.append((k, float(gk_mp), c1_iv, c3_iv, c5_iv, c1_hat, c3_hat, c5_hat, ratio_31))
        print(f"  gamma_{k:<2d} | {float(gk_mp):>12.6f} | {'ENCLOSED':>10} | "
              f"{float(c1_iv.a):>+18.10e} | {float(c3_iv.a):>+18.10e} | {float(c5_iv.a):>+18.10e} | {ratio_31:>8.4f}")

    # --------------------------------------------------------------------------
    # STEP 3: DUAL INTERVAL EVALUATOR CROSS-CHECK, 80 2D MVT BOXES & ARGUMENT PRINCIPLE
    # --------------------------------------------------------------------------
    print("\n" + "-" * 92)
    print(" STEP 3A: DUAL EVALUATOR CROSS-CHECK (EULER-MACLAURIN vs JACOBI-THETA IN mpmath.iv)")
    print("-" * 92)
    test_points = [
        (iv.mpf("0.00"), iv.mpf("14.13472514"), "Critical line at gamma_1"),
        (iv.mpf("0.10"), iv.mpf("14.13472514"), "Inner collar d=0.10 at gamma_1"),
        (iv.mpf("0.25"), iv.mpf("15.58570859"), "Mid-strip d=0.25 at extremum tau_1"),
        (iv.mpf("0.35"), iv.mpf("21.02203964"), "Outer collar d=0.35 at gamma_2"),
        (iv.mpf("0.25"), iv.mpf("22.09797728"), "Mid-strip d=0.25 at extremum tau_2"),
        (iv.mpf("0.50"), iv.mpf("25.01085758"), "Strip boundary d=0.50 at gamma_3"),
        (iv.mpf("0.45"), iv.mpf("49.77383248"), "Outer strip d=0.45 at gamma_10"),
    ]
    for d_pt, t_pt, desc in test_points:
        xi_em = iv_xi_euler_maclaurin(d_pt, t_pt)
        res_jac = iv_eval_jacobi_full(d_pt, t_pt)
        xi_jac = res_jac["Xi"]
        w_til = res_jac["W_tilde"]
        overlap_re = max(xi_em.real.a, xi_jac.real.a) <= min(xi_em.real.b, xi_jac.real.b)
        overlap_im = max(xi_em.imag.a, xi_jac.imag.a) <= min(xi_em.imag.b, xi_jac.imag.b)
        assert overlap_re and overlap_im and (w_til.a > 0)
        w_wt = float(w_til.b - w_til.a)
        print(f"  ({float(d_pt.mid):.2f}, {float(t_pt.mid):>8.4f}) | {desc:<34} | "
              f"W_tilde.a = {float(w_til.a):+.10e} | width = {w_wt:.3e} | OVERLAP OK")

    print("\n" + "-" * 92)
    print(" STEP 3B: COLLAR BOXES (NON-HARMONIC-SAFE MVT) & 5 ARGUMENT-PRINCIPLE RECTANGLES")
    print("-" * 92)
    critical_zones = [
        ("Zone 1: Collar around gamma_1 (t=14.1347)", 14.115, 14.155),
        ("Zone 2: Collar around extremum tau_1 (t=15.5857)", 15.565, 15.605),
        ("Zone 3: Collar around gamma_2 (t=21.0220)", 21.002, 21.042),
        ("Zone 4: Collar around extremum tau_2 (t=22.0980)", 22.078, 22.118),
        ("Zone 5: Collar around gamma_3 (t=25.0109)", 24.991, 25.031),
    ]
    d_edges = [iv.mpf("0.00"), iv.mpf("0.10"), iv.mpf("0.22"), iv.mpf("0.35"), iv.mpf("0.50")]
    n_t_sub = 4
    all_certified_boxes = []
    for z_name, t_start_z, t_end_z in critical_zones:
        zone_boxes = []
        t_step = (iv.mpf(str(t_end_z)) - iv.mpf(str(t_start_z))) / iv.mpf(n_t_sub)
        for id_d in range(len(d_edges) - 1):
            da = d_edges[id_d]
            db = d_edges[id_d + 1]
            for id_t in range(n_t_sub):
                ta = iv.mpf(str(t_start_z)) + iv.mpf(id_t) * t_step
                tb = ta + t_step
                sub_boxes = adaptive_certify_collar_iv(da, db, ta, tb, depth=0, max_depth=3)
                zone_boxes.extend(sub_boxes)
        min_box_a = min(float(b[5].a) for b in zone_boxes)
        min_hat_a = min(float(b[5].a) * math.exp(0.5 * math.pi * 0.5 * (b[2] + b[3])) for b in zone_boxes)
        assert min_box_a > 0.0, f"Failed positivity in {z_name}: min_box_a = {min_box_a}"
        all_certified_boxes.extend([(z_name, b) for b in zone_boxes])
        print(f"  {z_name:<48} | Boxes: {len(zone_boxes):>2d} | min W_box.a = {min_box_a:+.6e} | min W_hat.a = {min_hat_a:+.4f}")

    macro_boxes = [
        ("Omega_1: [10^-6, 1.20] x [ 0.0, 12.0]", "1e-6", "1.20", "0.0",  "12.0", 60, 20),
        ("Omega_2: [10^-6, 1.20] x [12.0, 23.0]", "1e-6", "1.20", "12.0", "23.0", 70, 22),
        ("Omega_3: [10^-6, 1.20] x [23.0, 35.0]", "1e-6", "1.20", "23.0", "35.0", 80, 24),
        ("Omega_4: [10^-6, 1.20] x [35.0, 45.0]", "1e-6", "1.20", "35.0", "45.0", 80, 24),
        ("Omega_5: [10^-6, 1.20] x [45.0, 55.0]", "1e-6", "1.20", "45.0", "55.0", 90, 24),
    ]
    cauchy_results = []
    for b_name, dm_s, dM_s, tm_s, tM_s, nt_s, nd_s in macro_boxes:
        N_iv, acute_ok, min_mod2, min_dot, min_seg_mar = certify_cauchy_winding_box_iv(
            dm_s, dM_s, tm_s, tM_s, n_pts_t=nt_s, n_pts_d=nd_s
        )
        assert acute_ok and (min_seg_mar > 0.0) and (N_iv.a > iv.mpf("-0.5")) and (N_iv.b < iv.mpf("0.5"))
        cauchy_results.append((b_name, float(tm_s), float(tM_s), N_iv, min_mod2, min_dot, min_seg_mar))
        print(f"  {b_name:<42} | N_zeros in [{float(N_iv.a):+.2e}, {float(N_iv.b):+.2e}] | "
              f"seg_margin = {min_seg_mar:+.2e} => N_zeros == 0")

    # --------------------------------------------------------------------------
    # STEP 4: GPU LANDSCAPE VISUALIZATION & IEEE-754 FLOAT64 CANCELLATION AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "-" * 92)
    print(" STEP 4: IEEE-754 FLOAT64 CANCELLATION-AMPLIFICATION AUDIT vs 50-DPS MPMATH")
    print("-" * 92)
    t_gpu_start = time.time()
    Nd, Nt, Nu = 201, 4501, 4096
    d_vec = torch.linspace(0.0, 0.50, Nd, device=device, dtype=torch.float64)
    t_vec = torch.linspace(0.0, 45.0, Nt, device=device, dtype=torch.float64)
    u_vec = torch.linspace(0.0, 2.00, Nu, device=device, dtype=torch.float64)
    du = (u_vec[1] - u_vec[0]).item()

    phi_u = torch.zeros_like(u_vec)
    e4 = torch.exp(4.0 * u_vec)
    e5 = torch.exp(5.0 * u_vec)
    e9 = torch.exp(9.0 * u_vec)
    for n in range(1, 18):
        n2 = float(n * n)
        n4 = n2 * n2
        phi_u += (2.0 * (math.pi**2) * n4 * e9 - 3.0 * math.pi * n2 * e5) * torch.exp(-math.pi * n2 * e4)
    trap_w = torch.ones_like(u_vec)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5
    w_base = 8.0 * phi_u * trap_w * du
    u2_f = 2.0 * u_vec
    u4_f = 4.0 * (u_vec ** 2)

    arg_d = 2.0 * d_vec.unsqueeze(1) * u_vec.unsqueeze(0)
    A_cosh = torch.cosh(arg_d) * w_base.unsqueeze(0)
    sinch_d = torch.where(
        arg_d < 1.0e-4,
        1.0 + (arg_d ** 2) / 6.0 + (arg_d ** 4) / 120.0,
        torch.sinh(arg_d) / torch.clamp(arg_d, min=1.0e-30),
    )
    A_sinch = sinch_d * (w_base * u2_f).unsqueeze(0)

    arg_t = 2.0 * u_vec.unsqueeze(1) * t_vec.unsqueeze(0)
    B_cos = torch.cos(arg_t)
    B_sin = torch.sin(arg_t)

    R_gpu = torch.matmul(A_cosh, B_cos)
    Rt_gpu = -torch.matmul(A_cosh * u2_f.unsqueeze(0), B_sin)
    Rtt_gpu = -torch.matmul(A_cosh * u4_f.unsqueeze(0), B_cos)
    Itil_gpu = torch.matmul(A_sinch, B_sin)
    Itilt_gpu = torch.matmul(A_sinch * u2_f.unsqueeze(0), B_cos)

    W_tilde_gpu = R_gpu * Itilt_gpu - Itil_gpu * Rt_gpu
    env_t = torch.exp(0.5 * math.pi * t_vec).unsqueeze(0)
    W_hat_gpu = W_tilde_gpu * env_t
    if cuda_ok:
        torch.cuda.synchronize()
    gpu_elapsed = time.time() - t_gpu_start
    print(f"  [*] Exploratory float64 landscape on [0, 0.50] x [0, 45] ({Nd} x {Nt}) evaluated in {gpu_elapsed:.3f} s")

    # Pointwise float64 vs 50-dps mpmath audit across t in [10, 60] at d = 0.05:
    t_audit = [10.0, 20.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    d_aud = 0.05
    t_aud_t = torch.tensor(t_audit, device=device, dtype=torch.float64)
    arg_da = torch.tensor([[2.0 * d_aud]], device=device, dtype=torch.float64) * u_vec.unsqueeze(0)
    Ach_a = torch.cosh(arg_da) * w_base.unsqueeze(0)
    sinch_da = torch.where(
        arg_da < 1.0e-4,
        1.0 + (arg_da ** 2) / 6.0 + (arg_da ** 4) / 120.0,
        torch.sinh(arg_da) / torch.clamp(arg_da, min=1.0e-30),
    )
    Ash_a = sinch_da * (w_base * u2_f).unsqueeze(0)
    Bc_a = torch.cos(2.0 * u_vec.unsqueeze(1) * t_aud_t.unsqueeze(0))
    Bs_a = torch.sin(2.0 * u_vec.unsqueeze(1) * t_aud_t.unsqueeze(0))
    Ra = torch.matmul(Ach_a, Bc_a).squeeze(0).cpu().numpy()
    Rta = (-torch.matmul(Ach_a * u2_f.unsqueeze(0), Bs_a)).squeeze(0).cpu().numpy()
    Ita = torch.matmul(Ash_a, Bs_a).squeeze(0).cpu().numpy()
    Itta = torch.matmul(Ash_a * u2_f.unsqueeze(0), Bc_a).squeeze(0).cpu().numpy()
    What_f64 = (Ra * Itta - Ita * Rta) * np.exp(0.5 * math.pi * np.array(t_audit))
    assert not np.any(np.isnan(What_f64)), "NaN detected in float64 audit!"

    # L1 absolute inner-product norms |x|^T |y| for Higham's true bound:
    L1_R = float(torch.sum(torch.abs(Ach_a)).item())
    L1_Rt = float(torch.sum(torch.abs(Ach_a * u2_f.unsqueeze(0))).item())
    L1_It = float(torch.sum(torch.abs(Ash_a)).item())
    L1_Itt = float(torch.sum(torch.abs(Ash_a * u2_f.unsqueeze(0))).item())
    u_mach = 2.0 ** -53
    gamma_Nu = (Nu * u_mach) / (1.0 - Nu * u_mach)

    audit_rows = []
    print(f"\n  {'t':>6} | {'W_hat_exact (50 dps)':>20} | {'W_hat_float64':>20} | {'|Abs Error|':>14} | {'Rel Error':>12} | {'Higham Bound':>14}")
    print("  " + "-" * 98)
    for idx_a, tv in enumerate(t_audit):
        res_iv = iv_eval_jacobi_full(iv.mpf("0.05"), iv.mpf(str(tv)))
        w_ex = float(res_iv["W_tilde"].mid) * math.exp(0.5 * math.pi * tv)
        R_ex = abs(float(res_iv["R"].mid))
        Rt_ex = abs(float(res_iv["Rt"].mid))
        It_ex = abs(float(res_iv["I"].mid)) / d_aud
        # Compute exact Itilt from W_tilde or mpmath:
        z_mp = mp.mpf(str(tv)) - mp.j * mp.mpf("0.05")
        xi_v = xi_exact_mp(z_mp)
        xip_v = mp.diff(xi_exact_mp, z_mp)
        wtil_mp = float(((xi_v.real * xip_v.imag - xi_v.imag * xip_v.real) / mp.mpf("0.05")) * mp.exp(mp.mpf("0.5") * mp.pi * mp.mpf(str(tv))))
        abs_err = abs(What_f64[idx_a] - wtil_mp)
        rel_err = abs_err / abs(wtil_mp)
        Itt_ex = abs(float(xip_v.imag)) / d_aud
        # True Higham inner-product bound amplified by exp(pi*t/2):
        err_Wtil_higham = gamma_Nu * (L1_R * Itt_ex + R_ex * L1_Itt + L1_It * Rt_ex + It_ex * L1_Rt)
        higham_hat = err_Wtil_higham * math.exp(0.5 * math.pi * tv)
        iv_width_hat = float(res_iv["W_tilde"].b - res_iv["W_tilde"].a) * math.exp(0.5 * math.pi * tv)
        audit_rows.append((tv, wtil_mp, What_f64[idx_a], abs_err, rel_err, higham_hat, iv_width_hat))
        print(f"  {tv:>6.1f} | {wtil_mp:>+20.8e} | {What_f64[idx_a]:>+20.8e} | {abs_err:>14.4e} | {rel_err:>11.3e} | {higham_hat:>14.4e}")

    # Dense curve of float64 error vs mpmath on t in [10, 60] for Figure 2(c)
    t_dense_aud = np.linspace(10.0, 60.0, 101)
    t_da_t = torch.tensor(t_dense_aud, device=device, dtype=torch.float64)
    Ra_d = torch.matmul(Ach_a, torch.cos(2.0 * u_vec.unsqueeze(1) * t_da_t.unsqueeze(0))).squeeze(0).cpu().numpy()
    Rta_d = (-torch.matmul(Ach_a * u2_f.unsqueeze(0), torch.sin(2.0 * u_vec.unsqueeze(1) * t_da_t.unsqueeze(0)))).squeeze(0).cpu().numpy()
    Ita_d = torch.matmul(Ash_a, torch.sin(2.0 * u_vec.unsqueeze(1) * t_da_t.unsqueeze(0))).squeeze(0).cpu().numpy()
    Itta_d = torch.matmul(Ash_a * u2_f.unsqueeze(0), torch.cos(2.0 * u_vec.unsqueeze(1) * t_da_t.unsqueeze(0))).squeeze(0).cpu().numpy()
    What_f64_dense = (Ra_d * Itta_d - Ita_d * Rta_d) * np.exp(0.5 * math.pi * t_dense_aud)
    rel_err_dense = []
    higham_rel_dense = []
    iv_rel_dense = []
    for idx_d, tv in enumerate(t_dense_aud):
        res_iv = iv_eval_jacobi_full(iv.mpf("0.05"), iv.mpf(f"{tv:.4f}"))
        env_v = math.exp(0.5 * math.pi * tv)
        w_ex = float(res_iv["W_tilde"].mid) * env_v
        R_ex = abs(float(res_iv["R"].mid))
        Rt_ex = abs(float(res_iv["Rt"].mid))
        It_ex = abs(float(res_iv["I"].mid)) / d_aud
        Itt_ex = abs(float(res_iv["It"].mid)) / d_aud
        abs_e = abs(What_f64_dense[idx_d] - w_ex)
        rel_err_dense.append(max(abs_e / abs(w_ex), 1e-17))
        h_hat = gamma_Nu * (L1_R * Itt_ex + R_ex * L1_Itt + L1_It * Rt_ex + It_ex * L1_Rt) * env_v
        higham_rel_dense.append(h_hat / abs(w_ex))
        iv_w = float(res_iv["W_tilde"].b - res_iv["W_tilde"].a) * env_v
        iv_rel_dense.append(iv_w / abs(w_ex))

    d_cpu = d_vec.cpu().numpy()
    t_cpu = t_vec.cpu().numpy()
    W_hat_cpu = np.maximum(W_hat_gpu.cpu().numpy(), 1e-3)
    R0_hat = (R_gpu[0, :] * torch.exp(0.25 * math.pi * t_vec)).cpu().numpy()
    Rt0_hat = (Rt_gpu[0, :] * torch.exp(0.25 * math.pi * t_vec)).cpu().numpy()
    Rtt0_hat = (Rtt_gpu[0, :] * torch.exp(0.25 * math.pi * t_vec)).cpu().numpy()

    # --------------------------------------------------------------------------
    # STEP 5: GENERATE PUBLICATION FIGURES (300 DPI PDF & PNG)
    # --------------------------------------------------------------------------
    fig_dir = Path(__file__).resolve().parent.parent / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    zeros_8_vals = [row[1] for row in dipole_cert_data if row[1] <= 45.0]
    zeros_10_vals = [row[1] for row in dipole_cert_data]

    # --- FIGURE 1: Parity-Reduced Wronskian Landscape & Jensen-Laguerre Moments ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(13.5, 10.2), dpi=300)

    ax1a = axes1[0, 0]
    D_sub, T_sub = np.meshgrid(d_cpu[::2], t_cpu[::5], indexing="ij")
    W_sub = W_hat_cpu[::2, ::5]
    im1a = ax1a.pcolormesh(
        T_sub, D_sub, W_sub,
        norm=LogNorm(vmin=max(np.min(W_sub), 1e-2), vmax=np.max(W_sub)),
        cmap="viridis", shading="auto", rasterized=True
    )
    cb1a = fig1.colorbar(im1a, ax=ax1a, pad=0.02)
    cb1a.set_label(r"Envelope-scaled $\widehat{W}(d,t) = \mathrm{e}^{\pi t/2}\widetilde{W}(d,t)$", fontsize=9)
    ax1a.plot(zeros_8_vals, np.zeros_like(zeros_8_vals), "^", color="#e74c3c", mec="white", ms=6.5, zorder=6,
              label=r"Critical-line zeros $\gamma_1, \dots, \gamma_8$ ($d=0$)")
    ax1a.set_xlim(0.0, 45.0)
    ax1a.set_ylim(0.0, 0.50)
    ax1a.set_title(r"(a) Numerical Landscape of $\widehat{W}(d,t)$ on $[0, 0.50] \times [0, 45]$", fontsize=10.5, fontweight="bold")
    ax1a.set_xlabel(r"Spectral height $t$", fontsize=9.5)
    ax1a.set_ylabel(r"Transverse offset $d = \sigma - 1/2$", fontsize=9.5)
    ax1a.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    ax1a.grid(True, alpha=0.20, color="white")

    ax1b = axes1[0, 1]
    d_plot = np.linspace(0.0, 0.50, 200)
    for k_idx, z_lbl, col in [
        (0, r"$\gamma_1 \approx 14.13$", "#1f77b4"),
        (1, r"$\gamma_2 \approx 21.02$", "#d62728"),
        (2, r"$\gamma_3 \approx 25.01$", "#2ca02c"),
        (4, r"$\gamma_5 \approx 32.94$", "#9467bd"),
        (9, r"$\gamma_{10} \approx 49.77$", "#e67e22"),
    ]:
        row_k = dipole_cert_data[k_idx]
        c1_v = float(row_k[2].mid)
        c3_v = float(row_k[3].mid)
        c5_v = float(row_k[4].mid)
        w_til_norm = 1.0 + (c3_v / c1_v) * (d_plot ** 2) + (c5_v / c1_v) * (d_plot ** 4)
        ax1b.plot(d_plot, w_til_norm, lw=2.1, color=col, label=rf"$\widetilde{{W}}(d,t)/c_1(t)$ at {z_lbl}")
    ax1b.axhline(1.0, color="black", ls="--", lw=1.4, label=r"Flat Neumann tangent $\partial_d\widetilde{W}(0,t) = 0$")
    ax1b.set_xlim(0.0, 0.50)
    ax1b.set_ylim(0.98, 1.34)
    ax1b.set_title(r"(b) Flat Neumann Tangent and Upward Convexity of $\widetilde{W}(d,t)/c_1(t)$",
                   fontsize=10.5, fontweight="bold")
    ax1b.set_xlabel(r"Transverse offset $d = \sigma - 1/2$", fontsize=9.5)
    ax1b.set_ylabel(r"Normalized parity-reduced $\widetilde{W}(d,t)/c_1(t)$", fontsize=9.5)
    ax1b.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax1b.grid(True, alpha=0.25)

    ax1c = axes1[1, 0]
    k_vals = np.array([row[0] for row in dipole_cert_data])
    c1_hats = np.array([float(row[5].a) for row in dipole_cert_data])
    c3_hats = np.array([float(row[6].a) for row in dipole_cert_data])
    c5_hats = np.array([float(row[7].a) for row in dipole_cert_data])
    w_bar = 0.26
    ax1c.bar(k_vals - w_bar, c1_hats, width=w_bar, color="#1f77b4", edgecolor="black", lw=0.6,
             label=r"$\widehat{c}_1(\gamma_k).a = \mathrm{e}^{\pi\gamma_k/2}L_1(\Xi;\gamma_k) > 0$")
    ax1c.bar(k_vals, c3_hats, width=w_bar, color="#2ca02c", edgecolor="black", lw=0.6,
             label=r"$\widehat{c}_3(\gamma_k).a = 2\mathrm{e}^{\pi\gamma_k/2}L_2(\Xi;\gamma_k) > 0$")
    ax1c.bar(k_vals + w_bar, c5_hats, width=w_bar, color="#e67e22", edgecolor="black", lw=0.6,
             label=r"$\widehat{c}_5(\gamma_k).a = 3\mathrm{e}^{\pi\gamma_k/2}L_3(\Xi;\gamma_k) > 0$")
    ax1c.set_yscale("log")
    ax1c.set_xticks(k_vals)
    ax1c.set_xticklabels([rf"$\gamma_{{{k}}}$" for k in k_vals], fontsize=9.0)
    ax1c.set_title(r"(c) Interval-Enclosed Moments $\widehat{c}_1, \widehat{c}_3, \widehat{c}_5 > 0$ at $\gamma_1, \dots, \gamma_{10}$",
                   fontsize=10.5, fontweight="bold")
    ax1c.set_xlabel(r"Non-trivial Riemann zeros $\gamma_1, \dots, \gamma_{10}$", fontsize=9.5)
    ax1c.set_ylabel(r"Envelope-scaled lower endpoint (log scale)", fontsize=9.5)
    ax1c.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    ax1c.grid(True, which="both", alpha=0.25)

    ax1d = axes1[1, 1]
    mask_t = (t_cpu >= 12.0) & (t_cpu <= 36.0)
    t_sub_d = t_cpu[mask_t]
    stir_scale = (t_sub_d / (2.0 * math.pi)) ** 3.5
    kin_term = (Rt0_hat[mask_t] ** 2) / stir_scale
    pot_term = -(R0_hat[mask_t] * Rtt0_hat[mask_t]) / stir_scale
    tot_c1 = kin_term + pot_term
    ax1d.plot(t_sub_d, tot_c1, color="#1a252f", lw=2.0,
              label=r"Laguerre expression $L_1(\Xi;t) = (\Xi'(t))^2 - \Xi(t)\Xi''(t)$")
    ax1d.plot(t_sub_d, kin_term, color="#2980b9", lw=1.4, ls="--",
              label=r"Square derivative term $(\Xi'(t))^2 \geq 0$ (vanishes at $\tau_1, \tau_2$)")
    ax1d.fill_between(t_sub_d, 0, tot_c1, color="#3498db", alpha=0.16)
    for z_t in zeros_10_vals[:5]:
        ax1d.axvline(z_t, color="#e74c3c", lw=1.0, ls=":", alpha=0.85)
    for tau_v in (float(wig_check["tau1"]), float(wig_check["tau2"])):
        ax1d.axvline(tau_v, color="#8e44ad", lw=1.0, ls="-.", alpha=0.80)
    ax1d.plot([], [], color="#e74c3c", lw=1.1, ls=":", label=r"Zeros $\gamma_1, \dots, \gamma_5$ ($\Xi(\gamma_k)=0$)")
    ax1d.plot([], [], color="#8e44ad", lw=1.1, ls="-.", label=r"Extrema $\tau_1 \approx 15.59,\ \tau_2 \approx 22.10$ ($\Xi'(\tau_k)=0$)")
    ax1d.set_xlim(12.0, 36.0)
    ax1d.set_ylim(0.0, max(np.max(tot_c1), np.max(kin_term)) * 1.22)
    ax1d.set_title(r"(d) Classical Laguerre Expression $\widetilde{W}(0,t) = L_1(\Xi;t)$ ($d = 0$)", fontsize=10.5, fontweight="bold")
    ax1d.set_xlabel(r"Spectral height $t$", fontsize=9.5)
    ax1d.set_ylabel(r"Stirling-normalized $\mathrm{e}^{\pi t/2}(t/2\pi)^{-7/2}L_1(\Xi;t)$", fontsize=9.5)
    ax1d.legend(loc="upper right", fontsize=7.4, framealpha=0.92)
    ax1d.grid(True, alpha=0.25)

    plt.tight_layout()
    f1_pdf = fig_dir / "fig1_parity_reduced_wronskian_and_dipoles.pdf"
    f1_png = fig_dir / "fig1_parity_reduced_wronskian_and_dipoles.png"
    fig1.savefig(f1_pdf, dpi=300)
    fig1.savefig(f1_png, dpi=300)
    plt.close(fig1)
    print(f"\n  [*] Saved Figure 1: {f1_pdf.name} ({f1_pdf.stat().st_size/1024:.1f} KB)")

    # --- FIGURE 2: 2D Interval-MVT Boxes, IEEE-754 Cancellation Audit & Argument Principle ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(13.5, 10.2), dpi=300)

    ax2a = axes2[0, 0]
    for _, b_tuple in all_certified_boxes:
        da_b, db_b, ta_b, tb_b, _, _, _ = b_tuple
        rect = Rectangle((ta_b, da_b), tb_b - ta_b, db_b - da_b,
                         linewidth=1.1, edgecolor="#c0392b", facecolor="#fadbd8", alpha=0.55, zorder=5)
        ax2a.add_patch(rect)
    ax2a.plot([], [], color="#c0392b", lw=1.4, label=rf"{len(all_certified_boxes)} 2D Interval-MVT collar boxes ($\widetilde{{W}}(\mathbf{{B}}).a > 0$)")
    for idx_c, (_, tm_c, tM_c, _, _, _, _) in enumerate(cauchy_results):
        rect_c = Rectangle((tm_c, 0.002), tM_c - tm_c, 1.18,
                           linewidth=1.5, linestyle="--", edgecolor="#2980b9", facecolor="#ebf5fb", alpha=0.45, zorder=3)
        ax2a.add_patch(rect_c)
        ax2a.text(0.5 * (tm_c + tM_c), 0.85, rf"$\Omega_{{{idx_c+1}}}$" + "\n" + r"$N=0$",
                  color="#1a252f", fontweight="bold", fontsize=8.2, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#2980b9", alpha=0.92), zorder=8)
    ax2a.plot([], [], color="#2980b9", lw=1.5, ls="--", label=r"Argument-Principle rectangles $\Omega_1, \dots, \Omega_5$ ($N_{\mathrm{zeros}}=0$)")
    ax2a.plot(zeros_10_vals, np.zeros_like(zeros_10_vals), "^", color="#e74c3c", mec="black", ms=6.0, zorder=9,
              label=r"Critical-line zeros $\gamma_1, \dots, \gamma_{10}$")
    ax2a.set_xlim(0.0, 55.0)
    ax2a.set_ylim(-0.03, 1.23)
    ax2a.set_title(r"(a) Benchmark Partition: Collar Boxes ($\gamma_k,\tau_k$) and Rectangles $\Omega_1, \dots, \Omega_5$", fontsize=10.2, fontweight="bold")
    ax2a.set_xlabel(r"Spectral height $t$", fontsize=9.5)
    ax2a.set_ylabel(r"Transverse offset $d = \sigma - 1/2$", fontsize=9.5)
    ax2a.legend(loc="upper left", fontsize=7.6, framealpha=0.94)
    ax2a.grid(True, alpha=0.25)

    ax2b = axes2[0, 1]
    box_indices = np.arange(1, len(all_certified_boxes) + 1)
    box_mids_hat, box_lows_hat, box_highs_hat = [], [], []
    for _, b_tuple in all_certified_boxes:
        da_b, db_b, ta_b, tb_b, W0_iv, Wbox_iv, _ = b_tuple
        t_mid_b = 0.5 * (ta_b + tb_b)
        scale_env = math.exp(0.5 * math.pi * t_mid_b)
        box_mids_hat.append(float(W0_iv.mid) * scale_env)
        box_lows_hat.append(float(Wbox_iv.a) * scale_env)
        box_highs_hat.append(float(Wbox_iv.b) * scale_env)
    box_mids_hat = np.array(box_mids_hat)
    box_lows_hat = np.array(box_lows_hat)
    box_highs_hat = np.array(box_highs_hat)

    ax2b.fill_between(box_indices, box_lows_hat, box_highs_hat, color="#2ecc71", alpha=0.35,
                      label=r"Interval enclosure $[\widehat{W}(\mathbf{B}).a, \widehat{W}(\mathbf{B}).b]$")
    ax2b.plot(box_indices, box_mids_hat, "o-", color="#1a252f", ms=2.6, lw=1.1,
              label=r"Box center evaluation $\widehat{W}(d_0, t_0)$")
    ax2b.plot(box_indices, box_lows_hat, "-", color="#27ae60", lw=1.6,
              label=r"Lower endpoint $\widehat{W}(\mathbf{B}).a > 0$")
    ax2b.axhline(0.0, color="#c0392b", lw=1.3, ls="--", label=r"Zero threshold $\widehat{W} = 0$")
    ax2b.set_xlim(1, len(all_certified_boxes))
    ax2b.set_ylim(-5000.0, max(box_highs_hat) * 1.10)
    ax2b.set_title(rf"(b) Non-Harmonic-Safe 2D Interval-MVT Enclosures ({len(all_certified_boxes)} Collar Boxes)", fontsize=10.2, fontweight="bold")
    ax2b.set_xlabel(rf"Collar box index $m \in \{{1, \dots, {len(all_certified_boxes)}\}}$ across $\gamma_1, \tau_1, \gamma_2, \tau_2, \gamma_3$", fontsize=9.2)
    ax2b.set_ylabel(r"Envelope-scaled $\widehat{W}(\mathbf{B}_m)$", fontsize=9.5)
    ax2b.legend(loc="upper left", fontsize=7.6, framealpha=0.92)
    ax2b.grid(True, alpha=0.25)

    # Panel (c): IEEE-754 float64 Cancellation Audit vs mpmath.iv
    ax2c = axes2[1, 0]
    ax2c.semilogy(t_dense_aud, rel_err_dense, color="#c0392b", lw=1.8,
                  label="Measured IEEE-754 float64 rel. error ($d=0.05$)")
    ax2c.semilogy(t_dense_aud, higham_rel_dense, color="#e67e22", lw=1.6, ls="--",
                  label=r"Higham $|x|^\top|y|$ bound $\propto u_{\mathrm{mach}}\mathrm{e}^{+\pi t/4}$")
    ax2c.semilogy(t_dense_aud, iv_rel_dense, color="#27ae60", lw=1.8, ls="-.",
                  label="Rigorous mpmath.iv rel. width (iv.dps=35)")
    ax2c.axhline(1.0, color="black", lw=1.2, ls=":", label=r"100% relative error ($O(1)$ loss of sign)")
    ax2c.set_xlim(10.0, 60.0)
    ax2c.set_ylim(1e-36, 1e3)
    ax2c.set_title("  (c) IEEE-754 float64 Cancellation Breakdown vs. mpmath.iv", fontsize=10.2, fontweight="bold")
    ax2c.set_xlabel(r"Spectral height $t$", fontsize=9.5)
    ax2c.set_ylabel(r"Relative error / relative interval width (log scale)", fontsize=9.5)
    ax2c.legend(loc="lower right", fontsize=7.4, framealpha=0.92)
    ax2c.grid(True, which="both", alpha=0.25)

    ax2d = axes2[1, 1]
    omega_idx = np.arange(1, len(cauchy_results) + 1)
    iv_radii = np.array([max(abs(float(r[3].a)), abs(float(r[3].b))) for r in cauchy_results])
    ax2d.semilogy(omega_idx, iv_radii, "s-", color="#2980b9", ms=7.0, lw=2.0,
                  label=r"Interval radius $\max(|a_k|, |b_k|)$ of $N_{\mathrm{zeros}}(\Omega_k) \in [a_k, b_k]$")
    ax2d.axhline(0.5, color="#c0392b", lw=1.8, ls="--",
                 label=r"Integer decision threshold $\pm 0.5$ ($N_{\mathrm{zeros}} \in \mathbb{Z}$)")
    ax2d.fill_between([0.6, 5.4], 1e-35, 0.5, color="#eafaf1", alpha=0.50,
                      label=r"Zero-free enclosure $|N_{\mathrm{zeros}}| < 0.5 \Rightarrow N_{\mathrm{zeros}} = 0$")
    for idx_o, r_val in zip(omega_idx, iv_radii):
        ax2d.annotate(rf"$\pm {r_val:.2e}$" + "\n" + r"($N=0$)", (idx_o, r_val),
                      textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7.8,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2980b9"))
    ax2d.set_xticks(omega_idx)
    ax2d.set_xticklabels([
        r"$\Omega_1$" + "\n" + r"$[0, 12]$",
        r"$\Omega_2$" + "\n" + r"$[12, 23]$",
        r"$\Omega_3$" + "\n" + r"$[23, 35]$",
        r"$\Omega_4$" + "\n" + r"$[35, 45]$",
        r"$\Omega_5$" + "\n" + r"$[45, 55]$",
    ], fontsize=8.8)
    ax2d.set_xlim(0.6, 5.4)
    ax2d.set_ylim(1e-35, 5.0)
    ax2d.set_title(r"(d) Integer Argument Principle Enclosures $N_{\mathrm{zeros}}(\Omega_k) \in (-0.5, 0.5) \cap \mathbb{Z} = \{0\}$",
                   fontsize=10.2, fontweight="bold")
    ax2d.set_xlabel(r"Macro-rectangles $\Omega_k = [10^{-6}, 1.20] \times [t_{k-1}, t_k]$", fontsize=9.5)
    ax2d.set_ylabel(r"Winding-number interval radius (log scale)", fontsize=9.5)
    ax2d.legend(loc="lower right", fontsize=7.6, framealpha=0.92)
    ax2d.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    f2_pdf = fig_dir / "fig2_interval_mvt_and_argument_principle.pdf"
    f2_png = fig_dir / "fig2_interval_mvt_and_argument_principle.png"
    fig2.savefig(f2_pdf, dpi=300)
    fig2.savefig(f2_png, dpi=300)
    plt.close(fig2)
    print(f"  [*] Saved Figure 2: {f2_pdf.name} ({f2_pdf.stat().st_size/1024:.1f} KB)")

    elapsed = time.time() - t0_global
    print("\n" + "=" * 92)
    print(f" [SUCCESS] PAPER II VERIFICATION SUITE COMPLETED IN {elapsed:.2f} SECONDS (EXIT CODE 0)")
    print("=" * 92)


if __name__ == "__main__":
    main()
