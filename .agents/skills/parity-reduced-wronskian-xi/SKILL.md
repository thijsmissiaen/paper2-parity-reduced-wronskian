---
name: parity-reduced-wronskian-xi
description: >-
  Mathematical knowledge and verification skill for Paper 2 ("Parity-Reduced Transverse
  Wronskians, Jensen-Laguerre Moment Identities, and Division-Free Interval Enclosures for
  Riemann's Xi-Function"). Activate this skill when working with the canonical parity-reduced
  Wronskian W_tilde(d,t) = W(d,t)/d = 32 int y^2 sinc_hyp(2dy) I(y,t) dy, the Jensen (1913)
  and Csordas-Varga (1990) differential invariants c_{2m-1}(t) = m L_m(Xi; t), non-harmonic-safe
  2D Interval-MVT bounds via integral representations of sinc_hyp and eta_hyp, or Higham's
  IEEE-754 float64 cancellation-amplification bound.
---

# Parity-Reduced Transverse Wronskian & Interval Enclosure Skill (`parity-reduced-wronskian-xi`)

This skill provides the canonical parity-reduced Wronskian identities, classical Jensen–Laguerre moment attributions, non-harmonic-safe interval Taylor bounds (`mpmath.iv`), and IEEE-754 `float64` cancellation obstructions established in Paper 2.

---

## 1. Canonical Parity-Reduced Wronskian and Jensen–Laguerre Moments

1. **Completed Riemann $\Xi$-Function and Canonical Transverse Wronskian:**
   With $s = \frac{1}{2} + d + \mathrm{i}t$ ($d = \sigma - \frac{1}{2}$) and $z = t - \mathrm{i}d$, $\Xi(t-\mathrm{i}d) = 4\int _ {-\infty}^\infty \Phi(u)\mathrm{e}^{2(d+\mathrm{i}t)u}\,\mathrm{d}u = R(d,t) + \mathrm{i}I _ {\mathrm{im}}(d,t)$ satisfies Schwarz reflection $\Xi(t+\mathrm{i}d) = \overline{\Xi(t-\mathrm{i}d)}$, so $W(d,t) := \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2 = -W(-d,t)$ satisfies

$$W(d,t) = 16\int_0^\infty y\sinh(2d y) I(y,t)\,\mathrm{d}y, \qquad \widetilde{W}(d,t) := \frac{W(d,t)}{d} = 32\int_0^\infty y^2 \mathrm{sinc} _ {\mathrm{hyp}}(2d y) I(y,t)\,\mathrm{d}y,$$

   where $I(y,t) = \int _ {-\infty}^\infty \Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})\cos(2tx)\,\mathrm{d}x = 2\pi\mathcal{W} _ \Phi(y/2, 2t)$ and $\mathrm{sinc} _ {\mathrm{hyp}}(w) := \frac{\sinh w}{w} = \int_0^1 \cosh(sw)\,\mathrm{d}s \ge 1$. On $d = 0$, $\partial_d\widetilde{W}(0,t) \equiv 0$ and $\widetilde{W}(0,t) = L _ 1(\Xi;t) := (\Xi'(t))^2 - \Xi(t)\Xi''(t)$ is the classical **Laguerre expression**.

2. **Jensen (1913) and Csordas–Varga (1990) Moment Hierarchy:**
   The Maclaurin series $\widetilde{W}(d,t) = \sum _ {m=1}^\infty c _ {2m-1}(t) d^{2m-2}$ has coefficients

$$c _ {2m-1}(t) = \frac{2^{2m+3}}{(2m-1)!}\int_0^\infty y^{2m}I(y,t)\,\mathrm{d}y = m L _ m(\Xi;t) := m\sum _ {j=0}^{2m}\frac{(-1)^{m+j}}{j! (2m-j)!}\Xi^{(j)}(t)\Xi^{(2m-j)}(t).$$

   Global positivity $L _ m(\Xi;t) > 0$ ($\forall m \ge 1, t \in \mathbb{R}$) is equivalent to RH (Csordas–Varga 1990) and fails on off-critical zero models ($L _ 1(F _ {\mathrm{off}}; t _ *) < 0$, Davenport–Heilbronn $\Xi _ {\mathrm{DH}}$). Note also that the true critical-line extrema solving $\Xi'(\tau _ k) = 0$ are $\tau _ 1 \approx 15.58570859$ and $\tau _ 2 \approx 22.09797728$ (distinct from both the extrema $t _ {Z',1} \approx 17.882582, t _ {Z',2} \approx 23.104651$ of Hardy's $Z(t)$ and the Gram points $g _ 0 \approx 17.8456, g _ 1 \approx 23.1703$).

## 2. Non-Harmonic-Safe Gradient Bounds and `float64` Obstruction

1. **Positive-Weight Integral Majorants (Proposition 4.1):**
   Because $\widetilde{I} = I _ {\mathrm{im}}/d$ and $\widehat{I} _ d$ are *not* harmonic, mixed $(d,t)$ Taylor remainders must use $\mathrm{sinc} _ {\mathrm{hyp}}(w) = \int_0^1 \cosh(sw)\,\mathrm{d}s$ and $\eta _ {\mathrm{hyp}}(w) = \int_0^1\int_0^1 s^2\cosh(s\tau w)\,\mathrm{d}\tau\,\mathrm{d}s$, giving $\lvert\partial_d^p\partial_t^q\widetilde{I}^{(k)}\rvert \le \frac{1}{p+1}M _ {p+q+k+1}$ and $\lvert\partial_d^p\partial_t^q\widehat{I}^{(k)}\rvert \le \frac{1}{(p+1)(p+3)}M _ {p+q+k+3}$ where $M _ m = \sup\lvert\Xi^{(m)}\rvert$.

2. **IEEE-754 `float64` Catastrophic Cancellation:**
   Higham's bound $\lvert\mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma _ {N _ u}\lvert x\rvert^\top\lvert y\rvert$ implies $\lvert\widehat{W} _ {\mathrm{float64}} - \widehat{W}\rvert \propto u _ {\mathrm{mach}}\mathrm{e}^{+\pi t/4}$ (4.51% error at $t = 55$, 1,375% error at $t = 60$); heuristic `float64` grid scans cannot certify positivity as $t$ grows.

## 3. Verification Execution

```bash
python scripts/verify_paper2_parity_reduced_wronskian.py
```
