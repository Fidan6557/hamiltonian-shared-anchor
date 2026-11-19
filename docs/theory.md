# Theory notes: what sharing a reading does to derived pairs

The notes distinguish exact noise-covariance identities, small-step/small-noise expansions, population
orthogonality, and **local-averaging idealizations**. Tests in `tests/` check specific algebra and asymptotic
orders; the `decompose` mechanism set checks how the idealizations describe finite-sample fits.

## Setup

Readings are y_r = z(t_r) + e_r with i.i.d. noise e_r (covariance Σ, per coordinate σ²; the
experiments use isotropic Σ = σ² I with σ = a·z_rms). A pair (r_a, r_b) separated by h gives an
implicit-midpoint (MP) anchor x̃ = (y_a + y_b)/2 and velocity v = (y_b − y_a)/h. For clean exact-flow data,
write x = (z_a + z_b)/2 and v_clean = (z_b − z_a)/h. Locally, v_clean = J∇Ȟ_h(x), with the
small-step inverse modified Hamiltonian expansion Ȟ_h = H + h² H₂ + h⁴ H₄ + …
(even powers because the midpoint generating function is odd in h; Hu et al. 2026, Prop. 2.5).
Test: `test_mp_target_is_second_order_and_richardson_fourth` (slope 2 and 4 for h ≤ 0.065).

Noise enters through the anchor noise U = (e_a + e_b)/2 and the velocity noise W = (e_b − e_a)/h.

## 1. A single MP pair: velocity noise is not a bias source

Cov(U, W) = (Var e_b − Var e_a)/(2h) = 0 whenever both readings have the same noise law (more
generally, E[e_b − e_a | e_a + e_b] = 0 for exchangeable i.i.d. noise). Gradient kernel ridge
regression is linear in v given the anchors, so for any fit that contains **one pair per unit**
(SI, IP, RA, and each stride fit of SA-RICH) the velocity noise adds variance only. The remaining
noise-induced bias is anchor errors-in-variables (EIV): an O(σ²) smoothing of f in the interior of
the sampling density, plus Tweedie-type terms where the density has gradients.
Test: `test_mp_anchor_and_velocity_noise_uncorrelated`.

What breaks this: (i) forward-difference anchoring (anchor = first reading): Cov(U, W) = −Σ/h, an
O(σ²/h) correlation term in the regression target (used as a **positive control**; see §7 for its
Hamiltonian projection); (ii) state-dependent (heteroscedastic) noise:
Cov(U, W) = (σ²(z_b) − σ²(z_a))/(2h) ≈ ½∇σ²·f, an h-independent bias; (iii) fits that contain two
pairs of the same unit whose noises are correlated (below). Stationary temporally correlated noise
does **not** break it (it rescales Var W by (1−ρ) and Var U by (1+ρ)).

## 2. Shared anchor: induced covariances (exact)

SA unit: readings at 0, h1, h2; pairs (0,h1), (0,h2) share e_0.

| quantity | value |
|---|---|
| Cov(W1, W2) | Σ/(h1 h2) |
| Cov(U1, W2) | −Σ/(2 h2) |
| Cov(U2, W1) | −Σ/(2 h1) |
| Cov(U1, U2) | Σ/4 |
| Cov(U1, W1), Cov(U2, W2) | 0 |

Test: `test_shared_anchor_covariances_match_theory`, `test_pair_covariances_sa`.
The cross terms Cov(U1, W2), Cov(U2, W1) are EIV sources only when both pairs enter the **same**
fit (pooled estimators). The positive Cov(W1, W2) reduces the variance of an opposite-sign
combination (Richardson) and increases that of a same-sign one (pooling).

MS3 unit (readings 0, h, 2h; pairs a=(0,h), b=(h,2h), c=(0,2h)): Cov(W_a, W_b) = −Σ/h², W_c = (W_a+W_b)/2
exactly, Cov(U_a, W_b) = −Σ/(2h), Cov(U_b, W_a) = +Σ/(2h): cross-pair EIV appears inside the stride-1 fit.

## 3. Richardson for any interval ratio

With r = h2/h1, the weights α1 = r²/(r²−1), α2 = −1/(r²−1) cancel the h² term of the **clean
theoretical targets evaluated at the same x**. The residual Hamiltonian term is −h1² h2² H₄,
giving O(h⁴) vector-field error for fixed r and smooth targets in the small-step regime.

This does not guarantee O(h⁴) error for regularized finite-sample fits f̂_1, f̂_2. Flow invariance
preserves the distribution of z(t), including the temporal midpoint z(t+h/2), but not generally that
of the arithmetic midpoint (z(t)+z(t+h))/2. Its distribution can depend on h; the noisy anchor
distribution can therefore also differ between strides. Shared (ℓ, η), a per-sample-normalized
ridge and comparable n do **not** make the fitted smoothing operators identical: their anchors,
Gram matrices and regularization scales can differ. If both fits did apply the same linear
operator S, it would preserve cancellation of the h² target term, while leaving S f − f and any
common EIV term because α1+α2 = 1. In the actual fits, incomplete cancellation, attenuation,
regularization and sampling error remain empirical questions; common leading attenuation is an
approximation, not a consequence of flow invariance. The slope test above concerns clean targets,
not KRR convergence rates.
On the double pendulum the O(h⁴) regime holds only for h ≲ 0.05–0.1; at h = 0.2 Richardson reduces
the pointwise MP error 1.7% → 0.5%, at h = 0.4 only 5.3% → 2.9% (higher-order terms).

## 4. Local-averaging variance factors (idealization)

If the smoother's bandwidth is large compared with within-unit anchor separations, each estimator
reduces per unit to a fixed combination c^T e of the unit's reading noise. Variance per unit
budget, in units of σ²/(h1² B), r = 2 (`hamsched.theory.variance_factor`):

| plan / estimator | factor | comment |
|---|---|---|
| SI(h1) | 4.0 | |
| SI(2 h1) | 1.0 | best variance; most discretization bias |
| IP-POOL | 1.6 | endpoint-defect weighting = inverse-variance for independent pairs |
| SA-POOL (OLS) | 1.68 | positive Cov(W1,W2) hurts a same-sign combination |
| SA-POOL (GLS) | 1.5 | GLS puts all weight on the long pair: the middle reading carries no slope information |
| IP-RICH = RA-RICH | 14.4 | independent noise; RA has half the starts of IP |
| SA-RICH | 9.5 | 4/3 more units than IP (14.4 → 10.8) and favourable correlation (→ 9.5) |
| MS3-RICH | 1.5 | stride-1 average telescopes to (e_2h − e_0)/(2h) = stride-2 noise: no amplification |

The MS3-RICH value is optimistic: when the smoother resolves the within-unit anchor shift |f|h,
an excess O((|f|h/ℓ)²) appears. Hu et al.'s Sec. 6.3 protocol (independent noise per stride)
would give 0.944 instead of 0.5 per unit (σ²/h²) for the same MS3 combination, i.e. it overstates
the noise cost of multistride Richardson by ≈1.9× relative to truly shared readings.

**Predictions** (pre-registered, tested in the experiments):
SA-RICH beats IP-RICH at identical (h1,h2); pooled two-interval fits do not beat a tuned SI;
MS3-RICH may dominate SA-RICH on the same readings; a single-interval defect correction may remove
much of the Richardson advantage; without noise the SI with small h wins (more starts, no noise).

## 5. Error model and crossover

Per plan, MSE(h) ≈ F(n_starts) + β h⁴ + c σ²/(h² B) (single interval, MP) or + γ h⁸ (Richardson,
asymptotically). Optimal-h errors scale as (σ²/B)^{2/3} and (σ²/B)^{4/5}: Richardson can only win
when σ²/B is small enough that the tuned SI is forced to an h where its bias exceeds the
Richardson variance penalty, and the floor difference F(B/3) − F(B/2) (fewer starts for 3-reading
plans) does not dominate. Phase 0 measures F(n) directly (exact-gradient fits).

## 6. Defect correction (single interval)

Given a fit Ĥ0 ≈ Ȟ_h, the model's own MP defect at anchor x, evaluated at the **chord midpoint**,
d = (z1′ − z0′)/h − J∇Ĥ0((z0′+z1′)/2) with z0′ = φ^{Ĥ0}_{−h/2}(x), z1′ = φ^{Ĥ0}_{+h/2}(x), estimates
J∇(Ȟ_h − H) up to O(h⁴ + h² ε₃) with ε₃ the error of Ĥ0's third derivatives. Evaluating the
model field at x instead of the chord midpoint leaves an O(h²) error of the same order as the
defect (the anchor-shift trap): `test_defect_correction_with_true_flow_is_fourth_order` checks both.

## 7. Orthogonality of the single-pair forward-anchor correlation term (found during the study)

For a single pair write x̃ = x + U and v = f_h(x) + W, with centered jointly Gaussian (U, W)
independent of the clean state x. Let Σ_U = Cov(U), C = Cov(W, U), and let ρ̃ be the noisy anchor
density. Tweedie's identity and Gaussian conditioning give the **exact** identities

    E[U | x̃] = −Σ_U ∇log ρ̃(x̃),
    E[W | x̃] = C Σ_U⁻¹ E[U | x̃] = −C ∇log ρ̃(x̃).

Thus the regression bias splits into:

* (a) the *correlation term* E[W | x̃] = −C ∇log ρ̃(x̃). For MP anchoring C = 0. For forward
  anchoring U = e₀ and W = (e₁−e₀)/h, so C = −Σ/h and the term is **+(Σ/h)∇log ρ̃(x̃)**.
* (b) the remaining *attenuation/EIV term* E[f_h(x̃−U) | x̃] − f_h(x̃). For small Gaussian noise
  and sufficiently smooth f_h and density, its expansion is
  (∇f_h)Σ_U∇log ρ̃ + ½ tr(Σ_U∇²f_h) + O(σ⁴).

With isotropic Gaussian reading noise Σ = σ²I, the forward-anchor term (a) is the gradient field
+(σ²/h)∇log ρ̃. For a Hamiltonian field J∇H with the required integrability and vanishing boundary terms,

    ∫ ρ̃ ∇log ρ̃ · J∇H = ∫ ∇ρ̃ · J∇H = −∫ ρ̃ div(J∇H) = 0.

Here J is the constant canonical symplectic matrix, so div(J∇H) = 0. This proves L²(ρ̃)
orthogonality of **this single-pair correlation term** to Hamiltonian fields. It contributes nothing
to the population least-squares projection onto that class. It does not prove that general
measurement-error bias vanishes, that a finite-sample fit removes the term exactly, or that pooled
cross-pair effects (including SA-POOL and the MS3 stride-1 fit in §2) are eliminated.

Term (b) need not be orthogonal: for isotropic Σ_U, its Laplacian part is itself Hamiltonian.
It is O(σ²) without the 1/h factor at fixed h, and its score part is zero only where ρ̃ is flat.
A flat region of the clean density does not guarantee exact flatness after Gaussian convolution.
Anisotropic noise and state-dependent noise also fall outside the stated orthogonality result.

In the synthetic Gaussian-anchor example (`tests/test_simex_bias_removal.py`),
∇log ρ̃(x̃) = −x̃/(1+σ²), so the corrected **positive** score prefactor gives the **negative** radial
coefficient −σ²/[h(1+σ²)] ≈ −0.41 at h = 0.2, σ = 0.3. The reported unstructured x-bias −0.45
and near-zero Hamiltonian x-component are consistent with this sign; a near-zero x-component is
not a test that attenuation vanishes. The test's explanatory docstring repeats the sign typo,
but its data generation, assertions and executable SIMEX conditional map use the correct sign;
no numerical result needs a sign reversal.

For the double-pendulum decompositions, **no remaining measurement-error bias was detectable above
Monte Carlo uncertainty in the tested settings**. Single-pair orthogonality can explain the absence
of the forward-anchor correlation component, but does not establish that statement for the total
bias or for pooled fits; those are empirical findings.
