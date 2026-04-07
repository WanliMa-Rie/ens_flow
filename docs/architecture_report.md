# Architecture Report: Feynman-Kac B-factor Supervision for SE(3) Flow Matching

## Overview

This report documents the implementation of the Feynman-Kac cumulative uncertainty framework for B-factor supervision in the RNA backbone generative model. The framework replaces the previous isotropic steer losses (L_qv, L_cov) with a mathematically rigorous path-integral formulation that aligns accumulated generative uncertainty with experimental B-factors.

## Design Principles

1. **Genuine path accumulation** — B-factor is modeled as a total uncertainty budget over the generative trajectory, not a per-step snapshot
2. **Anisotropic SE(3) diffusion** — Rotation noise is direction-dependent (3 independent axes), resolving the O4'/C3' degeneracy inherent in isotropic models
3. **TD-learning style training** — Avoids expensive full rollouts by enforcing temporal consistency between pairs of timesteps
4. **TLS model compatibility** — The parameterization corresponds to the validated crystallographic Translation-Libration-Screw model

## Modified Files

### `rna_backbone_design/models/amplitude_net.py`

**AmplitudeNet** — Output changed from `(σ_x, σ_ω)` [2 scalars/residue] to `(σ_x, s1, s2, s3)` [4 scalars/residue]:

- `σ_x` [B, N]: isotropic translation amplitude (softplus, 1 param)
- `s1, s2, s3` [B, N] each: anisotropic rotation variances along local x/y/z axes (softplus, 3 params)
- Replaced single `head_rots` (Linear → 1) with `head_rots_aniso` (Linear → 3)
- Extracted shared `_extract_features()` method for the backbone computation

**UncertaintyNet** (new class) — Predicts remaining cumulative uncertainty U_j(t):

- Same architecture as AmplitudeNet (FiLM + ResConvBlocks)
- Output: `[B, N, 3]` — one value per frame atom (C4', O4', C3')
- No positivity constraint (values decrease from B_norm at t=0 to 0 at t=1)

### `rna_backbone_design/data/steer_utils.py`

New additions:
- `FRAME_ATOM_LOCAL_COORDS` — Fixed tensor `[3, 3]` with C4'=(0,0,0), O4'=(1.45,0,0), C3'=(-0.38,1.48,0)
- `get_frame_atom_local_coords(device)` — Device-aware accessor
- `compute_frame_atom_trace_aniso(σ_x, s1, s2, s3, device)` → `[B, N, 3]`

The anisotropic trace formula per frame atom j:
```
q_j = 3σ_x² + (a_{j2}² + a_{j3}²)·s1 + (a_{j1}² + a_{j3}²)·s2 + (a_{j1}² + a_{j2}²)·s3
```

Sensitivity coefficients (precomputed from fixed geometry):

| Atom | s1 coeff (rot-x) | s2 coeff (rot-y) | s3 coeff (rot-z) |
|------|-------------------|-------------------|-------------------|
| C4'  | 0                 | 0                 | 0                 |
| O4'  | 0                 | 2.10              | 2.10              |
| C3'  | 2.19              | 0.14              | 2.33              |

Key property: s1 (rotation around x-axis) **only affects C3'**, not O4' (because O4' lies on the x-axis). This breaks the isotropic degeneracy.

Existing functions (`compute_atom_trace`, `compute_local_atom_coords`, etc.) are preserved for backward compatibility.

### `rna_backbone_design/models/flow_module.py`

**`__init__`**: Now instantiates both `AmplitudeNet` and `UncertaintyNet` when steer is enabled.

**`_feynman_kac_step()`** (new method): Computes three Feynman-Kac losses per training step:

1. **L_TD (temporal consistency)**: For two timesteps t1 < t2 along the flow:
   ```
   L_TD = ||U(t1) - U(t2) - q̄·(t2-t1)||²
   ```
   where q̄ = (q(t1) + q(t2))/2 is the trapezoidal approximation. This enforces that the uncertainty consumed between t1 and t2 matches the integral of instantaneous diffusion rate.

2. **L_term (terminal condition)**: U(x_1, t=1) = 0 — no remaining uncertainty at the end.

3. **L_init (initial condition)**: U(x_0, t≈0) = B_norm — total accumulated uncertainty equals normalized experimental B-factor.

Time sampling strategy:
- t1 reuses the batch's already-sampled interpolation time (from `corrupt_batch`)
- t2 is uniformly sampled in (t1, 1)
- The state at t2 is obtained by further interpolating from (x_t1, R_t1) toward (x_1, R_1)

**`configure_optimizers()`**: Three parameter groups — drift model (scaled LR), AmplitudeNet, UncertaintyNet.

### `rna_backbone_design/data/interpolant.py`

**`_rots_euler_maruyama_step()`**: Now accepts both scalar `[B, N]` and vector `[B, N, 3]` rotation amplitudes. When 3D, the noise is element-wise scaled per axis.

**`sample_stochastic()`**: Updated to unpack 4 AmplitudeNet outputs and construct anisotropic noise:
```python
L_omega = torch.stack([s1.sqrt(), s2.sqrt(), s3.sqrt()], dim=-1)  # [B, N, 3]
```

**`rollout_terminal()`**: Same update for anisotropic noise injection.

### `configs/config.yaml`

```yaml
steer:
  enabled: false
  amplitude_net:
    c_single_in: 384
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
  uncertainty_net:       # NEW
    c_single_in: 384
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
  loss:
    td_weight: 1.0       # L_TD weight
    term_weight: 1.0     # L_term weight
    init_weight: 1.0     # L_init weight
  optimizer:
    lr: 0.0003
  drift_lr_scale: 0.1
```

## Data Flow Diagram

```
Training Step:
                                    batch (x_0, x_1, B-factors)
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       corrupt_batch    _feynman_kac_step   model_step
                       (get x_t, t)          │              (flow losses)
                              │        ┌─────┴──────┐
                              ▼        ▼            ▼
                         AmplitudeNet    UncertaintyNet
                    (σ_x,s1,s2,s3)@t1,t2    U@t1,t2,0,1
                              │               │
                              ▼               ▼
                    compute_frame_atom     L_TD + L_term + L_init
                    _trace_aniso (q_j)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         L_flow          L_auxiliary       L_FK
              └───────────────┼───────────────┘
                              ▼
                         total_loss

Sampling:
    x_0 ~ prior ──► Euler-Maruyama loop ──► x_1 (generated structure)
                         │
                    At each step k:
                    ├─ model(x_t, t) → v_θ (drift)
                    ├─ AmplitudeNet(x_t, t) → (σ_x, s1, s2, s3)
                    ├─ x_{k+1} = x_k + Δt·v + √Δt·σ_x·ξ_x
                    └─ R_{k+1} = R_k·exp(Δt·ω + √Δt·L_ω·ξ_ω)
                         where L_ω = diag(√s1, √s2, √s3)
```

## Parameter Count Estimate

| Component | Parameters | Notes |
|-----------|-----------|-------|
| FlowModel (drift) | ~5M | Existing, IPA-based |
| AmplitudeNet | ~270K | 4 ResConvBlocks, hidden=128 |
| UncertaintyNet | ~270K | Same architecture as AmplitudeNet |
| **Total new** | **~540K** | ~10% overhead on top of drift model |

## Training Overhead

Each training step requires:
- 2 AmplitudeNet forward passes (at t1 and t2)
- 4 UncertaintyNet forward passes (at t1, t2, t=0, t=1)
- ~2x overhead compared to vanilla flow matching (no full rollouts needed)

This is significantly cheaper than the previous L_cov loss which required M full trajectory rollouts.

## Key Design Decisions

1. **Frame atoms only**: B-factor supervision targets only C4', O4', C3' (the 3 frame-defining atoms), not all 23 atoms. This is because the SE(3) frame only controls these atoms directly; sidechain atoms depend on torsion angles.

2. **Reuse batch interpolation**: t1 reuses the already-computed noisy state from `corrupt_batch()` rather than generating a fresh interpolation, saving one model forward pass.

3. **No rollouts for training**: The TD-style loss eliminates the need for trajectory rollouts during training. The value function U_φ learns to predict the integral, while AmplitudeNet learns to distribute the budget.

4. **Backward compatible**: The anisotropic `_rots_euler_maruyama_step` auto-detects scalar vs vector input, so existing isotropic code paths still work.

## Critical Design Constraints

### Positivity of B-factor targets

Since q_j = 3σ_x² + ... ≥ 0, the ODE dU/dt = -q implies U is monotonically non-increasing. Combined with U(1) = 0, this forces U(0) ≥ 0 for all atoms. Therefore, B-factor normalization **must preserve positivity** — z-score normalization (mean=0, negative values) is mathematically inconsistent. We use `normalize_b_factors_positive()` which applies log-shift-scale to keep all targets ≥ 0.

### Semi-gradient TD update

The TD loss detaches both U(t2) and q̄ from the gradient graph:
```
td_target = U(t2).detach() + q̄.detach() · Δt
L_TD = ||U(t1) - td_target||²
```
This ensures L_TD trains UncertaintyNet via U(t1) only (standard semi-gradient TD). AmplitudeNet receives gradients only through the non-detached q terms in q̄ via a separate path — it is **not** trained by L_TD in the current formulation. AmplitudeNet's noise schedule is indirectly constrained: if it assigns too much/too little budget, UncertaintyNet cannot simultaneously satisfy L_init (U(0)=B) and L_term (U(1)=0) while being consistent with L_TD.

### Prior state for L_init

The initial condition loss evaluates U at t≈0 where the state is pure noise. The prior is sampled correctly: centered Gaussian translations (scale = NM_TO_ANG_SCALE = 10Å) and uniform SO(3) rotations, matching the actual prior used in `corrupt_batch` and `sample`.
