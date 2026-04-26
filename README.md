# EnsFlow Level 3 — Flexibility-Modulated SE(3) Bridge with B-factor Anchoring

EnsFlow Level 3 generates RNA backbone **ensembles** by coupling an SE(3) flow-matching
drift with a per-residue flexibility field $\nu_i$ that is supervised by crystallographic
B-factors. It strictly extends Level 2 (tangent-space-lifted stochastic bridge) and
reduces to Level 2 exactly when $\nu_i \equiv 1$.

This README documents the method, configuration, and usage for **Level 3**. For the
shared SE(3) flow-matching backbone (Level 1) and the un-modulated bridge (Level 2),
see [docs/architecture_report.md](docs/architecture_report.md).

## Method

### Shared backbone (inherited from Levels 1 and 2)

Each nucleotide is a rigid frame $(x_i, R_i) \in \mathrm{SE}(3)$ (C4$'$ translation +
rotation). A `FlowModel` (IPA trunk from AlphaFold2 / FrameFlow + a torsion head for
the 8 RNA torsions) predicts the clean frame given the corrupted frame at time $t$
and the `(single, pair)` embeddings. The training loss is the SE(3) flow-matching
velocity MSE plus auxiliary terms (backbone atom MSE on $[\text{C3}', \text{C4}',
\text{O4}']$, pairwise distance MSE, torsion MSE); this loss stack is byte-for-byte
shared across all three levels.

### Level 3 forward corruption — a flexibility-modulated bridge

Level 3 replaces the Level 2 global widths $(\bar\sigma_x, \bar\sigma_\omega)$ with
**one** sequence-conditioned per-residue scalar $\nu_i > 0$ that scales both
translation and rotation widths jointly:

$$\sigma_{x,i} = \bar\sigma_x\sqrt{\nu_i}, \qquad \sigma_{\omega,i} = \bar\sigma_\omega\sqrt{\nu_i}$$

The strict Brownian-bridge envelope is:

$$\text{std}_{x,i}(t)^2 = \bar\sigma_x^2\,\nu_i\,t(1-t)$$

$$\text{std}_{\omega,i}(t)^2 = \bar\sigma_\omega^2\,\nu_i\,t(1-t)$$

Translations corrupt as a Euclidean Brownian bridge; rotations as a tangent-space-lifted
Brownian bridge on $\mathrm{SO}(3)$ pushed forward through $\mathrm{Exp}$ at $R_{i,0}$.
Both endpoints ($R_{i,0}$ at $t=0$, $R_{i,1}$ at $t=1$) are respected exactly.

**Why a single scalar.** Supervision is a scalar B-factor per residue; two independent
width heads would be unidentifiable (translation-large / rotation-small and the reverse
give the same residue-level uncertainty). A single jointly-scaling scalar resolves
this cleanly.

### FlexibilityNet — pair-only head

Flexibility is a **relational** property (who residue $i$ is packed / paired with),
so `FlexibilityNet` is a pair-only head implemented in
[rna_backbone_design/models/flexibility_net.py](rna_backbone_design/models/flexibility_net.py):

```
pair_embedding [B, N, N, c_pair]
  → LayerNorm
  → pool over j:  [mean_j ,  max_j ,  diag (self)]   (3 × c_pair)
  → MLP(GELU + Dropout × 2)
  → Linear(hidden → 1) → clamp(-3, 3) → softplus
  → ν_i ∈ (≈0.05, ≈e³), ν_i = 1 at masked-out positions
```

**Strict L2 ⊂ L3 nesting at init.** The final `Linear` has **zero weights** and
bias $\log(e-1)$, so $\mathrm{softplus}(\mathrm{bias}) = 1$ and $\nu_i \equiv 1$
at step 0 regardless of the upstream MLP. Level 3 therefore starts byte-identical
to Level 2 and can be warm-started from any Level 2 checkpoint.

No per-sample mean-1 normalization in the forward pass: the B-factor loss is
itself defined against a **per-chain normalized** target (see below), which pins
the scale.

### B-factor loss

Level 3 supervises $\nu$ directly in log-space against a per-chain normalized
B-factor. For each sample, let $\bar B$ be the mean B-factor over the
**valid-B** residues (resolved **and** $B_i > \text{min\_valid\_b}$, default
$1\,\text{Å}^2$, which drops near-zero crystallographic artifacts that would
dominate $\mathcal{L}_B$ in log-space). Define

$$\hat b_i = \log\!\left(\frac{B_i}{\bar B}\right), \qquad \hat b_i \leftarrow \mathrm{clamp}(\hat b_i, -2, 2)$$

The clamp (target $\nu_i \in [0.14, 7.4]$) prevents a single crystallographic
outlier from dragging $\nu$ to absurd scales. The loss averages over valid-B
residues only:

$$\mathcal{L}_B = \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}\left(\log\nu_i - \hat b_i\right)^2$$

See [rna_backbone_design/models/flow_module.py:480-506](rna_backbone_design/models/flow_module.py#L480-L506).

### Sigma warmup (phase-in schedule)

Level 3 training starts from pure flow matching (Level 1) and **linearly warms**
$\sigma_\text{scale}$ from 0 → 1 across `[warmup_start_step, warmup_end_step]`:

- Before `warmup_start`: $\sigma_\text{scale} = 0$ → deterministic geodesic
  (Level 1), FlexibilityNet is **not** queried, no $\mathcal{L}_B$ is added.
- After `warmup_end`: $\sigma_\text{scale} = 1$ → full bridge noise with
  B-factor supervision active.

All three features — per-residue bridge noise, FlexibilityNet forward, and
$\mathcal{L}_B$ — turn on together at `warmup_start`, which stabilizes the
early steepest-descent regime.

### Gradient routing and two-optimizer training

At Level 3 the bridge corruption uses `nu.detach()`, so the velocity-MSE loss
$\mathcal{L}_\text{bridge}$ updates **only** the `FlowModel`, and $\mathcal{L}_B$
updates **only** the `FlexibilityNet`. This prevents the velocity loss from
silently shrinking $\nu$ to reduce its MSE residual.

Training is **manual optimization** with two independent AdamW optimizers and
independent gradient clipping (`clip_grad_norm` = 1.0 each). This keeps
$\mathcal{L}_B$ on its own gradient budget — the FlowModel's global clip cannot
throttle the B-factor learning signal, and vice versa. See
[rna_backbone_design/models/flow_module.py:604-634](rna_backbone_design/models/flow_module.py#L604-L634).

### Inference — bridge-consistent Euler–Maruyama with $\nu$

The sampler is selected automatically from `level` (Level 1 → ODE,
Level 2/3 → SDE); override with `inference.interpolant.use_sde` for ablations.
At Level 3, `FlexibilityNet` is queried **once per sample** on the cluster's
`pair_embedding` (with an all-ones mask, since inference has no "unresolved"
concept) and then broadcast across the `num_generated` conformers of that
cluster. The per-residue diffusion amplitude is
$\bar\sigma\sqrt{\nu_i\,t(1-t)}$ — matched exactly to
the training bridge envelope per unit time.

A diagnostic mode `inference.dump_nu_only: true` writes the predicted
$\nu_i$ (and ground-truth $B_i$ if available) to `nu.csv` per cluster and
skips structure generation.

## Configuration (Level 3)

Level 3 is selected by the single field [`stochastic_bridge.level: 3`](configs/config.yaml#L91)
in [configs/config.yaml](configs/config.yaml):

```yaml
stochastic_bridge:
  level: 3

  bridge:
    sigma_x:         0.1       # Å   — global translation width
    sigma_omega:     0.0125    # rad — global rotation width
    warmup_start_step: 10000   # sigma phase-in begins
    warmup_end_step:   20000   # sigma phase-in ends (L3 fully active)

  flexibility:
    hidden_dim: 64             # FlexibilityNet MLP width
    dropout:    0.1

  bfactor:
    enabled:     true          # false → disable B-factor supervision
    lambda_b:    1.0           # weight on L_B
    min_valid_b: 1.0           # Å² — drops near-zero B artifacts from L_B
```

Optimizer config (Level 3 uses two AdamW groups):

```yaml
experiment:
  optimizer:
    lr: 0.0001
    weight_decay:      0.01    # FlowModel
    flex_weight_decay: 0.1     # FlexibilityNet (popped before AdamW init)
```

## Model architecture

```
Input batch:
  trans_t        [B, L, 3]
  rotmats_t      [B, L, 3, 3]
  t              [B, 1]
  single_embed   [B, L, 384]
  pair_embed     [B, L, L, 128]

┌─ FlowModel (shared L1/L2/L3) ──────────────────────────────────────┐
│  NodeEmbedder (pos + timestep + single → 256)                      │
│  EdgeEmbedder (distogram + relpos + pair → 128)                    │
│  N× IPA blocks { IPA + LN + TransformerEncoder                     │
│                 + StructureModuleTransition + BackboneUpdate       │
│                 + EdgeTransition (all but last) }                  │
│  TorsionAngleHead (8 RNA torsions, sin/cos)                        │
│  → pred_trans, pred_rotmats, pred_torsions                         │
└────────────────────────────────────────────────────────────────────┘

┌─ FlexibilityNet (Level 3 only, ≈50k params) ───────────────────────┐
│  LayerNorm(pair_embed)                                             │
│  pool_j = [mean_j, max_j, diag]   → [B, L, 3·c_pair]               │
│  MLP: Linear → GELU → Dropout → Linear → GELU → Dropout            │
│  head: Linear(hidden → 1), zero-init W, bias = log(e-1)            │
│  → clamp(-3, 3) → softplus                                         │
│  → ν_i ∈ (≈0.05, ≈e³), ν ≡ 1 at init                               │
└────────────────────────────────────────────────────────────────────┘
```

## Project structure

```
ens_flow/
├── configs/
│   ├── config.yaml                  # Training config (Level set by stochastic_bridge.level)
│   └── inference.yaml               # Inference config
├── docs/
│   ├── architecture_report.md       # Canonical method spec (all 3 levels)
│   ├── metrics.md                   # Evaluation metrics
│   ├── so3_sfm_derivation.md
│   └── sfm_endpoint_loss_derivation.md
├── rna_backbone_design/
│   ├── data/
│   │   ├── interpolant.py           # Corruption + sampling (L1/L2/L3)
│   │   ├── so3_utils.py             # SO(3) exp/log/geodesic
│   │   ├── rna_conformer_dataset.py # Cluster-level + single-sample datasets
│   │   ├── all_atom.py              # Frame → atom37 rebuild
│   │   └── …
│   ├── models/
│   │   ├── flow_module.py           # LightningModule (training + val + predict)
│   │   ├── flow_model.py            # FlowModel: IPA trunk + torsion head
│   │   ├── flexibility_net.py       # Level 3: pair-only ν_i head
│   │   ├── ipa_pytorch.py           # Invariant Point Attention
│   │   └── …
│   └── analysis/
│       ├── metrics.py               # RMSD, TM-score, ensemble coverage
│       └── utils.py                 # PDB/CIF writing, alignment
├── train_se3_flows.py               # Training entry point
├── inference_se3_flows.py           # Inference entry point
└── pyproject.toml
```

Evaluation scripts live in a sibling directory `../eval/` so that this
repo only contains training and inference code:

```
../eval/
├── evaluate_metrics.py     # Table 1 / 2 / 4 metrics
├── score_lociparse.py      # lociPARSE scoring (separate venv)
├── plot_flex_metrics.py    # figures for Table 1 + Table 2
└── stat_tests.py           # paired Wilcoxon + bootstrap CI
```

## Training

```bash
python train_se3_flows.py
```

The same entry point drives all three levels; set `stochastic_bridge.level` in the
config. Key training-time behavior specific to Level 3:

- Two AdamW optimizers (FlowModel + FlexibilityNet), manual optimization.
- `nu` is detached into the bridge corruption path.
- Sigma warmup: pure Level 1 before `warmup_start_step`, linearly blended up to
  full Level 3 by `warmup_end_step`.
- Logged scalars: `train/sigma_scale`, `train/L_B`, `train/loss` (sum of bridge
  losses and L_B), plus all the shared flow-matching losses.

Validation logs `bfactor_norm_log_mae` on the `single` loader: the mean absolute
error between $\log\nu_i$ and $\log(B_i / \bar B)$ on resolved + valid-B residues.

## Inference

```bash
python inference_se3_flows.py
```

Or use [scripts/run_inference.sh](scripts/run_inference.sh). Level 3 queries
`FlexibilityNet` once per cluster, broadcasts $\nu$ across `num_generated`
conformers, and runs the bridge-consistent Euler–Maruyama sampler.

To inspect predicted flexibility without running the SDE:

```yaml
# inference.yaml
inference:
  dump_nu_only: true
```

This writes `<output_dir>/<cluster_id>/nu.csv` with columns `residue, nu, b_factor`
and skips structure generation.

## Evaluation

```bash
python ../eval/evaluate_metrics.py
```

Ensemble metrics (definitions in [docs/metrics.md](docs/metrics.md)):

- **Coverage Recall ($\mathrm{COV\text{-}R}_\delta$)** — fraction of GT conformers
  covered by at least one generated sample within RMSD $\delta$.
- **Average Minimum RMSD Recall (AMR-R)** — mean RMSD from each GT conformer
  to its nearest generated sample.
- **Coverage Precision ($\mathrm{COV\text{-}P}_\delta$)** — fraction of generated
  samples within $\delta$ of some GT conformer.
- **Average Minimum RMSD Precision (AMR-P)** — mean RMSD from each generated
  conformer to its nearest GT conformer.
- **Pairwise RMSD** — intra-ensemble diversity (mode-collapse guard).

Single-structure metrics: C4$'$ RMSD and TM-score against the best-matching GT
conformer.

## Strict nesting — L3 ⊃ L2 ⊃ L1

- **L3 → L2** at $\nu_i \equiv 1$, which holds exactly at init (zero-weight head
  + $\log(e-1)$ bias + softplus).
- **L2 → L1** at $(\bar\sigma_x, \bar\sigma_\omega) = (0, 0)$ and
  $(\text{min\_sigma}_x, \text{min\_sigma}_\omega) = (0, 0)$.
- During the Level 3 sigma warmup, $\sigma_\text{scale} = 0$ gives exact Level 1
  dynamics (FlexibilityNet is not queried and $\mathcal{L}_B$ is not added).

## Acknowledgements

Adapted from [RNA-FrameFlow](https://github.com/rish-16/rna-backbone-design) and
[Protein-FrameFlow](https://github.com/microsoft/protein-frame-flow). The IPA
implementation follows [AlphaFold2](https://github.com/google-deepmind/alphafold).
