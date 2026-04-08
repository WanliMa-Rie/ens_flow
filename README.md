# EnsFlow: A Three-Level SE(3) Flow Framework for RNA Conformational Ensembles

EnsFlow is a three-level hierarchical model for RNA backbone ensemble generation
built on SE(3) flow matching. Each level strictly nests the previous one:

| Level | Forward process | Learnables | Supervision |
|-------|-----------------|------------|-------------|
| **1** | Deterministic geodesic on $\mathrm{SE}(3)^L$ | Drift only (FlowModel) | Flow-matching MSE |
| **2** | Tangent-space-lifted stochastic bridge on $\mathrm{SE}(3)^L$ with **fixed** $(\bar\sigma_x, \bar\sigma_\omega)$ | Drift only (FlowModel) | Flow-matching MSE (identical to L1) |
| **3** | Bridge with per-residue widths $\sigma_{x,i} = \bar\sigma_x\sqrt{\nu_i},\ \sigma_{\omega,i} = \bar\sigma_\omega\sqrt{\nu_i}$ | FlowModel + sequence-only `FlexibilityNet` | L1 loss + B-factor alignment |

**Strict nesting.** Level 2 → Level 1 when $(\bar\sigma_x, \bar\sigma_\omega) = (0, 0)$.
Level 3 → Level 2 when $\nu_i \equiv 1$ (achieved at initialization via a zero-weight
+ `log(e−1)` bias head, verified byte-for-byte in the smoke tests).

The three levels are selected by the single config field `stochastic_bridge.level`.
They share the same `FlowModel` drift network, data pipeline, and flow-matching
loss — the only differences are what gets added to the corruption path and
whether a separate flexibility head is trained.

## Method

### Level 1 — SE(3) Flow Matching

Each nucleotide is a rigid frame $(x_i, R_i) \in \mathrm{SE}(3)$ (C4$'$ translation +
rotation). Flow matching draws $t \sim \mathcal{U}(\text{min}\_t, 1-\text{min}\_t)$,
interpolates translations linearly (with optimal-transport alignment) and
rotations along the SO(3) geodesic, and regresses the velocity field predicted
by the `FlowModel`:

$$u^x(i,t) = \frac{\hat x_{i,1|t} - x_{i,t}}{1 - t}, \qquad u^\omega(i,t) = \frac{1}{1-t}\mathrm{Log}(R_{i,t}^\top \hat R_{i,1|t})^\vee$$

Loss terms (all also reused verbatim at Levels 2 and 3): translation VF MSE,
rotation VF MSE, backbone atom MSE on $[\text{C3}', \text{C4}', \text{O4}']$,
pairwise distance MSE, and torsion angle MSE. Inference is plain Euler ODE.

### Level 2 — Tangent-Space-Lifted Stochastic Bridge

Level 2 upgrades the deterministic geodesic to a *stochastic bridge* with
closed-form sampling and strict endpoint constraints. For translations this is
an ordinary Euclidean Brownian bridge:

$$x_{i,t} = (1-t) x_{i,0} + t\,x_{i,1} + \bar\sigma_x \sqrt{t(1-t)}\,\varepsilon_i^x, \qquad \varepsilon_i^x \sim \mathcal{N}(0, I_3)$$

For rotations we lift a Brownian bridge on $\mathbb{R}^3 \cong \mathfrak{so}(3)$
and push forward through the exponential map at $R_{i,0}$:

$$\omega_i = \mathrm{Log}(R_{i,0}^\top R_{i,1})^\vee, \quad \Omega_{i,t} = t\,\omega_i + \bar\sigma_\omega \sqrt{t(1-t)}\,\varepsilon_i^\omega, \quad R_{i,t} = R_{i,0}\cdot\mathrm{Exp}(\hat\Omega_{i,t})$$

Both endpoints are respected exactly ($R_{i,0}$ at $t=0$, $R_{i,1}$ at $t=1$).
The velocity target is the same simplified form as Level 1, so the velocity
MSE loss and the full auxiliary loss stack are **byte-for-byte identical** to
Level 1 — the only change is in `corrupt_batch`. Zero new parameters.

Inference is Euler–Maruyama on both translations and rotations with the fixed
$(\bar\sigma_x, \bar\sigma_\omega)$ from the config.

See `docs/architecture_report.md §2` for the full derivation including the
$\sigma = 0$ degeneration proof and the cut-locus / $t \to 1$ caveats.

### Level 3 — Flexibility-Modulated Bridge with B-factor Anchoring

Level 3 replaces the global $(\bar\sigma_x, \bar\sigma_\omega)$ with a
**single** sequence-conditioned per-residue multiplier $\nu_i(s) > 0$ that
scales both translation and rotation widths jointly:

$$\sigma_{x,i} = \bar\sigma_x\sqrt{\nu_i}, \qquad \sigma_{\omega,i} = \bar\sigma_\omega\sqrt{\nu_i}$$

**Why a single scalar.** The supervision is a scalar B-factor per residue;
two independent width heads would be unidentifiable (translation-large /
rotation-small vs. the reverse give the same residue-level uncertainty). A
single jointly-scaling scalar resolves this cleanly.

**Flexibility head.** `FlexibilityNet` is a 2-layer MLP over the dataset's
`single_embedding` ($c_{\text{single\_in}} = 384$), followed by softplus + $\varepsilon$
and per-sample mean-1 normalization:

$$a_i = \mathrm{softplus}(g_\psi(s_i)) + \varepsilon, \qquad \nu_i = \frac{a_i}{\frac{1}{L}\sum_k a_k}$$

The final `Linear` is zero-initialized with bias $\log(e-1)$, so
$\mathrm{softplus}(\mathrm{bias}) = 1$ and the mean-1 normalization gives
$\nu_i \equiv 1$ at step 0. Level 3 therefore **starts byte-identical to
Level 2** and can be warm-started from any Level 2 checkpoint.

**B-factor alignment loss.** Using $\widehat B_i \propto (3\bar\sigma_x^2 + 2c_i\bar\sigma_\omega^2)\nu_i$
from the residue-wise MSD of the bridge, with $c_i \approx 1.473$ Å$^2$
constant across all four RNA frame types (A/U/G/C differ by $< 10^{-3}$ Å$^2$),
the per-sample mean-1 normalization $\hat b_i = B_i / \bar B$ reduces the
target to exactly $\nu_i$:

$$\mathcal{L}_B = \frac{1}{L}\sum_{i=1}^{L}(\log \nu_i - \log \hat b_i)^2$$

A weak log-space regularizer $\mathcal{L}_\nu = \frac{1}{L}\sum_i (\log \nu_i)^2$
discourages runaway concentration.

**Gradient routing.** The bridge sampling path uses `nu.detach()`, so
$\mathcal{L}_{\text{bridge}}$ updates only the `FlowModel` and
$\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$ updates only the `FlexibilityNet`.
This prevents the velocity loss from silently shrinking $\nu$ to reduce the
MSE residual. A two-stage training protocol is supported via
`stochastic_bridge.training_phase`:
- `psi_only`: freeze `FlowModel`, train only `FlexibilityNet` on the B-factor
  losses (warm-start after a Level 2 pretrain).
- `full`: joint training with the stop-grad routing above.

**Inference.** `sample_bridge` computes $\nu$ once per sample from the
`single_embedding` and runs per-residue Euler–Maruyama with
$\bar\sigma\sqrt{\nu_i}$ as the diffusion amplitude.

Full derivation, the identifiability argument, strict nesting conditions, and
data-side B-factor gating details are in `docs/architecture_report.md §3`.

## Model Architecture

```
Input batch: trans_t [B, L, 3], rotmats_t [B, L, 3, 3], t [B, 1],
             single_embedding [B, L, 384], pair_embedding [B, L, L, D]
  │
  ├── FlowModel (Levels 1–3) ────────────────────────────────────────────┐
  │     NodeEmbedder (pos + timestep + single → 256)                     │
  │     EdgeEmbedder (distogram + relpos + pair → 128)                   │
  │     5× IPA blocks { IPA + LN + TransformerEncoder                    │
  │                    + StructureModuleTransition + BackboneUpdate      │
  │                    + EdgeTransition (all but last) }                 │
  │     TorsionAngleHead (8 RNA torsions, sin/cos)                       │
  │     → pred_trans, pred_rotmats, pred_torsions                        │
  │                                                                      │
  └── FlexibilityNet (Level 3 only) ────────────────────────────────────┐
        LayerNorm → Linear(384 → 128) → SiLU → Linear(128 → 1)          │
        softplus + ε → mean-1 normalization over residues               │
        → ν_i ∈ (0, ∞), with mean_i(ν_i) = 1 per sample                 │
```

The `FlowModel` uses Invariant Point Attention from AlphaFold2 / FrameFlow.
The `FlexibilityNet` adds ~50k parameters; it is not used at Levels 1 or 2.

## Configuration

Three levels are selected by a single field in `configs/config.yaml`:

```yaml
stochastic_bridge:
  level: 2                   # 1, 2, or 3

  # Used at Levels 2 and 3 (the global bridge scale).
  bridge:
    sigma_x: 2.0             # Å
    sigma_omega: 0.25        # rad

  # Level 3 only.
  flexibility:
    hidden_dim: 128
    eps: 1.0e-4
  bfactor:
    enabled: true
    lambda_b: 1.0
    lambda_nu: 0.01
    min_valid_bfactor: 0.0   # residues with b ≤ this are dropped from L_B
  training_phase: full       # "full" or "psi_only"
```

See `configs/config.yaml` for the full schema (data paths, model dims, optimizer,
checkpointer).

## Project Structure

```
ens_flow/
├── configs/
│   ├── config.yaml                   # Training configuration (all three levels)
│   └── inference.yaml                # Inference configuration
├── docs/
│   ├── architecture_report.md        # Canonical method spec (Levels 1–3)
│   ├── metrics.md                    # Evaluation metrics definitions
│   └── paper/                        # NeurIPS paper draft (see NOTE below)
├── rna_backbone_design/
│   ├── data/
│   │   ├── interpolant.py            # Corruption + sampling (all three levels)
│   │   ├── so3_utils.py              # SO(3) exp/log/geodesic primitives
│   │   ├── build_datasets.py         # Dataset construction (incl. b_factors)
│   │   ├── rna_conformer_dataset.py  # Sample-level and cluster-level datasets
│   │   ├── rna_conformer_datamodule.py
│   │   ├── all_atom.py               # Frame → atom37 rebuild
│   │   ├── nucleotide_constants.py
│   │   └── utils.py
│   ├── models/
│   │   ├── flow_module.py            # LightningModule: train/val/predict steps
│   │   ├── flow_model.py             # FlowModel: IPA trunk + torsion head
│   │   ├── flexibility_net.py        # Level 3: sequence-only ν_i head
│   │   ├── ipa_pytorch.py            # Invariant Point Attention
│   │   ├── node_embedder.py
│   │   ├── edge_embedder.py
│   │   └── torsion_net.py
│   └── analysis/
│       ├── metrics.py                # RMSD, TM-score, ensemble coverage
│       └── utils.py                  # PDB writing, alignment
├── train_se3_flows.py                # Training entry point
├── inference_se3_flows.py            # Inference script
├── eval_se3_flows.py                 # Evaluation script
└── pyproject.toml
```

## Training

```bash
python train_se3_flows.py
```

The same entry point handles all three levels — set `stochastic_bridge.level`
in the config. Key knobs:

| Field | Default | Notes |
|---|---|---|
| `interpolant.sampling.num_timesteps` | 60 | Integration steps during sampling |
| `interpolant.rots.sample_schedule` | `exp` (rate 10) | Rotation sampling schedule |
| `interpolant.min_t` | 0.01 | Avoids drift singularity at $t \to 1$ |
| `model.ipa.num_blocks` | 5 | IPA trunk depth |
| `model.node_embed_size` | 256 | |
| `experiment.optimizer.lr` | 1e-4 | AdamW |
| `experiment.training.translation_loss_weight` | 2.0 | |
| `experiment.training.aux_loss_t_pass` | 0.25 | Auxiliary-loss time gate |

Level 3 additionally reads `stochastic_bridge.bfactor.lambda_b`,
`lambda_nu`, and `training_phase`.

## Inference

```bash
python inference_se3_flows.py
```

Level 1 runs Euler ODE; Levels 2 and 3 run Euler–Maruyama. Level 3 computes
`ν` from the `single_embedding` once per sample and uses it in the per-residue
diffusion amplitudes.

## Evaluation Metrics

Ensemble metrics follow the definitions in `docs/metrics.md`:

- **Coverage Recall ($\mathrm{COV\text{-}R}_\delta$)** — fraction of ground-truth
  conformers covered by at least one generated sample within RMSD $\delta$.
- **Average Minimum RMSD Recall (AMR-R)** — mean RMSD from each GT conformer
  to its nearest generated sample.
- **Coverage Precision ($\mathrm{COV\text{-}P}_\delta$)** — fraction of generated
  samples that match at least one GT within $\delta$.
- **Average Minimum RMSD Precision (AMR-P)** — mean RMSD from each generated
  conformer to its nearest GT conformer.
- **Pairwise RMSD** — intra-ensemble diversity (guards against mode collapse).

Single-structure metrics: C4$'$ RMSD and TM-score against the best-matching
ground-truth conformer.

## Note on the Paper Draft

`docs/paper/main.tex` is a NeurIPS draft that currently describes an **earlier**
version of Level 2 (self-regulating NLL on drift residuals via a `BridgeWidthNet`)
and an earlier version of Level 3 (Feynman-Kac B-factor anchoring via
`AmplitudeNet` / `UncertaintyNet`). The code no longer implements either of
these. The Method section of the paper needs to be rewritten against the current
design in `docs/architecture_report.md` before submission.

## Acknowledgements

The base codebase is adapted from
[RNA-FrameFlow](https://github.com/rish-16/rna-backbone-design) and
[Protein-FrameFlow](https://github.com/microsoft/protein-frame-flow).
The IPA implementation follows
[AlphaFold2](https://github.com/google-deepmind/alphafold).
