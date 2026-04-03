# EnsFlow: Ensemble-Aware SE(3) Flow Matching for RNA 3D Backbone Design

## Table of Contents

- [Background](#background)
- [Method Overview](#method-overview)
  - [SE(3) Flow Matching](#se3-flow-matching)
  - [Model Architecture](#model-architecture)
  - [U-STEER: B-Factor Guided Ensemble Steering](#u-steer-b-factor-guided-ensemble-steering)
- [Mathematical Formulation](#mathematical-formulation)
  - [Flow Matching on SE(3)](#flow-matching-on-se3)
  - [Training Losses](#training-losses)
  - [Stochastic Generation via Conditional SDE](#stochastic-generation-via-conditional-sde)
  - [Quadratic-Variation Calibration Loss](#quadratic-variation-calibration-loss)
  - [Terminal Covariance Calibration Loss](#terminal-covariance-calibration-loss)
- [Novelty](#novelty)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)

---

## Background

RNA molecules are not static: they populate ensembles of conformational states that are essential to biological function. Current generative approaches for RNA 3D structure prediction typically produce a single deterministic backbone, ignoring this intrinsic conformational heterogeneity. However, experimentally determined B-factors (atomic displacement parameters) from X-ray crystallography encode rich information about per-residue positional uncertainty, serving as a natural proxy for the degree of conformational flexibility at each position.

**EnsFlow** addresses this gap by extending SE(3) flow matching to ensemble-aware RNA backbone generation. Rather than collapsing onto a single structure, the model learns to generate diverse conformational ensembles whose positional spread is calibrated against experimental B-factor profiles. This is achieved through a novel steering mechanism (**U-STEER**) that modulates stochastic diffusion amplitudes during the generative process, enabling principled control over ensemble diversity while maintaining high-quality individual structures.

The base architecture builds upon [RNA-FrameFlow](https://github.com/rish-16/rna-backbone-design), which adapts protein-centric SE(3) flow matching to RNA backbone design using Invariant Point Attention (IPA) and RNA-specific torsion angle prediction.

---

## Method Overview

### SE(3) Flow Matching

The model learns a conditional vector field that transports samples from a simple prior distribution (centered Gaussian for translations, uniform on SO(3) for rotations) to the data distribution of RNA backbone frames. Each nucleotide is represented as a rigid body in SE(3) = SO(3) x R^3, consisting of a rotation matrix and a translation vector defining the local coordinate frame.

During training, ground-truth frames are corrupted via interpolation:

- **Translations**: Linear interpolation with Optimal Transport (Kabsch alignment) to match noise to ground truth.
- **Rotations**: Geodesic interpolation on SO(3) using the exponential map.

The model predicts the clean endpoint frames from any intermediate noised state, and the loss supervises the implied vector field.

### Model Architecture

```
Input: Noised frames (trans_t, rotmats_t), timestep t, sequence embedding, pair embedding
  |
  v
NodeEmbedder ── timestep + position + single_embedding (384D) ──> node features [B, L, 256]
EdgeEmbedder ── pairwise distances (RBF) + relative position + pair_embedding (128D) ──> edge features [B, L, L, 128]
  |
  v
IPA Trunk (4 blocks, each containing):
  ├── Invariant Point Attention (8 heads, 8 qk_points, 12 v_points)
  ├── LayerNorm + residual
  ├── Transformer Encoder (4 heads, 2 layers)
  ├── StructureModuleTransition
  ├── BackboneUpdate (translation + rotation)
  └── EdgeTransition (except last block)
  |
  v
TorsionAngleHead ── predicts 8 RNA backbone torsion angles
  |
  v
Output: pred_trans [B, L, 3], pred_rotmats [B, L, 3, 3], pred_torsions [B, L, 8, 2]
```

The backbone network uses **Invariant Point Attention (IPA)** from AlphaFold2/FrameFlow, which reasons over both node features and 3D point positions attached to each frame, ensuring SE(3)-equivariant structure updates.

### U-STEER: B-Factor Guided Ensemble Steering

The key extension is the **U-STEER** (Uncertainty-STEERing) mechanism, which converts the deterministic ODE-based generation into a stochastic SDE-based process with B-factor-calibrated diffusion amplitudes:

```
AmplitudeNet A_psi(B_tilde, t):
  Input: normalized B-factors [B, L] + timestep t [B, 1]
    |
    v
  B-factor projection: Linear(1 -> 128)
  Timestep conditioning: sinusoidal -> MLP -> FiLM parameters
    |
    v
  4x ResConvBlock (kernel=5, FiLM-modulated):
    LayerNorm -> Conv1D -> FiLM(t) -> SiLU -> Conv1D -> residual
    |
    v
  Separate output heads:
    head_trans -> Softplus -> a_trans [B, L]  (translation amplitude)
    head_rots  -> Softplus -> a_rots  [B, L]  (rotation amplitude)
```

During stochastic sampling, the Euler-Maruyama integrator uses these per-residue amplitudes to inject calibrated noise:

```
x_{k+1} = x_k + dt * v_theta + sqrt(dt) * a_trans * xi_k       (translations)
R_{k+1} = R_k * exp(dt * omega_theta + sqrt(dt) * a_rots * zeta_k)  (rotations)
```

Residues with high B-factors receive larger diffusion amplitudes, naturally producing greater conformational spread at flexible positions.

---

## Mathematical Formulation

### Flow Matching on SE(3)

Each RNA backbone of length $L$ is represented as a collection of rigid frames:

$$G = (R^{(1)}, x^{(1)}, \ldots, R^{(L)}, x^{(L)}) \in \text{SE}(3)^L$$

where $R^{(i)} \in \text{SO}(3)$ and $x^{(i)} \in \mathbb{R}^3$ are the rotation and translation of nucleotide $i$.

**Prior sampling:**

$$x_0^{(i)} \sim \mathcal{N}(0, I_3), \quad R_0^{(i)} \sim \text{Uniform}(\text{SO}(3))$$

**Interpolation (corruption) at time $t \in [0, 1]$:**

$$x_t = (1 - t) \cdot x_0 + t \cdot x_1$$

$$R_t = \text{Exp}\bigl(t \cdot \text{Log}(R_1 R_0^{-1})\bigr) \cdot R_0$$

where translations use Optimal Transport alignment and rotations use geodesic interpolation on SO(3).

### Training Losses

The total training loss is:

$$\mathcal{L} = \mathcal{L}\_{\text{SE(3)-VF}} + \mathcal{L}\_{\text{aux}} + \lambda\_{\text{qv}} \mathcal{L}\_{\text{qv}} + \lambda\_{\text{cov}} \mathcal{L}\_{\text{cov}}$$

**SE(3) Vector Field Loss** (main):

$$\mathcal{L}\_{\text{SE(3)-VF}} = \mathcal{L}\_{\text{trans}} + \mathcal{L}\_{\text{rots}}$$

where:

$$\mathcal{L}\_{\text{trans}} = w\_{\text{trans}} \cdot \frac{1}{|\mathcal{M}|} \sum\_{i \in \mathcal{M}} \left\| \frac{(\hat{x}\_1^{(i)} - x\_1^{(i)}) \cdot s\_{\text{trans}}}{1 - \min(t, 0.9)} \right\|^2$$

$$\mathcal{L}\_{\text{rots}} = w\_{\text{rots}} \cdot \frac{1}{|\mathcal{M}|} \sum\_{i \in \mathcal{M}} \left\| \frac{\text{Log}(R\_t^{(i)\top} \hat{R}\_1^{(i)}) - \text{Log}(R\_t^{(i)\top} R\_1^{(i)})}{1 - \min(t, 0.9)} \right\|^2$$

The factor $1/(1 - \min(t, 0.9))$ normalizes by the remaining interpolation distance, preventing loss vanishing near $t = 1$.

**Auxiliary Losses** (applied only when $t > 0.25$):

$$\mathcal{L}\_{\text{aux}} = \mathcal{L}\_{\text{bb-atom}} + \mathcal{L}\_{\text{dist-mat}} + \mathcal{L}\_{\text{torsion}}$$

- $\mathcal{L}\_{\text{bb-atom}}$: MSE on backbone atom positions (C3', C4', O4')
- $\mathcal{L}\_{\text{dist-mat}}$: Pairwise distance matrix preservation
- $\mathcal{L}\_{\text{torsion}}$: 8 RNA backbone torsion angles (sin/cos representation)

### Stochastic Generation via Conditional SDE

U-STEER extends the deterministic flow ODE to a Stratonovich SDE:

$$dG_t = V\_\theta(G_t, t, c)\,dt + \Sigma\_{\theta,\psi}(G_t, t, c, \widetilde{B}) \circ dW_t, \quad G_t \in \text{SE}(3)^L$$

where $V_\theta$ is the learned drift (vector field), $\Sigma\_{\theta,\psi}$ is the B-factor-modulated diffusion term, $c$ is the sequence condition, and $\widetilde{B}$ is the normalized B-factor profile.

Projecting onto the translation channel $X_t = \Pi_x(G_t) \in \mathbb{R}^{3L}$ and converting to Ito form:

$$dX_t = b_t\,dt + \sigma_t\,dW_t$$

The diffusion tensor is $D_t = \sigma_t \sigma_t^\top \in \mathbb{R}^{3L \times 3L}$, with per-residue block $D\_{x,i}(t) \in \mathbb{R}^{3 \times 3}$.

**B-factor normalization:**

$$\widetilde{B}\_i = \text{Norm}\bigl(\log(B_i + \varepsilon)\bigr)$$

where Norm denotes per-molecule z-score normalization.

### Quadratic-Variation Calibration Loss

From the pathwise quadratic variation of the Ito process:

$$[X^{(i)}]\_1 = \int_0^1 D\_{x,i}(t)\,dt$$

The residue-wise cumulative uncertainty is:

$$Q_i = \mathbb{E}\left[\int_0^1 \text{Tr}(D\_{x,i}(t))\,dt \;\middle|\; c, \widetilde{B}\right]$$

With the AmplitudeNet parameterization ($U_\theta = I$): $Q_i \approx \sum_k (a\_{x,\psi}^{(i)})^2 \cdot \Delta t$.

The quadratic-variation loss enforces that the relative distribution of injected uncertainty matches the B-factor profile:

$$\mathcal{L}\_{\text{qv}} = \sum\_{i=1}^{L} \bigl(\text{Norm}(\log(Q_i + \varepsilon)) - \widetilde{B}\_i\bigr)^2$$

This loss constrains **the diffusion mechanism** --- how much stochastic uncertainty is injected along the generation path.

### Terminal Covariance Calibration Loss

Applying the Ito formula to the second moment $X_t X_t^\top$ yields the covariance evolution:

$$\frac{d}{dt} C_t = \mathbb{E}\bigl[(X_t - \mu_t) b_t^\top + b_t(X_t - \mu_t)^\top \mid c, \widetilde{B}\bigr] + \mathbb{E}[D_t \mid c, \widetilde{B}]$$

The terminal covariance $C_1 = \text{Cov}(X_1 \mid c, \widetilde{B})$ is not equal to the quadratic variation because the drift field transports, compresses, and amplifies the intermediate uncertainty. Using $M$ Monte Carlo rollouts:

$$\hat{s}\_i^2 = \text{Tr}(\widehat{C}\_i) = \sum\_{d=1}^{3} \text{Var}\_{m=1}^{M}(x\_{1,d}^{(m,i)})$$

$$\mathcal{L}\_{\text{cov}} = \sum\_{i=1}^{L} \bigl(\text{Norm}(\log(\hat{s}\_i^2 + \varepsilon)) - \widetilde{B}\_i\bigr)^2$$

This loss constrains **the diffusion outcome** --- the actual ensemble spread at the terminal state after drift propagation.

**Why both losses are necessary:** $\mathcal{L}\_{\text{qv}}$ alone cannot prevent the drift field from collapsing the injected noise; $\mathcal{L}\_{\text{cov}}$ alone provides no direct gradient to the diffusion amplitudes through the stochastic path. Together, they constrain both the injection mechanism and the terminal result.

---

## Novelty

1. **Ensemble-aware RNA backbone generation.** To our knowledge, this is the first SE(3) flow matching model that generates conformational ensembles for RNA, rather than single structures. The framework bridges generative structural biology with experimental uncertainty quantification.

2. **B-factor calibrated stochastic transport.** The U-STEER mechanism provides a principled mathematical framework (rooted in Stratonovich SDEs) for using experimental B-factors to modulate generative stochasticity. Rather than regressing B-factors directly, we constrain the second-order statistics of the diffusion process.

3. **Dual uncertainty calibration losses.** The quadratic-variation loss ($\mathcal{L}\_{\text{qv}}$) and terminal covariance loss ($\mathcal{L}\_{\text{cov}}$) are derived rigorously from the same SDE but target complementary aspects: the former supervises the diffusion injection mechanism, the latter supervises the observable ensemble spread. This dual supervision prevents pathological behavior where noise is injected but subsequently collapsed by the drift.

4. **Conformer-level data and evaluation.** The training pipeline distinguishes individual conformers (for structure quality) from clustered ensembles (for ensemble coverage/diversity evaluation), enabling rigorous assessment of both single-structure and ensemble-level performance.

5. **Per-residue adaptive diffusion via AmplitudeNet.** The FiLM-conditioned convolutional architecture predicts spatially varying, timestep-dependent diffusion amplitudes, allowing the model to learn that flexible loop regions (high B-factor) should exhibit large conformational spread while rigid helical regions (low B-factor) remain tightly constrained.

---

## Project Structure

```
ens_flow/
├── configs/
│   ├── config.yaml              # Training configuration (Hydra)
│   └── inference.yaml           # Inference configuration
├── docs/
│   ├── SE3L_diffusion_losses_derivation.md  # Full mathematical derivation
│   ├── steer.md                 # U-STEER design document
│   └── metrics.md               # Evaluation metrics definition
├── rna_backbone_design/
│   ├── data/
│   │   ├── interpolant.py       # Flow interpolation, ODE/SDE sampling
│   │   ├── rna_conformer_dataset.py    # Conformer & cluster datasets
│   │   ├── rna_conformer_datamodule.py # PyTorch Lightning datamodule
│   │   ├── data_transforms.py   # Feature extraction and preprocessing
│   │   ├── build_datasets.py    # Dataset construction pipeline
│   │   ├── all_atom.py          # Atom37 RNA representation
│   │   ├── so3_utils.py         # SO(3) operations (geodesic, exp/log map)
│   │   └── utils.py             # B-factor normalization, alignment, etc.
│   ├── models/
│   │   ├── flow_model.py        # FlowModel: IPA trunk + torsion head
│   │   ├── flow_module.py       # FlowModule: Lightning training/val/inference
│   │   ├── amplitude_net.py     # AmplitudeNet: U-STEER diffusion amplitudes
│   │   ├── ipa_pytorch.py       # Invariant Point Attention implementation
│   │   ├── node_embedder.py     # Node feature embedding
│   │   ├── edge_embedder.py     # Edge feature embedding
│   │   └── torsion_net.py       # Torsion angle prediction head
│   └── analysis/
│       ├── metrics.py           # RMSD, TM-score, ensemble metrics
│       └── utils.py             # PDB writing, structure alignment
├── train_se3_flows.py           # Training entry point
├── inference_se3_flows.py       # Inference script
├── eval_se3_flows.py            # Evaluation script
└── pyproject.toml               # Dependencies and project metadata
```

---

## Training

Training is configured via Hydra (`configs/config.yaml`) and executed with PyTorch Lightning DDP:

```bash
python train_se3_flows.py
```

**Key hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `interpolant.sampling.num_timesteps` | 60 | Integration steps during sampling |
| `interpolant.rots.sample_schedule` | exp (rate=10) | Rotation sampling schedule |
| `model.ipa.num_blocks` | 4 | Number of IPA layers |
| `model.node_embed_size` | 256 | Node feature dimension |
| `experiment.optimizer.lr` | 1e-4 | Learning rate (drift model) |
| `experiment.training.translation_loss_weight` | 2.0 | Weight on translation VF loss |
| `experiment.training.aux_loss_t_pass` | 0.25 | Auxiliary loss time threshold |

**U-STEER training** (enable in config):

```yaml
steer:
  enabled: true
  amplitude_net:
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
  loss:
    qv_weight: 1.0
    cov_weight: 0.0  # Enable for terminal covariance loss
    cov_num_rollouts: 4
  optimizer:
    lr: 3e-4
  drift_lr_scale: 0.1  # Drift model uses lr * 0.1
```

When U-STEER is enabled, the optimizer uses separate learning rates: the pre-trained drift model (FlowModel) is fine-tuned at a reduced rate (`lr * drift_lr_scale`), while the AmplitudeNet trains at its own learning rate.

---

## Inference

```bash
python inference_se3_flows.py
```

The inference pipeline supports both deterministic (ODE) and stochastic (SDE) sampling. When U-STEER is enabled, the model generates conformational ensembles using Euler-Maruyama integration with B-factor-guided amplitude modulation.

---

## Evaluation Metrics

### Single-Structure Metrics

- **RMSD**: Root mean square deviation of C4' atoms against the best-matching ground truth conformer.
- **TM-score**: Template modeling score for global structural similarity.

### Ensemble Metrics

- **Coverage Recall** ($\text{COV-R}\_\delta$): Fraction of ground truth conformers covered by at least one generated sample within RMSD threshold $\delta$.

$$\text{COV-R}\_\delta = \frac{1}{K} \sum\_{k=1}^{K} \mathbb{1}\bigl(\min\_{x \in \text{Gen}} \text{RMSD}(x, \text{GT}\_k) \leq \delta\bigr)$$

- **Average Minimum RMSD Recall** (AMR-R): Average distance from each GT conformer to its nearest generated sample.

$$\text{AMR-R} = \frac{1}{K} \sum\_{k=1}^{K} \min\_{x \in \text{Gen}} \text{RMSD}(x, \text{GT}\_k)$$

- **Coverage Precision** ($\text{COV-P}\_\delta$): Fraction of generated samples matching at least one GT conformer within $\delta$.

$$\text{COV-P}\_\delta = \frac{1}{N} \sum\_{i=1}^{N} \mathbb{1}\bigl(\min\_{k} \text{RMSD}(x\_i, \text{GT}\_k) \leq \delta\bigr)$$

- **Average Minimum RMSD Precision** (AMR-P): Average distance from each generated conformer to its nearest GT.

$$\text{AMR-P} = \frac{1}{N} \sum\_{i=1}^{N} \min\_{k} \text{RMSD}(x_i, \text{GT}\_k)$$

- **Pairwise RMSD**: Measures intra-ensemble diversity to detect mode collapse.

Coverage evaluates how well the generated ensemble spans the true conformational landscape; precision evaluates whether generated samples correspond to real states rather than noise.

---

## Acknowledgements

The base codebase is adapted from [RNA-FrameFlow](https://github.com/rish-16/rna-backbone-design) and [Protein-FrameFlow](https://github.com/microsoft/protein-frame-flow). The IPA implementation follows [AlphaFold2](https://github.com/google-deepmind/alphafold).
