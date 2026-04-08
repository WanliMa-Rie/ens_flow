# 架构报告：EnsFlow — 三级 SE(3) RNA 构象集合生成模型

## 概述

本报告描述 EnsFlow 的三级模型架构。三级之间递进：Level 1 是确定性 SE(3) 流匹配；Level 2 在其上引入条件异方差随机桥（conditional heteroscedastic stochastic bridge）；Level 3 进一步用实验 B-factor 直接监督桥宽度网络。

| Level | 方法 | 训练额外开销 | 推断方式 |
|-------|------|-------------|---------|
| 1 | SE(3) Flow Matching | — | 确定性 ODE (Euler) |
| 2 | + Learnable Brownian-Bridge Width (NLL) | 1× BridgeWidthNet 前向 | SDE (Euler-Maruyama) |
| 3 | + B-factor Anchor | 同 Level 2 | 同 Level 2 |

通过 `stochastic_bridge.level` 配置字段切换，三级共用同一套代码。

---

## Level 1：SE(3) Flow Matching

纯确定性流匹配。**记号约定**：$t=0$ 为先验（噪声），$t=1$ 为 clean data；$\hat{x}_{i,1|t}$、$\hat{R}_{i,1|t}$ 表示 FlowModel 在时刻 $t$ 给出的对 clean endpoint $(x_{i,1}, R_{i,1})$ 的预测（对应代码中 `pred_trans` / `pred_rotmats`）。诱导的速度场：

$$u^x(i,t) = \frac{\hat{x}_{i,1|t} - x_{i,t}}{1 - t}, \qquad u^\omega(i,t) = \frac{1}{1-t}\log(R_{i,t}^\top \hat{R}_{i,1|t})$$

训练时从线性/测地线插值采样中间状态 $G_t$，推断时用 Euler ODE 积分。

---

## Level 2：Learnable Brownian-Bridge Width via Heteroscedastic NLL

### 核心思想

Level 2 只做一件事：**让 Brownian bridge 的宽度 $\sigma(t, i)$ 可学习**。漂移 (drift) 与 Level 1 完全一致 —— 依旧是标准 SE(3) flow matching，训练时走确定性插值、损失函数与 Level 1 相同。变化全部集中在"如何学 $\sigma$"上。

为此我们把 drift 的残差 $\hat{x}_{i,1|t} - x_{i,1}$ 视为服从均值零、方差 $\sigma^2(t, i)$ 的高斯观测，用一个 BridgeWidthNet 预测 $\sigma$，并以该高斯预测分布的 log-likelihood 做监督。这是教科书式的异方差回归 (Seitzer et al. 2022 等)，使 $\sigma$ 在训练时就被锚定到"drift 的典型残差尺度"上。推断时 $\sigma$ 自然作为 Euler-Maruyama 的扩散强度。

### Base schedule 与 modulation

有效桥宽度为 base schedule 与可学习 modulation 的乘积：

$$\sigma_x(t, i) = \sigma_x^0(t) \cdot \alpha^x_{i,t}, \qquad \sigma_\omega(t, i) = \sigma_\omega^0(t) \cdot \alpha^\omega_{i,t}$$

其中 base schedule 满足端点消失条件（保证 $t=0, 1$ 处桥宽度自动归零）：

$$\sigma_x^0(t) = \lambda_x \sqrt{t(1-t)}, \qquad \sigma_\omega^0(t) = \lambda_\omega \sqrt{t(1-t)}$$

modulation factor $\alpha$ 有界：

$$\alpha = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \text{sigmoid}(u)$$

### BridgeWidthNet

**核心设计动机**：相同序列在同一族中可以对应多个不同构象。仅靠序列嵌入 $c_i$ 无法区分它们；pair representation $p_{ij}$ 编码了核苷酸两两之间的结构关系（距离 bin、相对朝向），对不同构象成员是不同的。BridgeWidthNet 通过引入 pair representation，使得同一序列的不同结构成员能够得到差异化的桥宽分配。

**输入**：

$$z_{i,t} = \left[c_i,\; \Delta^x_{i,t},\; \Delta^\omega_{i,t},\; \frac{1}{N}\sum_j p_{ij},\; t\right]$$

- $c_i$ [B,N,384]：序列嵌入
- $\Delta^x_{i,t} = x_{i,t} - \hat{x}_{i,1|t}$，$\Delta^\omega_{i,t} = \log(\hat{R}_{i,1|t}^\top R_{i,t})$：几何残差（推断时一致性保证）
- $\frac{1}{N}\sum_j p_{ij}$ [B,N,128]：pair representation 沿列 mean-pool，聚合核苷酸 $i$ 与所有其他核苷酸的关系

**架构**（注意：FiLM 不是融合后的独立层，而是**内嵌在每个 ResConvBlock 内部**；时间条件 `cond` 全网络共享）：

```
# 1. 时间条件（全网络共享）
t [B, 1] → sinusoidal_embed(64) → Linear(64, H) → SiLU → Linear(H, H) = cond [B, H]

# 2. Single 路径
[single_emb=384, delta_x=3, delta_omega=3] (cat → 390)
    → Linear(390, H) → SiLU → Linear(H, H)              [B, N, H]

# 3. Pair 路径
pair_emb [B, N, N, 128]
    → mean over dim=2 → [B, N, 128]
    → Linear(128, H)                                     [B, N, H]

# 4. 融合 + 主干
x = single_out + pair_out                                 [B, N, H]
for _ in range(4):
    x = ResConvBlock(x, cond)
    #   h = LayerNorm(x)
    #   h = Conv1D(k=5)(h)
    #   h = FiLMLayer(cond)(h)   ← FiLM 在块内部
    #   h = SiLU(h)
    #   h = Conv1D(k=5)(h)
    #   x = x + h                ← 残差连接

# 5. 两个独立输出头
raw_x = LayerNorm → Linear(H, 1)        → alpha_x
raw_w = LayerNorm → Linear(H, 1)        → alpha_omega
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(raw)
```

### 训练：corruption 仍为确定性插值

`corrupt_batch` 不加任何 bridge 噪声，只做与 Level 1 完全相同的确定性插值：

$$x_{i,t} = (1 - t) x_{i,0} + t\, x_{i,1}, \qquad R_{i,t} = \text{geodesic}_t(R_{i,0}, R_{i,1})$$

$\sigma$ 不出现在 corruption 中，也不出现在 drift 的 flow loss 中 —— 这样 drift 的训练路径与 Level 1 严格一致，避免引入 $\sigma$ 相关的梯度耦合与训练不稳定。$\sigma$ 仅通过下面的 NLL 损失与推断时的 SDE 起作用。

### Level 2 训练信号：heteroscedastic NLL

**前向复用**：`_bridge_width_step` 直接复用 `model_step` 已经算出的 `model_output`，不会再跑第二次 FlowModel。BridgeWidthNet 的输入通过 `.detach()` 与 drift 解耦，确保 bridge 损失的梯度只落在 BridgeWidthNet 上。

**观测量**（detach，不反传到 FlowModel）：

$$r^x_i = \|\hat{x}_{i,1|t} - x_{i,1}\|_2, \qquad r^\omega_i = \|\log(R_{i,1}^\top \hat{R}_{i,1|t})\|_2$$

**预测分布**：BridgeWidthNet 输出 $\alpha$，有效标准差为

$$\sigma_x(t, i) = \sigma_x^0(t) \cdot \alpha^x_{i,t}, \qquad \sigma_\omega(t, i) = \sigma_\omega^0(t) \cdot \alpha^\omega_{i,t}$$

**损失**（1D-Gaussian NLL 作用在残差**范数**上，两路按 Level 1 的 per-modality 权重加权）：

$$L_{\text{width}} = \lambda_{\text{width}} \cdot \frac{1}{w_x + w_\omega}\,\mathbb{E}\!\left[ w_x\!\left(\frac{(r^x)^2}{\sigma_x^2} + \log \sigma_x^2\right) + w_\omega\!\left(\frac{(r^\omega)^2}{\sigma_\omega^2} + \log \sigma_\omega^2\right)\right]$$

其中 $w_x = $ `training.translation_loss_weight`（当前 **2.0**），$w_\omega = $ `training.rotation_loss_weights`（当前 **1.0**）。这两个权重与 Level 1 完全复用同一组 config 字段。

> **为什么需要 per-modality 权重**：$r_x$ 的单位是 Å，$r_\omega$ 的单位是 rad，两者没有任何 canonical 的换算比例。直接相加等于隐式假设"1 Å ≈ 1 rad"，会导致共享 backbone 的梯度被 magnitude 较大的模态主导。Level 1 用 `trans_scale=0.1` + `translation_loss_weight=2.0` 来手动平衡 trans / rot 两路 loss；Level 2 的 NLL 复用 Level 1 的 per-modality 权重（但**不**使用 `trans_scale`），而是保持 $\sigma$ 在物理单位里（translation 用 Å、rotation 用 rad），由 `lambda_x / lambda_omega` 的 base schedule 决定各自的 prior scale。这样推断时 `_trans_euler_maruyama_step` 直接消费的 $\sigma_x$ 也是 Å，单位在训练/推断之间是一致的。

> **关于 NLL 的精确形式**：严格的 3D isotropic Gaussian NLL 为 $\frac{\|r\|^2}{2\sigma^2} + \frac{3}{2}\log\sigma^2$，最优点在 $\sigma^{*2} = \|r\|^2/3$。我们代码里用的是 $\frac{\|r\|^2}{\sigma^2} + \log\sigma^2$，可等价地看作"把标量 $\|r\|$ 作为单一 1D 观测、用 1D Gaussian $\mathcal{N}(0, \sigma^2)$ 建模"，最优点在 $\sigma^* = \|r\|$。两者相差一个常数因子（$\sqrt{3}$），对 $\alpha$ 的相对尺度估计没有影响 —— $\sigma$ 被解释为"drift 残差范数的典型值"而非"isotropic 每轴 std"。这一选择让学到的 $\sigma$ 直接就是"残差幅度估计器"，无需额外乘 $\sqrt{3}$ 的校正。

**为什么这是一个干净的训练目标**：对 BridgeWidthNet 取导，令 $\partial L/\partial \sigma = 0$ 得

$$\sigma^{*2}(t, i) = (r_i)^2$$

即 $\sigma$ 的最优解就是当前 drift 残差范数，等价于学一个"以 $\sigma^0(t)$ 为 prior scale、当前 $(t, i)$ 条件下 drift 误差的典型幅度估计器"。$\log \sigma^2$ 项自动起到 regularizer 的作用 —— 防止 $\sigma \to 0$ 崩塌、也防止 $\sigma \to \infty$ 平凡解。再加上 sigmoid 把 $\alpha$ 夹在 $[\alpha_{\min}, \alpha_{\max}]$，避免极端值即可数值稳定。

**为什么能感应同族结构差异**：相同序列的不同构象成员对应不同的真值 $(x_1, R_1)$。对同一时间步 $t$ 的 $(x_t, R_t)$，FlowModel 的预测误差 $r$ 在不同成员上不同 —— BridgeWidthNet 看到不同的 $(\Delta x, \Delta \omega,\; \text{pair mean-pool})$，通过 NLL 学出差异化的 $\sigma$。

**与 UDBM 的关系**：UDBM 用独立的静态不确定性头 + 显式残差 MSE 回归。本设计的 NLL 形式在最优解处等价于 UDBM 的残差匹配目标（上式 $\sigma^{*2} = r^2$），但免除了手写 clamp/target 并且梯度在 $\sigma$ 远离最优值时仍然良好 —— 这是标准异方差回归的常规推导。

### 推断：Euler-Maruyama SDE

每步采样中：

1. FlowModel → 参考预测 $(\hat{x}_{1|t}, \hat{R}_{1|t})$
2. 几何残差 → BridgeWidthNet → $(\alpha_x, \alpha_\omega)$
3. 有效桥宽度 $\sigma = \sigma^0(t) \cdot \alpha$
4. Euler-Maruyama 步：

$$x_{t+\Delta t} = x_t + \Delta t \cdot u^x_\theta + \sqrt{\Delta t} \cdot \sigma_x \cdot \xi^x$$

$$R_{t+\Delta t} = R_t \cdot \exp(\Delta t \cdot u^\omega_\theta + \sqrt{\Delta t} \cdot \sigma_\omega \cdot \xi^\omega)$$

### 退化关系

- $\alpha \equiv 1$：退化为固定宽度 SE(3) Brownian bridge
- $\lambda_x = \lambda_\omega = 0$：$\sigma \equiv 0$，退化为 Level 1 的确定性流匹配

---

## Level 3：B-factor 物理锚点

Level 3 在 Level 2 的 heteroscedastic NLL 之上，额外加一项对 $\alpha$ 的直接回归锚点：

$$L_{\text{bfactor}} = \lambda_{\text{bf}} \cdot \mathbb{E}\left[\frac{1}{2}\left((\alpha^x - B^{\text{norm}})^2 + (\alpha^\omega - B^{\text{norm}})^2\right)\right]$$

$$L_{\text{Level 3}} = L_{\text{width}} + L_{\text{bfactor}}$$

其中 $B^{\text{norm}}$ 为正值归一化后的实验 B-factor（log → 平移 min 至 0 → 缩放 mean 至 1）。

**两个信号的互补性**：
- $L_{\text{width}}$（NLL）：从数据内部拟合 drift 残差分布的宽度 —— 模型的**认知不确定性**
- $L_{\text{bfactor}}$：把 $\alpha$ 拉向实验观测到的**物理柔性**
- 两者应在柔性区域一致；B-factor 起到"先验锚点"的作用，防止 NLL 在早期 drift 残差噪声大时把 $\alpha$ 学飞。

**与旧 Feynman-Kac 框架的区别**：

| | 旧 FK 框架（已移除） | Level 3（当前） |
|---|---|---|
| 辅助网络 | AmplitudeNet + UncertaintyNet | BridgeWidthNet only |
| 监督方式 | 间接（$U(0)=B$, TD 一致性） | NLL + 直接 B-factor 回归 |
| 损失数量 | 3 项（$L_{\text{TD}}$, $L_{\text{term}}$, $L_{\text{init}}$） | 2 项（NLL + bfactor） |
| 输出语义 | 无界 $\sigma$（softplus） | 有界 $\alpha$（sigmoid mapping） |
| 输入特征 | 原始噪声状态 | 几何残差 + pair representation |

---

## 修改的文件

### `rna_backbone_design/models/bridge_width_net.py`（新建）

- `BridgeWidthNet` — 预测有界 $(\alpha_x, \alpha_\omega)$，输入为 single + pair + 几何残差 + 时间步
- `compute_geometric_residuals()` — 计算 $\Delta x, \Delta \omega$
- `base_schedule(t, \lambda)` — 端点消失的桥宽度基准 $\lambda\sqrt{t(1-t)}$

### `rna_backbone_design/models/flow_module.py`

- `__init__` — 根据 `stochastic_bridge.level` 决定是否实例化 `BridgeWidthNet`
- `model_step` — 返回 `(batch_losses, model_output)`，使 bridge step 可复用 drift 前向，避免重复前向
- `training_step` — 所有 level 都调用相同的 `corrupt_batch(batch)`（纯确定性插值）；按 50% 概率做 self-conditioning 额外前向拿 `trans_sc`；Level 2/3 时把 `model_output` 传给 `_bridge_width_step`，累加到 `train_loss`
- `_bridge_width_step(noisy_batch, model_output, num_batch)` — Level 2 heteroscedastic NLL；Level 3 + B-factor 锚点；`pred_trans_1 / pred_rotmats_1` 入口处 `.detach()`，`r_x / r_ω` 用 `torch.no_grad()` 计算，bridge 梯度不回流 drift；两路 NLL 按 `translation_loss_weight : rotation_loss_weights` 加权（复用 Level 1 的 per-modality 权重配置）；另记录 `train/bridge_width_nll`、`train/bridge_nll_x`、`train/bridge_nll_w`、`train/bridge_alpha_x_mean`、`train/bridge_alpha_w_mean`、`train/bridge_r_x_mean`、`train/bridge_r_w_mean` 到 WandB，方便监控两路的尺度平衡并按需校准 `lambda_x / lambda_omega`
- `configure_optimizers` — Level 1 单参数组；Level 2/3 两参数组：drift lr = `experiment.optimizer.lr × drift_lr_scale`（当前 1e-5），bridge_width_net lr = `stochastic_bridge.optimizer.lr`（当前 3e-4）
- `validation_step` / `predict_step` — Level ≥ 2 使用 `interpolant.sample_stochastic(..., bridge_width_net, bridge_cfg=...)`，Level 1 使用原 `interpolant.sample(...)`

### `rna_backbone_design/data/interpolant.py`

- `corrupt_batch(batch)` — 纯确定性插值：translation 用 optimal-transport 匹配的 Brownian 端点 + 线性插值；rotation 用 IGSO(3)(σ=1.5) 采样 `rotmats_0` 后测地线插值；不注入任何 bridge 噪声（$\sigma$ 仅在 NLL 损失与推断 SDE 中出现）
- `_trans_euler_maruyama_step` / `_rots_euler_maruyama_step` — Euler-Maruyama 单步，drift 复用 Level 1 同样的 velocity 公式，并在末尾加 $\sqrt{\Delta t} \cdot \sigma \cdot \xi$ 噪声
- `sample_stochastic(num_batch, num_res, model, bridge_width_net, context, bridge_cfg)` — 每步计算几何残差 → BridgeWidthNet → $\sigma = \sigma^0 \cdot \alpha$ → Euler-Maruyama；最后一步回落为单次 FlowModel 前向得到最终构象

### `configs/config.yaml`

当前仓库的配置（`level: 2`，正在训练 Level 2）：

```yaml
stochastic_bridge:
  level: 2                    # 1=确定性流, 2=异方差桥, 3=桥+B-factor
  bridge_width_net:
    c_single_in: ${model.node_features.c_single_in}  # 384
    c_pair_in: ${model.edge_features.c_pair_in}      # 128
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
    alpha_min: 0.1
    alpha_max: 3.0
  base_schedule:
    lambda_x: 4.0             # σ_x^0(t) = 4.0 · √(t(1-t))，峰值 2.0 Å（匹配典型 r_x）
    lambda_omega: 0.5         # σ_ω^0(t) = 0.5 · √(t(1-t))，峰值 0.25 rad
  width_loss_weight: 1.0      # Level 2/3: heteroscedastic NLL 的全局缩放
  bfactor_supervision:
    weight: 1.0               # Level 3: B-factor 回归权重
  optimizer:
    lr: 0.0003                # BridgeWidthNet 全速学习率
  drift_lr_scale: 0.1         # drift lr = experiment.optimizer.lr × 0.1
```

配合 `experiment.optimizer.lr: 0.0001`，实际生效的分组学习率为：
- drift (FlowModel): `1e-4 × 0.1 = 1e-5`
- bridge_width_net: `3e-4`

drift 降速的原因：Level 2 接着 Level 1 预训练权重继续训练时，FlowModel 已经较稳定，只需要 bridge_width_net 从零开始快速收敛。

### 保留不动的文件

- `amplitude_net.py` — 保留旧 AmplitudeNet/UncertaintyNet 类定义，供旧 checkpoint 兼容
- `fk_utils.py` — `normalize_b_factors_positive` 仍被 Level 3 使用

---

## 数据流

### 训练阶段

```
batch (x_1, R_1, B-factors, single_emb, pair_emb)
         │
    corrupt_batch(batch)        ← 纯确定性插值，对所有 level 一致
    └─ (trans_t, rotmats_t, t)
         │
         ▼
    [self-condition, 50% 概率]
    with torch.no_grad():
        model_sc = FlowModel(noisy_batch)
        noisy_batch["trans_sc"] = model_sc["pred_trans"]
         │
         ▼
    model_step(noisy_batch)
    ├─ model_output = FlowModel(noisy_batch) = (pred_trans, pred_rotmats, pred_torsions, ...)
    └─ 返回 (batch_losses, model_output)
         │
         ├─ L_trans     = MSE(trans velocity)                    × translation_loss_weight=2.0
         ├─ L_rots_vf   = MSE(rot velocity)                      × rotation_loss_weights=1.0
         ├─ L_se3_vf    = L_trans + L_rots_vf
         ├─ L_bb_atom   = MSE(backbone atoms, ATOM37)            [gated: t > 0.25]
         ├─ L_dist_mat  = MSE(pairwise distances)                [gated: t > 0.25]
         ├─ L_torsion   = MSE(torsion sin/cos)                   [gated: t > 0.25]
         └─ L_aux       = (L_bb_atom + L_dist_mat + L_torsion)
                         × (t > aux_loss_t_pass=0.25) × aux_loss_weight=1.0
         │
         ├─ L_flow = L_se3_vf + L_aux             [所有 level]
         │
         └─ [Level 2/3] _bridge_width_step(noisy_batch, model_output, num_batch)
                │
                pred_trans_1.detach(), pred_rotmats_1.detach()
                │
                compute_geometric_residuals → (Δx, Δω)
                   Δx = trans_t - pred_trans_1                 (detached)
                   Δω = log(pred_rotmats_1^T · rotmats_t)       (detached)
                │
                BridgeWidthNet(single_emb, Δx, Δω, pair_emb, t)
                   → (α_x, α_ω)  ∈ [0.1, 3.0]
                │
                σ_x = λ_x · √(t(1-t)) · α_x                    (in Å, λ_x=4.0)
                σ_ω = λ_ω · √(t(1-t)) · α_ω                    (in rad, λ_ω=0.5)
                │
                观测量（no_grad, raw physical units）：
                   r_x = ||pred_trans_1 - trans_1||_2            (Å)
                   r_ω = ||log(rotmats_1^T · pred_rotmats_1)||_2 (rad)
                │
                NLL_x = r_x²/σ_x² + log σ_x²                    ← 1D-Gaussian NLL on ||r_x||
                NLL_ω = r_ω²/σ_ω² + log σ_ω²                    ← 1D-Gaussian NLL on ||r_ω||
                │
                # Level 1 复用的 per-modality weights 平衡单位
                w_x = translation_loss_weight = 2.0
                w_ω = rotation_loss_weights   = 1.0
                L_width = (w_x · NLL_x + w_ω · NLL_ω) / (w_x + w_ω)
                │
                [Level 3] + L_bfactor = (1/2) · MSE(α_x, B_norm) + (1/2) · MSE(α_ω, B_norm)
                │
                L_bridge = width_loss_weight · L_width
                         + [Level 3] bfactor_supervision.weight · L_bfactor
         │
         ▼
    train_loss = L_flow + L_bridge
```

> **梯度路由**：`model_output` 在 `_bridge_width_step` 内部统一 `.detach()`，所以
> L_width / L_bfactor 的梯度只更新 `bridge_width_net`，不会回流到 FlowModel。
> 同时 bridge step 不再跑第二次 FlowModel 前向 —— 直接复用 model_step 的输出。

### 推断阶段

```
x_0 ~ 先验（中心高斯平动 + 均匀 SO(3) 转动）
    │
    ├─ [Level 1] Euler ODE 循环
    │
    └─ [Level 2/3] Euler-Maruyama 循环
              │
         每步 k：
         ├─ FlowModel(x_t, t) → (x̂_{1|t}, R̂_{1|t})  [漂移 + 参考预测]
         ├─ Δx, Δω = 几何残差
         ├─ BridgeWidthNet(single, pair, Δx, Δω, t) → (α_x, α_ω)
         ├─ σ = σ₀(t) · α
         ├─ x_{k+1} = x_k + Δt·v_θ + √Δt·σ_x·ξ_x
         └─ R_{k+1} = R_k · exp(Δt·ω_θ + √Δt·σ_ω·ξ_ω)
              │
         x_1（生成结构）
```

---

## 参数量

| 模块 | 参数量 | 备注 |
|------|--------|------|
| FlowModel（漂移） | 14,248,510 (~14.25M) | IPA 架构，5 blocks |
| BridgeWidthNet | 898,178 (~898K) | 4× ResConvBlock，hidden=128；time_mlp 24.8K, single_proj 66.6K, pair_proj 16.5K, blocks 789.5K, heads 0.8K |
| **Level 1 总计** | **14,248,510 (~14.25M)** | |
| **Level 2/3 总计** | **15,146,688 (~15.15M)** | +6.30% vs Level 1 |

---

## 关键设计决策

**1. 几何残差而非原始噪声状态作为 BridgeWidthNet 输入**

旧 AmplitudeNet 输入 `(trans_t, rot_feat, single_emb, t)`，其中 `trans_t` 和 `rot_feat` 是原始噪声状态。BridgeWidthNet 改用几何残差 $(\Delta x, \Delta \omega)$，即当前状态相对于模型最佳估计的偏差。优势：(1) 推断一致——不依赖不可见的真实目标；(2) 语义更清晰——残差大意味着模型尚不确定，应放大桥宽度。

**2. 有界输出而非无界 softplus**

旧 AmplitudeNet 用 softplus 保证正值但无上界，可能导致训练不稳定。BridgeWidthNet 用 sigmoid mapping 将 $\alpha$ 限制在 $[\alpha_{\min}, \alpha_{\max}]$，配合 base schedule 的端点消失性质，保证端点约束和数值稳定。

**3. Corruption 保持确定性；$\sigma$ 只进 NLL 与推断**

我们刻意不把 $\sigma$ 放进 `corrupt_batch`，保证 drift 的训练过程与 Level 1 严格一致。代价是推断时 drift 会见到"训练未显式暴露过的 $\sigma\cdot\xi$ 扰动"，但：(i) $\alpha$ 被 sigmoid 夹在一个温和的区间 $[\alpha_{\min}, \alpha_{\max}]$ 内，$\sigma^0(t)$ 又随 $t \to 0, 1$ 归零；(ii) drift 是 clean-endpoint 预测器，对中等扰动具有鲁棒性。实验上这是 UDBM 等残差式异方差方法的主流做法。

**4. 单次 FlowModel 前向 + detach 解耦**

`_bridge_width_step` 直接复用 `model_step` 的 `model_output`，并对 `pred_trans / pred_rotmats` 以及下游的 $\Delta x, \Delta \omega, r_x, r_\omega$ 统一 `.detach()`。这达成两个目的：(a) 每 step 只跑一次 drift 前向；(b) bridge 损失的梯度流只进入 BridgeWidthNet，不污染 drift。

**5. B-factor 作为物理锚点而非主损失**

Level 3 的 B-factor 回归只是一条额外的 MSE，默认权重与 NLL 相当；其核心作用是给 $\alpha$ 一个"先验形状"防止 NLL 训练早期 drift 残差抖动导致 $\alpha$ 剧烈震荡。移除 B-factor 后退回 Level 2，训练仍然可行。

**6. Trans / rot 两路 NLL 在 raw 物理单位下用 per-modality 权重平衡**

translation 残差 $r_x$ 的单位是 Å、rotation 残差 $r_\omega$ 的单位是 rad —— 两者没有物理意义上的换算比例，直接相加会让共享 backbone 的梯度被 magnitude 较大的模态主导。处理方式上我们**对齐了 Level 1 的约定**，但做了一处取舍：

- **复用** Level 1 的 `translation_loss_weight=2.0` / `rotation_loss_weights=1.0` 作为两路 NLL 的加权系数
- **不使用** Level 1 的 `trans_scale=0.1`。Level 2 NLL 里的 $\sigma$ 有明确的物理含义（桥的 Brownian 噪声幅度），保留 Å/rad 物理单位可以让推断时 `_trans_euler_maruyama_step` 直接消费 BridgeWidthNet 的输出，不需要额外的 scale 转换。单位平衡的任务改由 base schedule 的 `lambda_x / lambda_omega` 承担：把两路的 $\sigma^0(t)$ 在各自单位下**调到典型残差的量级**，就能让 $\alpha^* = r/\sigma^0 \approx 1$ 落在 $[0.1, 3.0]$ 的中段，两路的 NLL 数值也自然接近

**校准规则**：在训练中监控 `train/bridge_r_x_mean` 和 `train/bridge_r_w_mean`，调整 `lambda_x ≈ 2 · median(r_x)`、`lambda_omega ≈ 2 · median(r_w)` 使得 $\sigma^0$ 峰值 $= \lambda/2$ 正好匹配典型残差；同时观察 `bridge_alpha_{x,w}_mean` 避免饱和到 0.1 或 3.0。当前配置的初始值 `lambda_x=4.0, lambda_omega=0.5` 基于 RNA backbone 的经验估计（典型 $r_x \approx 2$ Å、$r_\omega \approx 0.25$ rad），需要第一个 epoch 后再根据实际观测做一次微调。
