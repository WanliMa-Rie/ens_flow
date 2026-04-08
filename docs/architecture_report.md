# 架构报告：EnsFlow — 三级 SE(3) RNA 构象集合生成模型

## 概述

本报告描述 EnsFlow 的三级模型架构。三级之间逐步增强：Level 1 是确定性 SE(3) 流匹配；Level 2 把前向过程升级为**切空间提升的 SE(3) 随机桥** (tangent-space-lifted stochastic bridge)，引入闭式、端点严格的随机过程；Level 3 在此基础上用序列条件网络预测物理 B-factor，让桥宽度变成残基级的物理量。

| Level | 前向过程 | 辅助模块 | 推断 |
|-------|---------|----------|------|
| 1 | Deterministic geodesic on SE(3) | — | ODE (Euler) |
| 2 | **SE(3) Stochastic Bridge** (切空间提升，固定 $\sigma$) | — | SDE (Euler–Maruyama) |
| 3 | SE(3) Stochastic Bridge with 序列条件 B-factor 预测 (残基级 $\sigma$) | Sequence-only BFactorNet (**planned**) | SDE |

三级通过 `stochastic_bridge.level` 字段切换，共享同一套 FlowModel 漂移网络和损失形式 —— **各级的唯一差别是前向过程 (corruption) 和 $\sigma$ 的来源**。

---

## Level 1：SE(3) Flow Matching

纯确定性流匹配。**记号约定**：$t=0$ 为先验（噪声），$t=1$ 为 clean data；$\hat{x}_{i,1|t}$、$\hat{R}_{i,1|t}$ 表示 FlowModel 在时刻 $t$ 给出的对 clean endpoint $(x_{i,1}, R_{i,1})$ 的预测（对应代码中 `pred_trans` / `pred_rotmats`）。诱导的速度场：

$$u^x(i,t) = \frac{\hat{x}_{i,1|t} - x_{i,t}}{1 - t}, \qquad u^\omega(i,t) = \frac{1}{1-t}\mathrm{Log}(R_{i,t}^\top \hat{R}_{i,1|t})^\vee$$

训练时 `corrupt_batch` 做确定性插值：

$$x_{i,t} = (1-t) x_{i,0} + t\, x_{i,1}, \qquad R_{i,t} = R_{i,0}\cdot\mathrm{Exp}(t\,\hat{\omega}_i),\; \omega_i = \mathrm{Log}(R_{i,0}^\top R_{i,1})^\vee$$

分别对应 `_corrupt_trans` 的线性插值和 `_corrupt_rotmats` 调用 `so3_utils.geodesic_t(t, rotmats_1, rotmats_0)`。推断时用 Euler ODE 积分 (`interpolant.sample`)。

---

## Level 2：SE(3) Stochastic Bridge (Tangent-Space Lifted)

### 2.1 动机

Level 1 的前向过程是完全确定的 —— 给定 $(R_{i,0}, R_{i,1}, t)$，中间态 $R_{i,t}$ 唯一确定。我们希望引入随机性让前向过程成为一个从 $\rho_0$ 到 $\rho_1$ 的 **stochastic bridge**，并同时保留三个性质：

1. **闭式采样**（simulation-free 训练）
2. **严格端点约束**（$t=0$ 必为 $R_0$，$t=1$ 必为 $R_1$）
3. **可解析的 drift**（能直接做 velocity matching）

$SO(3)$ 上由 Laplace–Beltrami 算子生成的**内禀** Brownian bridge 没有闭式（只能用 heat-kernel 的 series representation，参见 IGSO(3)-based diffusion 工作）。我们的做法是：**在李代数 $\mathfrak{so}(3)\cong\mathbb{R}^3$ 中定义闭式 Brownian bridge，再通过指数映射推前到 $SO(3)$**。这个构造得到的是一个 **chart-induced**（非内禀）的随机桥，但三条性质全部满足。

> **诚实的表述**：这不是 $SO(3)$ 上的 intrinsic Brownian bridge，而是一个依赖以 $R_0$ 为基点的图 (chart) 的切空间提升桥。对应的 marginal $p_t(R_t)$ 是围绕测地线的 wrapped Gaussian，不是 heat-kernel 分布。我们选择它是因为 tractability 高于 geometric purity。

### 2.2 平动桥（欧氏 Brownian Bridge）

对每个残基 $i$，标准欧氏 BB：

$$x_{i,t} = (1-t)\,x_{i,0} + t\,x_{i,1} + \sigma_x \sqrt{t(1-t)}\cdot\varepsilon_{i}^x, \qquad \varepsilon_{i}^x \sim \mathcal{N}(0, I_3)$$

等价的条件分布：

$$x_{i,t}\mid x_{i,0}, x_{i,1} \sim \mathcal{N}\!\left((1-t)x_{i,0} + t\,x_{i,1},\; \sigma_x^2\, t(1-t)\,I_3\right)$$

### 2.3 转动桥（切空间 BB）

对每个残基 $i$，先在 $\mathfrak{so}(3)$ 中定义对数坐标下的相对旋转：

$$\omega_i = \mathrm{Log}(R_{i,0}^\top R_{i,1})^\vee \in \mathbb{R}^3$$

然后在 $\mathfrak{so}(3)\cong\mathbb{R}^3$ 中做 Brownian bridge：

$$\Omega_{i,t} = t\,\omega_i + \sigma_\omega \sqrt{t(1-t)}\cdot\varepsilon_{i}^\omega, \qquad \varepsilon_{i}^\omega \sim \mathcal{N}(0, I_3)$$

最后通过指数映射推前到 $SO(3)$：

$$R_{i,t} = R_{i,0}\cdot\mathrm{Exp}(\hat{\Omega}_{i,t})$$

### 2.4 端点约束与条件分布（命题）

**命题 1（端点约束）**：上述构造满足 $R_{i,t=0} = R_{i,0}$ 且 $R_{i,t=1} = R_{i,1}$。

*证明*：$t=0$ 时 $\Omega_{i,0}=0$，$\mathrm{Exp}(0)=I$，故 $R_{i,0}$ 保持。$t=1$ 时 $\Omega_{i,1}=\omega_i$，$\mathrm{Exp}(\hat{\omega}_i) = R_{i,0}^\top R_{i,1}$，故 $R_{i,1} = R_{i,0}(R_{i,0}^\top R_{i,1}) = R_{i,1}$。$\square$

**命题 2（闭式条件分布）**：

$$\Omega_{i,t}\mid R_{i,0}, R_{i,1} \sim \mathcal{N}\!\left(t\,\omega_i,\; \sigma_\omega^2\, t(1-t)\, I_3\right)$$

$R_{i,t}$ 的条件分布是此 Gaussian 在 $\Phi_{R_{i,0}}(\Omega) = R_{i,0}\mathrm{Exp}(\hat{\Omega})$ 下的 pushforward —— 一个围绕测地线的 wrapped Gaussian。

### 2.5 漂移与 velocity matching 目标

$\Omega_{i,t}$ 满足 $\mathfrak{so}(3)$ 中的 BB SDE：

$$d\Omega_{i,t} = \frac{\omega_i - \Omega_{i,t}}{1-t}\,dt + \sigma_\omega\,dW_{i,t}$$

速度场目标与 Level 1 **一字不差**：

$$u^x_i(t) = \frac{x_{i,1} - x_{i,t}}{1-t}, \qquad u^\omega_i(t) = \frac{1}{1-t}\mathrm{Log}(R_{i,t}^\top R_{i,1})^\vee$$

唯一的改动在 $R_{i,t}$（现在从 BB 采样而非沿测地线）。Conditional Flow Matching 理论保证，只要 conditional 采样分布正确，训练就能拟合对应的 marginal velocity field —— 训练代码、损失、网络、per-modality 权重全部复用 Level 1。

### 2.6 Corruption 实现（`interpolant.py`）

Level 2 的唯一代码改动：把 `_corrupt_rotmats` 从确定性测地线换成切空间 BB 采样，`_corrupt_trans` 在线性均值上加 BB 噪声。

```python
def _corrupt_trans_bridge(self, trans_0, trans_1, t, res_mask, sigma_x):
    mean  = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
    std   = sigma_x * torch.sqrt(t * (1 - t) + 1e-8)          # [B, 1]
    noise = torch.randn_like(mean) * std[..., None]
    trans_t = mean + noise
    trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
    return trans_t * res_mask[..., None]

def _corrupt_rotmats_bridge(self, rotmats_0, rotmats_1, t, res_mask, sigma_omega):
    # ω = Log(R_0^T R_1) ∈ R^3 (利用现有 calc_rot_vf)
    omega = so3_utils.calc_rot_vf(rotmats_0, rotmats_1)         # [B, N, 3]
    std   = sigma_omega * torch.sqrt(t * (1 - t) + 1e-8)        # [B, 1]
    noise = torch.randn_like(omega) * std[..., None]
    Omega_t = t[..., None] * omega + noise                      # [B, N, 3]
    # Push forward: R_t = R_0 · Exp(hat(Ω_t))
    delta_R   = so3_utils.rotvec_to_rotmat(Omega_t)             # exp 映射
    rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_0, delta_R)

    identity = torch.eye(3, device=rotmats_t.device)
    rotmats_t = (rotmats_t * res_mask[..., None, None]
                 + identity[None, None] * (1 - res_mask[..., None, None]))
    return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)
```

注意 Level 1 现有代码的 rot corruption `so3_utils.geodesic_t(t, rotmats_1, rotmats_0)` 等价于 $R_0\cdot\mathrm{Exp}(t\hat\omega)$，也就是 BB 的**均值**。Level 2 只是在 $t\omega$ 上叠加了一个方差为 $\sigma_\omega^2 t(1-t)$ 的高斯扰动。这也意味着**设 $\sigma_\omega = 0$ 时 Level 2 严格退化为 Level 1**。

### 2.7 损失：Level 1 形式完全复用

Level 2 的训练损失**与 Level 1 完全相同** —— 标准的 flow-matching MSE 按 per-modality 权重加权，不引入任何新的 NLL 项，也不需要任何新参数：

$$L_{\text{flow}} = w_x \cdot \Big\|v_\theta^x - \frac{x_1 - x_t}{1-t}\Big\|^2 + w_\omega \cdot \Big\|v_\theta^\omega - \frac{\mathrm{Log}(R_t^\top R_1)^\vee}{1-t}\Big\|^2$$

其中 $w_x$ = `translation_loss_weight` (2.0)，$w_\omega$ = `rotation_loss_weights` (1.0)。辅助损失 $L_{\text{bb\_atom}} + L_{\text{dist\_mat}} + L_{\text{torsion}}$ 与 Level 1 一字不差。

**为什么这个损失在 BB 之下仍然正确**：由 conditional flow matching 理论（参见正文 Proposition 6），只要 conditional 采样分布 $p_t(G_t\mid G_0, G_1)$ 和 conditional target $u_t(G\mid G_0, G_1)$ 组成一对合法的 interpolant–velocity pair，用它做 MSE 回归就能拟合对应的 marginal velocity field。简化目标 $u^\omega = \mathrm{Log}(R_t^\top R_1)/(1-t)$ 对应 BB 采样下的 marginal drift 是有效的，相当于 "flow matching with stochastic interpolant"。

### 2.8 推断：Euler–Maruyama（固定 $\sigma$）

每步采样：

1. FlowModel → $(\hat{x}_{1|t}, \hat{R}_{1|t})$
2. 漂移速度：$u_\theta^x = (\hat x_{1|t} - x_t)/(1-t)$，$u_\theta^\omega = \mathrm{Log}(R_t^\top \hat R_{1|t})^\vee/(1-t)$
3. Euler–Maruyama 单步（固定 $\sigma$）：

$$x_{k+1} = x_k + \Delta t \cdot u_\theta^x + \sqrt{\Delta t}\cdot\sigma_x\cdot\xi_k$$

$$R_{k+1} = R_k\cdot\mathrm{Exp}\!\left(\widehat{\Delta t\cdot u_\theta^\omega + \sqrt{\Delta t}\cdot\sigma_\omega\cdot\zeta_k}\right)$$

$\sigma_x, \sigma_\omega$ 直接从 config 读取，**不再依赖 BridgeWidthNet**。现有的 `_trans_euler_maruyama_step` / `_rots_euler_maruyama_step` 完全可以复用（它们本来就接受标量 $\sigma$），只需要新增一个 `sample_bridge(num_batch, num_res, model, context, sigma_x, sigma_omega)` 方法，走固定 $\sigma$ 的 SDE 循环。

### 2.9 Level 2 与 Level 1 的关系

- **$\sigma_x = \sigma_\omega = 0$** → BB 退化为确定性测地线，SDE 退化为 ODE。Level 2 **严格包含** Level 1。
- **$\sigma$ 相关的唯一训练改动** 是 `corrupt_batch` 里两项高斯噪声。损失、网络、优化器组完全不动。
- **参数量**：与 Level 1 相同 ($\approx 14.25$ M)，没有新增任何模块。

### 2.10 关键假设与 caveats

- **Cut locus**：$|\omega_i| < \pi$ 假设 principal log 单值。训练 prior 为 `IGSO(3)(σ=1.5) · R_1`（现有代码），$R_0$ 集中在 $R_1$ 附近，$|\omega_i|\ll\pi$ 高概率成立，基本不碰 cut locus。推断 prior 是 uniform SO(3)，但推断循环**不显式计算 $\omega$**（只逐步推进 $R_t$ 到 $\hat R_1$），cut locus 不直接触发。
- **$t\to 1$ 漂移奇异性**：训练 $t$ 采样继续使用 `t ~ U(min_t, 1 - min_t)`，`min_t = 1e-2`（config 现有值）。
- **$\sqrt{t(1-t)}$ 数值稳定**：加 `+1e-8` 防止端点处梯度 NaN（已在代码片段中）。
- **$\sigma_x, \sigma_\omega$ 标定**：参考 Level 1 的典型 drift residual（$r_x\approx 2$ Å，$r_\omega\approx 0.25$ rad），初始取 $\sigma_x = 2.0$ Å，$\sigma_\omega = 0.25$ rad。BB 方差在 $t=0.5$ 处为 $\sigma^2\cdot 0.25$，对应每个残基 $t=0.5$ 处的噪声幅度是 $\sigma/2$，与典型残差同量级。

### 2.11 Config

```yaml
stochastic_bridge:
  level: 2                    # 1=deterministic flow, 2=SE(3) bridge, 3=bridge + b-factor
  bridge:
    sigma_x: 2.0              # Å — translation bridge width (fixed)
    sigma_omega: 0.25         # rad — rotation bridge width (fixed)
```

原 `bridge_width_net`、`base_schedule`、`width_loss_weight`、`bfactor_supervision`、独立 `optimizer` / `drift_lr_scale` 字段全部移除 —— 这些属于旧 Level 2/3 的 BridgeWidthNet-based 设计，不再使用。

---

## Level 3：B-factor-Constrained Flexibility Bridge

### 3.1 动机与设计原则

Level 2 的 $(\bar\sigma_x, \bar\sigma_\omega)$ 是全局常数，所有残基共享同一个桥宽。RNA 骨架的物理柔性是**残基级**的 —— 保守区 stiff、loop 区 flexible —— 这种柔性由实验 B-factor 直接测量 ($B \propto \langle u^2 \rangle$)。Level 3 引入一个**序列条件、残基级、时间无关**的 flexibility multiplier $\nu_i(s) > 0$，把 Level 2 的固定桥宽升级为 residue-wise 桥宽，再用 B-factor 做监督。

**两条强设计原则**：

1. **严格嵌套 (L2 ⊂ L3)**：令 $\nu_i \equiv 1$，Level 3 的 corruption、velocity target、loss 与 Level 2 逐 bit 相同。不是"approximately similar"，是严格退化。
2. **Identifiability**：每个残基只有一个标量 B-factor，因此只应该学一个标量 $\nu_i$。把 translation width 和 rotation width 通过**同一个** $\nu_i$ 联合缩放，避免 "translation 变大/rotation 变小" 这种不可辨识的自由度。

### 3.2 Flexibility head

给定 per-residue 条件特征 $s_i \in \mathbb{R}^{c_\text{single}}$（直接用 dataset 的 `single_embedding`，不和 FlowModel backbone 共享），定义一个小 MLP head $g_\psi: \mathbb{R}^{c_\text{single}} \to \mathbb{R}$ 和 per-sample 归一化：

$$a_i = \mathrm{softplus}(g_\psi(s_i)) + \varepsilon, \qquad \nu_i = \frac{a_i}{\frac{1}{L}\sum_{k=1}^{L} a_k}$$

其中 $\varepsilon > 0$ 是数值稳定常数，$L$ 是当前样本的有效 residue 数 (`res_mask.sum()`)。

**关键性质**：

- **Mean-1 归一化**：$\frac{1}{L}\sum_i \nu_i \equiv 1$ 对任意 $\psi$、任意 batch 恒成立，不仅是 init 时
- **退化条件**：$a_i$ 对 $i$ 是常数 (例如 $\psi$ 的最后一层 weight 全零) $\Rightarrow$ $\nu_i \equiv 1$。Level 3 在 init 时严格等同于 Level 2
- **全局尺度消除**：$\nu$ 只刻画 "residue 之间的相对 flexibility"，全局 sigma scale 仍由 $(\bar\sigma_x, \bar\sigma_\omega)$ 控制，避免 gauge 自由度

Head 实现：2 层 MLP，最后一层 `Linear` **零初始化 weight**、bias 任意（一般设 `log(e-1) ≈ 0.5413`，这样 softplus 输出 1），保证 $\nu_i \equiv 1$ 严格成立。

### 3.3 Residue-wise bridge

**平动桥**：Level 2 的固定 $\bar\sigma_x$ 换成 residue-wise $\sigma_{x,i} = \bar\sigma_x\sqrt{\nu_i}$：

$$x_{i,t} = (1-t)x_{i,0} + t x_{i,1} + \bar\sigma_x \sqrt{\nu_i}\sqrt{t(1-t)}\,\varepsilon_{i}^x,\quad \varepsilon_i^x \sim \mathcal{N}(0, I_3)$$

**转动桥**：同理 $\sigma_{\omega,i} = \bar\sigma_\omega\sqrt{\nu_i}$：

$$\omega_i = \mathrm{Log}(R_{i,0}^\top R_{i,1})^\vee, \quad \Omega_{i,t} = t\omega_i + \bar\sigma_\omega\sqrt{\nu_i}\sqrt{t(1-t)}\,\varepsilon_i^\omega, \quad R_{i,t} = R_{i,0}\cdot\mathrm{Exp}(\hat\Omega_{i,t})$$

Level 3 相对 Level 2 的**唯一结构性改动**：

$$\bar\sigma_x^2 \mapsto \nu_i \bar\sigma_x^2, \quad \bar\sigma_\omega^2 \mapsto \nu_i \bar\sigma_\omega^2$$

Bridge family 不变，drift target 不变，endpoint 严格性不变。

### 3.4 Velocity target：与 Level 2 一字不差

**理论上**的 canonical rotation target 是 Jr-corrected 形式：

$$a^\star_{t,i} = J_r(\Omega_{t,i})\cdot\frac{\omega_i - \Omega_{t,i}}{1-t}$$

**代码实现**复用 Level 2 的简化形式（与 Level 1 一字不差）：

$$u^x_i(t) = \frac{x_{i,1} - x_{i,t}}{1-t}, \qquad u^\omega_i(t) = \frac{\mathrm{Log}(R_{i,t}^\top R_{i,1})^\vee}{1-t}$$

两种形式在 $\mathcal{O}(\theta^2)$ 项开始不同（BCH 展开的 $\frac{1}{2}(\Omega_t \times \omega)$ 项符号相反），但都是 conditional flow matching 意义下合法的 conditional velocity target —— 都能拟合出对应的 marginal velocity field。简化形式的选择使 Level 2 和 Level 3 **共享完全相同的 drift network 代码**，L2 checkpoint 可以直接热启动 L3（参见 §3.7 training protocol）。这也与 FrameFlow / FoldFlow / RNA-FrameFlow 等 SE(3) flow matching 基线保持一致。

> **Appendix C 的 "drift 不依赖 $\nu$" 证明**对两种 target 形式都成立 —— 因为 Brownian bridge 的 drift 本来就与扩散幅度解耦，$\nu$ 只出现在 conditional path 的方差里，不出现在 drift 里。

### 3.5 B-factor alignment loss

**Bridge-induced MSD 推导**（appendix proof sketch D）：设 residue $i$ 的局部 frame-defining 模板原子集合为 $\mathcal{A}_i$，原子 $j$ 的 local 坐标 $q_{ij}\in\mathbb{R}^3$。小角度近似下：

$$\delta y_{ij}(t) \approx \delta x_i(t) + \delta\theta_i(t) \times q_{ij}$$

其中 $\delta x_i(t) \sim \mathcal{N}(0, \nu_i\bar\sigma_x^2 t(1-t) I_3)$，$\delta\theta_i(t) \sim \mathcal{N}(0, \nu_i\bar\sigma_\omega^2 t(1-t) I_3)$。取原子平均与时间平均 ($\int_0^1 t(1-t)\,dt = 1/6$)：

$$\overline{\mathrm{MSD}}_i = \frac{\nu_i}{6}\Big(3\bar\sigma_x^2 + 2\bar\sigma_\omega^2 c_i\Big), \qquad c_i = \frac{1}{|\mathcal{A}_i|}\sum_{j\in\mathcal{A}_i}|q_{ij}|^2$$

用标准各向同性关系 $B = \frac{8\pi^2}{3}\mathrm{MSD}$ 定义 bridge uncertainty proxy：

$$\widehat{B}_i^{\text{bridge}} = \frac{4\pi^2}{9}\nu_i\Big(3\bar\sigma_x^2 + 2\bar\sigma_\omega^2 c_i\Big) \equiv \kappa_i \nu_i$$

**代码简化**：对 RNA frame atoms $[\text{C3}', \text{C4}', \text{O4}']$，从 `nucleotide_constants.nttype_rigid_group_default_frame` 读出的 local 坐标得到

$$c_i \approx \tfrac{1}{3}\Big(|\text{C3}'|^2 + |\text{C4}'|^2 + |\text{O4}'|^2\Big) \approx 1.473\ \text{Å}^2 \equiv c$$

**对所有 residue 是同一个常数**（A/U/G/C 的差别 < $10^{-3}$ Å，完全可忽略）。量级检查：$3\bar\sigma_x^2 = 12\ \text{Å}^2$ vs $2\bar\sigma_\omega^2 c \approx 0.18\ \text{Å}^2$，translation 贡献主导。因此 $\kappa_i \equiv \kappa$ 也是常数。

**归一化 B-factor target**（per-sample，消除 resolution/refinement scale 差异）：

$$b_i = \frac{B_i}{\frac{1}{L}\sum_{k=1}^{L} B_k}, \qquad \widehat{b}_i = \frac{\kappa_i\nu_i}{\frac{1}{L}\sum_k \kappa_k\nu_k} = \frac{\nu_i}{\frac{1}{L}\sum_k \nu_k} = \nu_i$$

（最后一步用到 $\kappa_i$ 常数 + $\nu_i$ 已经 mean-1 归一化）

**Loss 最终形式**：

$$\mathcal{L}_B = \frac{1}{L}\sum_{i=1}^{L}\big(\log\nu_i - \log b_i\big)^2$$

干净到一行 —— `L_B` 本质上是 `log ν` 去拟合 `log b`。$c_i$ 和 $\kappa$ 在 loss 里不出现，只用于 writeup 的物理动机。

### 3.6 总 loss

$$\mathcal{L}_{\text{level3}} = \mathcal{L}_{\text{bridge}} + \lambda_B \mathcal{L}_B + \lambda_\nu \mathcal{L}_\nu$$

其中：

- $\mathcal{L}_{\text{bridge}}$ = Level 2 的标准 SE(3) flow matching loss（translation + rotation VF + backbone atom + dist mat + torsion），**一字不改**
- $\mathcal{L}_B$ = §3.5 的 B-factor alignment
- $\mathcal{L}_\nu = \frac{1}{L}\sum_i (\log\nu_i)^2$ = 很弱的 $\log$-space 正则，防止 $\nu$ 过度尖锐；因为 $\nu$ 已经 mean-1 归一化，这项也自动中心化

### 3.7 Training protocol：warm-start + always stop-grad

**Stage 0 — Level 2 pretrain**：先用 Level 2 训到稳定的 FlowModel。Level 3 从这个 checkpoint 热启动。

**Stage 1 — $\psi$-only warmup**（可选）：冻结 FlowModel，只训 FlexibilityNet，目标只有 $\mathcal{L}_B$。让 $\nu$ 先学会合理的 residue profile 再解冻 backbone。通过 `stochastic_bridge.training_phase: psi_only` 开启。

**Stage 2 — joint fine-tune**：解冻所有参数，目标 $\mathcal{L}_{\text{level3}}$。通过 `stochastic_bridge.training_phase: joint_stopgrad` 开启。

**Stop-gradient 协议（始终启用）**：在 `corrupt_batch` 里使用 `nu.detach()` 作为 bridge width，因此 $\mathcal{L}_{\text{bridge}}$ 不会通过 bridge sampling 路径回传到 $\psi$。$\psi$ 只由 $\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$ 更新。这避免了 velocity loss "偷偷缩小 $\nu$ 让 MSE 变小" 的退化。代码上的体现：

```python
nu_raw = self.flexibility_net(single_emb, res_mask)   # gradient path → ψ
# bridge sampling 用 detached nu: L_bridge 不回传 ψ
noisy_batch = self.interpolant.corrupt_batch(batch, nu=nu_raw.detach())
# flow model forward → L_bridge
batch_losses = self.model_step(noisy_batch)           # gradient path → FlowModel
# 用非 detached nu 算 L_B：只更新 ψ
L_B = log_mse(nu_raw, b_factors_normalized)
L_nu = (nu_raw.log() ** 2).mean()
train_loss = L_bridge + λ_B · L_B + λ_ν · L_nu
```

这样 FlowModel 只受 $\mathcal{L}_{\text{bridge}}$ 梯度，FlexibilityNet 只受 $\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$ 梯度。Stage 切换靠 `configure_optimizers` 根据 `training_phase` 选择 freeze 哪些参数组。

### 3.8 Inference：per-residue Euler-Maruyama

对给定 sequence，先一次性算出 $\nu_i = \text{FlexibilityNet}(s)$（no_grad），然后每个 residue 用各自的宽度做 Euler-Maruyama：

$$x_{k+1,i} = x_{k,i} + \Delta t \cdot u_\theta^x + \bar\sigma_x\sqrt{\nu_i}\sqrt{\Delta t}\cdot\xi$$

$$R_{k+1,i} = R_{k,i}\cdot\mathrm{Exp}\!\left(\widehat{\Delta t\cdot u_\theta^\omega + \bar\sigma_\omega\sqrt{\nu_i}\sqrt{\Delta t}\cdot\zeta}\right)$$

Rigid residue ($\nu_i < 1$) 自动使用更小噪声，flexible residue ($\nu_i > 1$) 自动使用更大噪声。若需 probability-flow ODE，把所有噪声项置零即可。

### 3.9 B-factor 数据与 NMR fallback

- **数据源**：`build_datasets.py` 从 PDB `atom.bfactor` 解析，聚合到 C4' 原子得到 residue-level `b_factors`，通过 `conformer_collate` 以 `[B, N]` 形式进入 batch
- **unresolved residues**：`b_factors[unsolved] = 0.0`，通过 `res_mask == 0` 隐式标识
- **NMR 结构 fallback**：现代 PDB 里 NMR 结构的 b-factor 通常全为 0 或 ensemble-derived。Level 3 在 `L_B` 计算前做 per-residue + per-sample gating：
  - Residue 级：只对 `res_mask == 1 AND b_factors > 0` 的 residue 计 loss
  - Sample 级：整条 RNA 的 valid `b_factors` 全为 0 或方差为 0，该样本的 `L_B = 0`，不进 backward
- **标定**：在 training 开始前统计 dataset 的 `b_factors` 全局均值与方差，用于 `λ_B` 的 warm-start 调整

### 3.10 与 Level 2 的严格嵌套关系（sanity 条款）

- $\psi$ 最后一层 weight = 0 $\Rightarrow$ $a_i$ 对 $i$ 常数 $\Rightarrow$ $\nu_i \equiv 1$ $\Rightarrow$ bridge corruption 与 Level 2 完全相同
- 在此 init 状态下，$\log \nu_i = 0$，$\mathcal{L}_\nu = 0$
- $\mathcal{L}_B$ 在此 init 状态下 = $(\log 1 - \log b_i)^2$ 的平均，只驱动 $\psi$，不影响 FlowModel
- 若 $\lambda_B = 0$ 并且 $\psi$ 从零 init，Level 3 训练动力学与 Level 2 完全相同

---

## 修改的文件（新 Level 2 所需）

### `rna_backbone_design/data/interpolant.py`（改动）

- **新增** `_corrupt_trans_bridge(trans_0, trans_1, t, res_mask, sigma_x)`：线性均值 + $\sigma_x\sqrt{t(1-t)}$ 高斯噪声
- **新增** `_corrupt_rotmats_bridge(rotmats_0, rotmats_1, t, res_mask, sigma_omega)`：切空间 BB 采样 $\Omega_t = t\omega + \sigma_\omega\sqrt{t(1-t)}\varepsilon$，pushforward $R_t = R_0\mathrm{Exp}(\hat\Omega_t)$
- **修改** `corrupt_batch(batch)`：当 `level >= 2` 时路由到 bridge 采样，`level == 1` 时保持原 deterministic 插值。Bridge 超参 `sigma_x`、`sigma_omega` 从 `self._cfg.bridge` 读取
- **新增** `sample_bridge(num_batch, num_res, model, context, sigma_x, sigma_omega)`：Euler–Maruyama 推断，固定 $\sigma$，替代原依赖 BridgeWidthNet 的 `sample_stochastic`
- **保留** `_trans_euler_maruyama_step` / `_rots_euler_maruyama_step`：SDE 单步函数不变，只是输入 $\sigma$ 来自 config 而非 BridgeWidthNet

### `rna_backbone_design/models/flow_module.py`（改动）

- **删除** Level 2 的 `_bridge_width_step`：新 Level 2 不再有 bridge NLL 损失
- **删除** Level 2 下的 BridgeWidthNet 实例化、调用、dual-optimizer 设置。Level 2 单参数组，与 Level 1 相同
- **修改** `training_step`：Level 1 / Level 2 共享同一条代码路径，仅 `corrupt_batch` 在 Level 2 时注入 BB 噪声；损失函数完全相同
- **修改** `validation_step` / `predict_step`：Level 2 调用 `interpolant.sample_bridge(...)`，Level 1 保持 `interpolant.sample(...)`

### `rna_backbone_design/models/bridge_width_net.py`（弃用）

- 当前 Level 2 的 BridgeWidthNet（heteroscedastic NLL + pair/single 条件）被移出 Level 2
- **选项 1**（推荐）：直接删除 `bridge_width_net.py`
- **选项 2**：保留作为 Level 3 的备选组件（若 sequence-only 预测效果不佳可复用其 pair-aware 架构）
- `base_schedule`、`compute_geometric_residuals` 辅助函数在新 Level 2 中不再使用

### `configs/config.yaml`（改动）

```yaml
stochastic_bridge:
  level: 2
  bridge:
    sigma_x: 2.0
    sigma_omega: 0.25
```

删除 `bridge_width_net`、`base_schedule`、`width_loss_weight`、`bfactor_supervision`、`optimizer`、`drift_lr_scale` 字段。

### 保留不动

- `rna_backbone_design/models/flow_model.py`：FlowModel 漂移网络不变
- `rna_backbone_design/data/so3_utils.py`：`calc_rot_vf`、`rotvec_to_rotmat`、`geodesic_t` 直接可用
- Level 1 的所有数据 pipeline、loss 组件、self-conditioning 逻辑

---

## 数据流

### 训练阶段（Level 2）

```
batch (x_1, R_1, single_emb, pair_emb)
   │
   ├─ t ~ U(min_t, 1 - min_t)                     [avoid drift singularity]
   ├─ x_0 ~ N(0, I) · NM_TO_ANG + OT match        [same as Level 1]
   ├─ R_0 ~ IGSO(3)(σ=1.5) · R_1                  [same as Level 1]
   │
   ├─ [Level 2 closed-form BB corruption]
   │     x_t = (1-t)·x_0 + t·x_1 + σ_x · √(t(1-t)) · ε_x
   │     ω   = Log(R_0^T · R_1)
   │     Ω_t = t·ω + σ_ω · √(t(1-t)) · ε_ω
   │     R_t = R_0 · Exp(hat(Ω_t))
   │
   ▼
[self-condition, 50% 概率, 复用 Level 1 逻辑]
   │
   ▼
FlowModel(x_t, R_t, t, single_emb, pair_emb) → (x̂_1, R̂_1, pred_torsions, ...)
   │
   ├─ L_trans    = w_x · MSE(v_θ^x, (x_1 - x_t)/(1-t))
   ├─ L_rots_vf  = w_ω · MSE(v_θ^ω, Log(R_t^T R_1)/(1-t))
   ├─ L_se3_vf   = L_trans + L_rots_vf
   ├─ L_bb_atom  = MSE(backbone atoms, ATOM37)       [gated: t > 0.25]
   ├─ L_dist_mat = MSE(pairwise distances)           [gated: t > 0.25]
   ├─ L_torsion  = MSE(torsion sin/cos)              [gated: t > 0.25]
   │
   └─ train_loss = L_se3_vf + (L_bb_atom + L_dist_mat + L_torsion) · aux_loss_weight
```

**与 Level 1 的唯一差别**：`corrupt_batch` 里多了两项 BB 噪声。损失、网络前向、梯度路由完全相同。**无需任何 detach 处理**（因为没有 bridge-dependent 辅助网络）。

### 推断阶段（Level 2）

```
x_0 ~ N(0, I),  R_0 ~ U(SO(3))
   │
   for k = 0 … num_timesteps - 1:
       FlowModel(x_k, R_k, t_k) → (x̂_1, R̂_1)
       │
       u_x = (x̂_1 - x_k) / (1 - t_k)
       u_ω = Log(R_k^T · R̂_1) / (1 - t_k)
       │
       x_{k+1} = x_k + Δt · u_x + √Δt · σ_x · ξ
       R_{k+1} = R_k · Exp(hat(Δt · u_ω + √Δt · σ_ω · ζ))
       │
       t_{k+1} = t_k + Δt
   │
   最后一步直接使用 (x̂_1, R̂_1) 生成 ATOM37 构象（含 pred_torsions）
```

---

## 参数量

| 模块 | 参数量 | 备注 |
|------|--------|------|
| FlowModel（漂移） | 14,248,510 (~14.25M) | IPA 架构，5 blocks，与 Level 1 相同 |
| **Level 1 / Level 2 总计** | **14,248,510 (~14.25M)** | Level 2 相对 Level 1 **零额外参数** |
| Level 3 SequenceBFactorNet | TBD | 预计 $<1$M，待设计确定 |

Level 2 相对 Level 1 **没有新增任何参数**，全部随机性由 BB 噪声引入。这相对旧 Level 2 设计是一次显著简化 —— 旧设计引入了 898K 参数的 BridgeWidthNet，现已完全移除。

---

## 关键设计决策

### 1. 为什么在 $\mathfrak{so}(3)$ 中做 BB 而不在 $SO(3)$ 直接做

内禀 BB（Laplace–Beltrami 生成）没有闭式采样。在切空间做 BB 然后 pushforward，以牺牲 "intrinsic" 换来 tractability。我们在论文/报告中诚实表述：this is a tangent-space-lifted bridge, NOT an intrinsic Brownian bridge on $SO(3)$。对应的 marginal $p_t(R_t)$ 是 wrapped Gaussian around geodesic，而非 heat-kernel 分布。

### 2. 为什么 Level 2 使用固定 $\sigma$，而不是 BridgeWidthNet

旧 Level 2 用 BridgeWidthNet 学习 heteroscedastic $\sigma(t, i)$，通过 NLL on drift residual 做监督。这个设计有两个问题：

- **并非真正的 stochastic bridge**：corruption 仍然是确定性的；$\sigma$ 只进入 inference SDE 和 NLL 损失，不进入 forward process。严格来讲模型学的是 "deterministic flow matching + post-hoc 异方差回归"，而非 bridge 本身
- **level 边界混乱**：heteroscedasticity 和 B-factor 监督绑在同一个网络上，不好 ablate

新设计将这两件事分开：

- **Level 2** 负责把 forward process 改成真正的随机桥（闭式、端点精确），$\sigma$ 固定
- **Level 3** 负责引入物理先验（B-factor 调制），让 $\sigma$ 变成残基级

这样 L1 vs L2 ablation 可以 isolate "随机化 forward process" 带来的增益，L2 vs L3 ablation 可以 isolate "物理先验 $\sigma$" 带来的增益 —— story 更干净。

### 3. 为什么 Level 2 不在 loss 上加任何新项

加 NLL、heteroscedastic、b-factor 等额外损失会混淆 "stochastic bridge 本身的贡献" 和 "辅助监督的贡献"。Level 2 严格只改 corruption，损失保持 Level 1 原样 —— L1 vs L2 ablation 完美 isolate "forward process 由 deterministic 变 stochastic" 的贡献。**所有可学习 / 异方差 / 物理先验的组件都下沉到 Level 3**。

### 4. 为什么 inference 和 training 使用同一组 $\sigma$

严格地讲，在 bridge matching / stochastic interpolant 框架下，inference SDE 的噪声强度是一个**独立的 hyperparameter**，不必等于训练 bridge 的 $\sigma$（参见 Albergo–Vanden-Eijnden 的 stochastic interpolant 论文）。但为了 level 定义的简洁，我们默认 $\sigma^{\text{train}} = \sigma^{\text{inference}}$。如需 tune 可以加 config 字段 `bridge.inference_sigma_scale`。

### 5. 数据 prior 保留 `IGSO(3)(σ=1.5)` 以规避 cut locus

训练时 $R_0 \sim \text{IGSO(3)}(\sigma=1.5)\cdot R_1$，$R_0$ 集中在 $R_1$ 附近，$|\omega_i|\ll\pi$ 高概率成立，远离 cut locus。推断时 $R_0$ 从 uniform SO(3) 采样，可能远离目标，但推断过程**不显式计算 $\omega$**（只逐步推进 $R_t \to \hat R_1$），cut locus 不直接 break 推断。这是与 Level 1 一致的处理。

---

## Level 2 训练 / 验证 checklist

1. **Corruption 正确性**：打印 `R_t` 在 $t=0$、$t=1$ 处的恒等性（分别应为 $R_0, R_1$）；检查 $\Omega_t$ 的 mean / var 是否匹配 $\mathcal{N}(t\omega, \sigma_\omega^2 t(1-t))$
2. **$\sigma$ 标定**：第一个 epoch 后检查 drift residual $\|x_1 - \hat x_1\|$ 和 $\|\mathrm{Log}(R_1^\top \hat R_1)\|$ 的分布，调整 `sigma_x, sigma_omega` 到典型残差的 0.5–1.5 倍
3. **退化检查**：设 `sigma_x=0, sigma_omega=0` 运行，结果应**严格等于** Level 1 baseline（确认 Level 2 是 Level 1 的严格超集）
4. **$t$ 采样边界**：确认 `t ∈ [min_t, 1 - min_t]`，避免端点处 $(1-t)^{-1}$ 发散
5. **推断 SDE 一致性**：推断使用的 $\sigma$ 和训练 corruption 使用同一组，全程物理单位（Å / rad），无单位转换

---

## 开放问题

1. **Inference 噪声调度**：固定 $\sigma$ vs 按 $t$ 退火 vs 独立 hyperparameter
2. **Level 3 的 $\sigma \leftrightarrow$ b-factor 物理映射**：translation 和 rotation 分别如何归一化并可加到 B-factor
3. **RNA b-factor 数据覆盖率**：NMR 结构如何处理

---

> **历史说明**：旧 Level 2（BridgeWidthNet-based heteroscedastic NLL）和旧 Level 3（NLL + B-factor MSE anchor）设计的完整描述已从此文档移除。如需查阅，参见 git log 中 commit `bee0f44 Level 2 heteroscedastic bridge: BridgeWidthNet + NeurIPS draft` 之前的版本。
