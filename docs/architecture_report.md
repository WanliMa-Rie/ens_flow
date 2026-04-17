# 架构报告：EnsFlow — 三级 SE(3) RNA 构象集合生成模型

## 概述

本报告描述 EnsFlow 的三级模型架构。三级之间逐步增强：Level 1 是确定性 SE(3) 流匹配；Level 2 把前向过程升级为**切空间提升的 SE(3) 随机桥** (tangent-space-lifted stochastic bridge)，引入闭式、端点严格的随机过程；Level 3 在此基础上用序列条件网络预测物理 B-factor，让桥宽度变成残基级的物理量。

| Level | 前向过程（训练 corruption） | 训练 loss | 辅助模块 | 推断 |
|-------|---------------------------|-----------|----------|------|
| 1 | Deterministic geodesic on SE(3) | FM (endpoint) | — | ODE (Euler) |
| 2 | **SE(3) Stochastic Bridge** (切空间提升，固定 $\sigma$) | **SFM** (endpoint + $\delta/(2t)$ 修正) | — | ODE (Euler) |
| 3 | SE(3) Stochastic Bridge with 残基级 $\sigma = \bar\sigma\sqrt{\nu}$ | **SFM** ($\delta$ 含 $\nu$ 缩放) | Pair-biased `FlexibilityNet` + B-factor 监督 | bridge-consistent SDE (Euler-Maruyama, $\nu$ 调制噪声) |

三级通过 `stochastic_bridge.level` 字段切换，共享同一套 FlowModel 漂移网络。**Level 1 使用确定性 flow matching loss；Level 2-3 使用 Stochastic Flow Matching (SFM) loss**，其 velocity target 包含 bridge noise 修正项（§2.5, §2.7）。Level 3 在 $\nu \equiv 1$ 的初始化下**严格退化**为 Level 2，可以从 Level 2 checkpoint 热启动。

> **记号约定**：下文所有公式（除显式求和外）默认都是对单个残基 (即固定一个 residue) 的操作，因此**省略残基下标 $i$**。$x, R, \omega, \Omega, \nu, a, s$ 等记号均隐含"per-residue"含义。显式求和时使用 $\sum_{k=1}^{L}$ 作为遍历残基的 dummy index。

---

## Level 1：SE(3) Flow Matching

纯确定性流匹配。**时间约定**：$t=0$ 为先验（噪声），$t=1$ 为 clean data；$\hat{x}_{1|t}$、$\hat{R}_{1|t}$ 表示 FlowModel 在时刻 $t$ 给出的对 clean endpoint $(x_1, R_1)$ 的预测（对应代码中 `pred_trans` / `pred_rotmats`）。诱导的速度场：

$$u^x(t) = \frac{\hat{x}_{1|t} - x_t}{1 - t}, \qquad u^\omega(t) = \frac{1}{1-t}\mathrm{Log}(R_t^\top \hat{R}_{1|t})^\vee$$

训练时 `corrupt_batch` 做确定性插值：

$$x_t = (1-t)\, x_0 + t\, x_1, \qquad R_t = R_0\cdot\mathrm{Exp}(t\,\hat{\omega}),\; \omega = \mathrm{Log}(R_0^\top R_1)^\vee$$

分别对应 `_corrupt_trans` 的线性插值和 `_corrupt_rotmats` 调用 `so3_utils.geodesic_t(t, rotmats_1, rotmats_0)`。推断时用 Euler ODE 积分 (`interpolant.sample`)。

---

## Level 2：SE(3) Stochastic Bridge (Tangent-Space Lifted)

### 2.1 动机

Level 1 的前向过程是完全确定的 —— 给定 $(R_0, R_1, t)$，中间态 $R_t$ 唯一确定。我们希望引入随机性让前向过程成为一个从 $\rho_0$ 到 $\rho_1$ 的 **stochastic bridge**，并同时保留三个性质：

1. **闭式采样**（simulation-free 训练）
2. **严格端点约束**（$t=0$ 必为 $R_0$，$t=1$ 必为 $R_1$）
3. **可解析的 drift**（能直接做 velocity matching）

$SO(3)$ 上由 Laplace–Beltrami 算子生成的**内禀** Brownian bridge 没有闭式（只能用 heat-kernel 的 series representation，参见 IGSO(3)-based diffusion 工作）。我们的做法是：**在李代数 $\mathfrak{so}(3)\cong\mathbb{R}^3$ 中定义闭式 Brownian bridge，再通过指数映射推前到 $SO(3)$**。这个构造得到的是一个 **chart-induced**（非内禀）的随机桥，但三条性质全部满足。

> **诚实的表述**：这不是 $SO(3)$ 上的 intrinsic Brownian bridge，而是一个依赖以 $R_0$ 为基点的图 (chart) 的切空间提升桥。对应的 marginal $p_t(R_t)$ 是围绕测地线的 wrapped Gaussian，不是 heat-kernel 分布。我们选择它是因为 tractability 高于 geometric purity。

### 2.2 平动桥（欧氏 Brownian Bridge）

标准欧氏 BB：

$$x_t = (1-t)\,x_0 + t\,x_1 + \sigma_x \sqrt{t(1-t)}\cdot\varepsilon^x, \qquad \varepsilon^x \sim \mathcal{N}(0, I_3)$$

等价的条件分布：

$$x_t\mid x_0, x_1 \sim \mathcal{N}\!\left((1-t)x_0 + t\,x_1,\; \sigma_x^2\, t(1-t)\,I_3\right)$$

### 2.3 转动桥（切空间 BB）

先在 $\mathfrak{so}(3)$ 中定义对数坐标下的相对旋转：

$$\omega = \mathrm{Log}(R_0^\top R_1)^\vee \in \mathbb{R}^3$$

然后在 $\mathfrak{so}(3)\cong\mathbb{R}^3$ 中做 Brownian bridge：

$$\Omega_t = t\,\omega + \sigma_\omega \sqrt{t(1-t)}\cdot\varepsilon^\omega, \qquad \varepsilon^\omega \sim \mathcal{N}(0, I_3)$$

最后通过指数映射推前到 $SO(3)$：

$$R_t = R_0\cdot\mathrm{Exp}(\hat{\Omega}_t)$$

### 2.4 端点约束与条件分布（命题）

**命题 1（端点约束）**：上述构造满足 $R_{t=0} = R_0$ 且 $R_{t=1} = R_1$。

*证明*：$t=0$ 时 $\Omega_0=0$，$\mathrm{Exp}(0)=I$，故 $R_0$ 保持。$t=1$ 时 $\Omega_1=\omega$，$\mathrm{Exp}(\hat{\omega}) = R_0^\top R_1$，故 $R_0\cdot(R_0^\top R_1) = R_1$。$\square$

**命题 2（闭式条件分布）**：

$$\Omega_t\mid R_0, R_1 \sim \mathcal{N}\!\left(t\,\omega,\; \sigma_\omega^2\, t(1-t)\, I_3\right)$$

$R_t$ 的条件分布是此 Gaussian 在 $\Phi_{R_0}(\Omega) = R_0\,\mathrm{Exp}(\hat{\Omega})$ 下的 pushforward —— 一个围绕测地线的 wrapped Gaussian。

### 2.5 Stochastic Flow Matching 速度场目标

$\Omega_t$ 满足 $\mathfrak{so}(3)$ 中的 BB SDE：

$$d\Omega_t = \frac{\omega - \Omega_t}{1-t}\,dt + \sigma_\omega\,dW_t$$

**与 Level 1 的关键区别**：Level 2 使用 stochastic interpolant，其条件速度场 (SFM velocity target) 不同于确定性插值的目标——包含一个依赖桥噪声实现的修正项。本节给出从 exact form 到代码一阶近似的完整推导。

定义 bridge noise：

$$\delta^x_t \triangleq x_t - \bigl[(1-t)x_0 + tx_1\bigr], \qquad \delta^\Omega_t \triangleq \Omega_t - t\omega$$

#### 2.5.1 $\mathbb{R}^3$ 平动 SFM velocity

对一般欧氏 stochastic interpolant $z_t = \mu_t + \sigma_t \varepsilon$，exact conditional velocity 为 $u_t = \dot\mu_t + (\dot\sigma_t / \sigma_t)(z_t - \mu_t)$。代入 $\mu_t = (1-t)x_0 + tx_1$, $\sigma_t = \sigma_x\sqrt{t(1-t)}$：

$$\dot\mu_t = x_1 - x_0, \qquad \frac{\dot\sigma_t}{\sigma_t} = \frac{1-2t}{2t(1-t)}$$

$$u_t^x(x \mid x_0, x_1) = (x_1 - x_0) + \frac{1-2t}{2t(1-t)}\cdot\delta^x_t$$

$\mathbb{R}^3$ 是平坦空间，此即 exact target，无近似。

#### 2.5.2 $SO(3)$ 转动：李代数中的 exact SFM velocity

因为桥在 $\mathfrak{so}(3)\cong\mathbb{R}^3$ 中是欧氏 BB（$\Omega_t = t\omega + \sigma_\omega\sqrt{t(1-t)}\varepsilon$），完全同样的 stochastic interpolant 公式给出李代数坐标 $\Omega_t$ 的 exact conditional velocity：

$$u_t^\Omega = \omega + \frac{1-2t}{2t(1-t)}\cdot\delta^\Omega_t$$

注意这仍然是**李代数坐标的时间导数** $\dot\Omega_t$，而非群上的 body velocity。

#### 2.5.3 $SO(3)$ 转动：exact body velocity（右 Jacobian）

群上的轨迹 $R_t = R_0 \mathrm{Exp}(\hat\Omega_t)$。由 Lie group 指数映射的标准关系，body-frame 角速度与李代数坐标导数之间的联系为：

$$R_t^\top \dot R_t = \widehat{J_r(\Omega_t)\,\dot\Omega_t}$$

其中 $J_r(\Omega_t)$ 是 $SO(3)$ 的右 Jacobian。因此，**理论完备的** $SO(3)$ exact SFM target 为：

$$\boxed{a_t^\star = J_r(\Omega_t)\left[\omega + \frac{1-2t}{2t(1-t)}\,\delta^\Omega_t\right]}$$

这个 target 与当前 forward path 完全严格对齐，且与 $\mathbb{R}^3$ 的 SFM 逻辑一致（先在李代数写 stochastic interpolant velocity，再通过群几何推到 body frame）。

#### 2.5.4 一阶近似推导

代码使用 endpoint parameterization $v_\theta^\omega = \mathrm{Log}(R_t^\top \hat R_{1|t})^\vee / (1-t)$，以下推导将 exact target 化为此参数化下的 code-friendly form。

**Step 1. $J_r \approx I$**：当 $\|\Omega_t\| \ll \pi$ 时（训练 prior `IGSO(3)(σ=1.5)` 保证高概率成立），$J_r(\Omega_t) = I + O(\|\Omega_t\|)$，取一阶：

$$(1-t)\,a_t^\star \approx (1-t)\omega + \frac{1-2t}{2t}\,\delta^\Omega_t$$

**Step 2. 分解 $(1-t)\omega$**：由 $\Omega_t = t\omega + \delta^\Omega_t$ 得 $\omega - \Omega_t = (1-t)\omega - \delta^\Omega_t$，故 $(1-t)\omega = (\omega - \Omega_t) + \delta^\Omega_t$。代入并合并 $\delta^\Omega_t$ 系数（$1 + (1-2t)/(2t) = 1/(2t)$）：

$$(1-t)\,a_t^\star \approx (\omega - \Omega_t) + \frac{\delta^\Omega_t}{2t}$$

**Step 3. BCH 一阶截断**：$R_t^\top R_1 = \mathrm{Exp}(-\hat\Omega_t)\,\mathrm{Exp}(\hat\omega)$。BCH 一阶下 $\mathrm{Log}(\mathrm{Exp}(-\hat\Omega_t)\mathrm{Exp}(\hat\omega))^\vee \approx \omega - \Omega_t$。于是得到**代码友好的一阶近似 target**：

$$\boxed{(1-t)\,a_t^\star \approx \mathrm{Log}(R_t^\top R_1)^\vee + \frac{\delta^\Omega_t}{2t}}$$

这正是 exact target 经过 (i) $J_r \approx I$, (ii) BCH 一阶截断, (iii) 不同切空间 identity-style 识别后的实现形式。近似阶 $O(\|\Omega_t\|^2)$ 与现有 velocity target 本身的 BCH 近似阶一致。

#### 2.5.5 与代码 endpoint 参数化的对应

综合以上推导，模型预测 $(\hat x_{1|t}, \hat R_{1|t})$，SFM loss 的 target 为：

- **$\mathbb{R}^3$ (exact)**：$\hat x_{1|t}^{\star} = x_1 + \delta^x_t / (2t)$
- **$SO(3)$ (一阶近似)**：$\mathrm{Log}(R_t^\top R_1)^{\vee,\star} = \mathrm{Log}(R_t^\top R_1)^\vee + \delta^\Omega_t / (2t)$
- **$SO(3)$ (exact, 供论文使用)**：$(1-t)\,a_t^\star = (1-t)\,J_r(\Omega_t)\bigl[\omega + \tfrac{1-2t}{2t(1-t)}\delta^\Omega_t\bigr]$

Level 1 的 velocity target 是上述公式在 $\sigma \to 0$（$\delta = 0$）时的特例。

### 2.6 Corruption 实现（`interpolant.py`）

Level 2 的代码改动：(a) `_corrupt_rotmats_bridge` 在切空间做 BB 采样并 push-forward 到 SO(3)；(b) `_corrupt_trans_bridge` 在线性均值上加 BB 噪声。两个函数都接受可选的 `nu`（Level 3 用），`nu=None` 时退化为常数 $\sigma$ 的 Level 2 桥。**噪声是 per-residue 独立的**——不做 CoM 去均值；引入的 CoM 偏移量级 $O(\bar\sigma\sqrt{t(1-t)}\langle\sqrt\nu\rangle/\sqrt{m_R})$，在典型尺度下次于噪声地板，且 SE(3) 下游对小幅 CoM 漂移不敏感。

```python
def _corrupt_trans_bridge(self, trans_1, t, res_mask, sigma_x, nu=None):
    trans_0 = self._batch_ot(_centered_gaussian(*res_mask.shape, ...) * NM_TO_ANG,
                             trans_1, res_mask)
    mean = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1   # [B, N, 3]
    std  = sigma_x * torch.sqrt(t * (1 - t) + 1e-8)                # [B, 1]

    eta_factor = torch.ones_like(res_mask) if nu is None \
                 else torch.sqrt(nu.clamp(min=0.0) + 1e-12)        # [B, N]
    bridge_noise = std[..., None] * eta_factor[..., None] * torch.randn_like(mean)
    trans_t = mean + bridge_noise
    trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
    return trans_t * res_mask[..., None], bridge_noise * res_mask[..., None]

def _corrupt_rotmats_bridge(self, rotmats_1, t, res_mask, sigma_omega, nu=None):
    rotmats_0 = self._sample_rotmats_0(rotmats_1, res_mask)        # IGSO3(σ=1.5) · R_1
    omega     = so3_utils.calc_rot_vf(rotmats_0, rotmats_1)        # [B, N, 3]
    std       = sigma_omega * torch.sqrt(t * (1 - t) + 1e-8)       # [B, 1]
    std_bn    = std if nu is None \
                else std * torch.sqrt(nu.clamp(min=0.0) + 1e-12)   # [B, N]
    bridge_noise = torch.randn_like(omega) * std_bn[..., None]     # [B, N, 3]
    Omega_t   = t[..., None] * omega + bridge_noise                # [B, N, 3]
    delta_R   = so3_utils.rotvec_to_rotmat(Omega_t)                # exp map
    rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_0, delta_R)
    return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask), bridge_noise * res_mask[..., None]
```

注意 Level 1 现有代码的 rot corruption `so3_utils.geodesic_t(t, rotmats_1, rotmats_0)` 等价于 $R_0\cdot\mathrm{Exp}(t\hat\omega)$，也就是 BB 的**均值**。Level 2 只是在 $t\omega$ 上叠加了一个方差为 $\sigma_\omega^2 t(1-t)$ 的高斯扰动。这也意味着**设 $\sigma_\omega = 0$（且 trans 桥的 $\sigma_x = 0$）时 Level 2 corruption 与 Level 1 严格 bit-identical**（modulo 后者不走 BB 噪声路径而前者走零噪声路径）。

> **关于 translation 重心化**：Level 1 的 OT-aligned prior 让每个 sample 的残基均值在零附近。Level 2/3 的 BB 噪声若按 IID 注入，会把这条性质打破——在 Level 3 异质 $\nu$ 下尤其明显。Post-multiply 重心化（先按 $\sqrt{\nu_i}$ 缩放噪声，再减去 resolved residues 上的均值）保证 $\sum_{i\in\mathcal R}\tilde\eta_i = 0$ 对任意 $\nu$ 成立，drift 网络看到的 corrupted batch 仍然分布在零均值附近。Level 2 (`nu=None`) 路径也走相同重心化，使代码路径统一；带来的额外修正是 $O(1/m_R)$，在数据集长度上低于 B-factor 噪声底线。

### 2.7 损失：Stochastic Flow Matching

Level 2 的训练损失与 Level 1 **不同**——target 包含 §2.5 推导的 bridge noise 修正项。本节给出 exact 和一阶近似两种 loss 形式。

#### 2.7.1 Exact theoretical loss

若直接使用 §2.5.3 的 exact body velocity $a_t^\star$，在 endpoint parameterization 下：

$$\mathcal{L}^{\text{exact}}_{\text{SFM}} = w_x \cdot \Big\|\frac{\hat x_{1|t} - x_t}{1-t} - u_t^x\Big\|^2 + w_\omega \cdot \Big\|\mathrm{Log}(R_t^\top \hat R_{1|t})^\vee - (1-t)\,J_r(\Omega_t)\Big[\omega + \frac{1-2t}{2t(1-t)}\delta^\Omega_t\Big]\Big\|^2$$

其中 $\mathbb{R}^3$ 部分的 SFM velocity $u_t^x$ 是 exact 的（不涉及近似）；$SO(3)$ 部分需要 $\Omega_t$, $\omega$ 和右 Jacobian $J_r(\Omega_t)$。

#### 2.7.2 一阶近似 loss（代码实现）

经 §2.5.4 的推导（$J_r \approx I$ + BCH 一阶），$\mathbb{R}^3$ 的 endpoint target 和 $SO(3)$ 的 tangent-vector target 分别化为：

$$\hat x_{1|t}^{\star} = x_1 + \frac{\delta^x_t}{2t}, \qquad \mathrm{Log}(R_t^\top R_1)^{\vee,\star} = \mathrm{Log}(R_t^\top R_1)^\vee + \frac{\delta^\Omega_t}{2t}$$

代码实现的 loss：

$$\mathcal{L}_{\text{SFM}} = w_x \cdot \frac{1}{\mathrm{norm}^2}\Big\|\hat x_{1|t} - \Big(x_1 + \frac{\delta^x_t}{2t}\Big)\Big\|^2 + w_\omega \cdot \frac{1}{\mathrm{norm}^2}\Big\|v_\theta^\omega - \Big(\mathrm{Log}(R_t^\top R_1)^\vee + \frac{\delta^\Omega_t}{2t}\Big)\Big\|^2$$

其中 $\mathrm{norm} = 1 - \min(t,\, t_{\text{clip}})$，$w_x$ = `translation_loss_weight` (2.0)，$w_\omega$ = `rotation_loss_weights` (1.0)。辅助损失 $\mathcal{L}_{\text{bb\_atom}} + \mathcal{L}_{\text{dist\_mat}} + \mathcal{L}_{\text{torsion}}$ 与 Level 1 一致（不含 SFM 修正）。

**Level 1 退化**：当 $\sigma_x = \sigma_\omega = 0$ 时 $\delta = 0$，SFM loss 严格退化为 Level 1 的 endpoint / tangent-vector loss。

#### 2.7.3 实现要求

1. `corrupt_batch` 需额外存储 bridge noise $\delta^x_t$, $\delta^\Omega_t$ 到 batch dict
2. `model_step` 在 Level $\ge 2$ 时读取 bridge noise 并修正 target：`gt_trans_1 += delta_x / (2*t)`，`gt_rot_vf += delta_Omega / (2*t)`
3. Level 1 路径不受影响（bridge noise 不存在或为零）

> **论文写法建议**：正文给 exact form（§2.7.1 含 $J_r$），implementation / appendix 说明一阶近似（§2.7.2）及其来源（§2.5.4 的三步推导）。这样 reviewer 既能看到理论完备性，也能看到实现透明度。

### 2.8 推断：ODE drift 与 bridge-consistent 随机采样器

由于 Level 2 的训练目标采用的是 **stochastic flow matching (SFM)** 条件速度场，而非 Brownian bridge 的 drift 回归，因此在推断时，最一致、最理论自洽的做法是：**直接积分网络学习到的速度场**，而不是再额外注入固定扩散噪声。

具体地，给定初始状态 $(x_0, R_0)$，在离散时间网格 $0 = t_0 < t_1 < \cdots < t_K = 1$ 上做前向积分。每一步，FlowModel 接收当前状态 $(x_k, R_k, t_k)$，输出 clean endpoint prediction $(\hat x_{1|t_k}, \hat R_{1|t_k})$。由 endpoint parameterization 诱导的速度场写为

$$u_\theta^x(t_k) = \frac{\hat x_{1|t_k} - x_k}{1-t_k}, \qquad u_\theta^\omega(t_k) = \frac{1}{1-t_k}\mathrm{Log}(R_k^\top \hat R_{1|t_k})^\vee$$

随后采用 Euler 离散化推进状态：

$$x_{k+1} = x_k + \Delta t_k \cdot u_\theta^x(t_k)$$

$$R_{k+1} = R_k \cdot \mathrm{Exp}\!\left(\widehat{\Delta t_k \cdot u_\theta^\omega(t_k)}\right)$$

其中 $\Delta t_k = t_{k+1} - t_k$。最后一步得到的 $(x_K, R_K)$ 即为生成的 backbone frame；原子坐标和 torsion 角仍按 FlowModel 的标准解码头恢复。

这种 ODE 推断与训练中学习的 SFM velocity field 保持一致：训练阶段，随机性仅用于定义和采样 conditional stochastic interpolant；推断阶段，则通过积分学习到的平均速度场将先验样本推送到目标分布。因此，生成过程中的随机性主要来自初始状态 $(x_0, R_0)$ 的采样，而不是中途额外注入的扩散项。

#### 初始分布

平动初值从 centered Gaussian prior 采样，转动初值从 $SO(3)$ 上的先验分布采样：

$$x_0 \sim p_0^x, \qquad R_0 \sim p_0^R$$

在实现上，这与 Level 1 保持一致：translation 使用 centered Gaussian prior，rotation 使用 $SO(3)$ prior 采样。由此，Level 2 相比 Level 1 的主要差别体现在**训练时的随机 bridge corruption 和 SFM loss**，而不是推断器本身的形式。

#### Bridge-consistent stochastic sampler（Level $\geq$ 2）

纯 ODE 采样在实践中会出现 mode collapse：由于 drift 是确定性的，rollout 的多样性完全由初始状态 $(x_0, R_0)$ 的采样决定，而 FlowModel 的 SFM-corrected endpoint predictor 在中段 $t$ 就已经收敛到少数几个 attractor。为恢复 ensemble coverage，Level $\geq$ 2 在 drift 之上注入**训练带宽形状一致**的高斯探索噪声：

$$\alpha_i(t) = \sqrt{\nu_i}\,\sqrt{t(1-t)}$$

平移 / 转动更新：

$$x_{k+1,i} = x_{k,i} + \Delta t_k\, u_\theta^x + \sigma_x\,\alpha_i(t_k)\sqrt{\Delta t_k}\,\tilde\xi^x_{k,i}$$

$$R_{k+1,i} = R_{k,i} \cdot \mathrm{Exp}\!\left(\widehat{\Delta t_k\, u_\theta^\omega + \sigma_\omega\,\alpha_i(t_k)\sqrt{\Delta t_k}\,\xi^R_{k,i}}\right)$$

其中 $\tilde\xi^x$ 是 per-residue 独立的高斯噪声（不做 CoM 去均值，与 corruption 一致）。关键性质：

- **Bridge envelope 匹配**：$\alpha_i(t)$ 的 $\sqrt{t(1-t)}$ 形状与训练 corruption 严格一致，噪声在两端自动消失；最后一步自动 snap 到 $\hat x_{1|t}, \hat R_{1|t}$。每单位时间方差与训练桥严格相等，因此是 "bridge-consistent"。
- **严格嵌套**：$\nu_i \equiv 1$ 退化为 homoscedastic Level 2 sampler。
- **Flexibility-aware**：Level 3 下 $\nu_i$ 由 `FlexibilityNet` 从 single/pair embeddings 预测，一次 per sample（纯 sequence-conditioned，与 $t$、状态无关）；较 flexible 的残基获得更宽的 stochastic exploration。
- **解释**：建立在 flow drift 之上的 "bridge-shaped annealed Langevin"，不是对 forward bridge SDE 的逐时刻精确 reverse，但与训练 corruption 的方差谱在每单位时间上严格对齐。

#### 实现备注

- `Interpolant.sample(..., nu)` 单一入口，由 `self._level` 决定走哪条分支：`level == 1` 走 `_trans_euler_step` / `_rots_euler_step`（确定性 ODE）；`level >= 2` 走 `_bridge_sde_step`（Euler-Maruyama）。无 `tau` 参数。
- `FlexibilityNet` 只在 Level 3 checkpoint 时被调用，缓存 $\nu$ 一次，不进入逐步循环。Level 2 调用时传 `nu=None`，相当于 $\nu_i \equiv 1$。
- `rots.sample_schedule` 在 inference 下默认 `linear`（$\Delta t_k / (1 - t_k)$），与 SFM loss 的线性 velocity 形式对齐；旧的 `exp_rate=10` 恒定 scaling 在本推断器下已被移除。

### 2.9 Level 2 与 Level 1 的关系

- **$\sigma_x = \sigma_\omega = 0$** → BB 退化为确定性测地线，SDE 退化为 ODE。Level 2 **严格包含** Level 1。
- **$\sigma$ 相关的训练改动** 有两处：(a) `corrupt_batch` 里两项高斯噪声；(b) loss target 的 SFM 修正 $\delta/(2t)$（§2.7）。网络结构、优化器组不变。
- **参数量**：与 Level 1 相同 ($\approx 14.25$ M)，没有新增任何模块。

### 2.10 关键假设与 caveats

- **Cut locus**：$|\omega| < \pi$ 假设 principal log 单值。训练 prior 为 `IGSO(3)(σ=1.5) · R_1`（现有代码），$R_0$ 集中在 $R_1$ 附近，$|\omega|\ll\pi$ 高概率成立，基本不碰 cut locus。推断 prior 是 uniform SO(3)，但推断循环**不显式计算 $\omega$**（只逐步推进 $R_t$ 到 $\hat R_1$），cut locus 不直接触发。
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

Level 2 的 $(\bar\sigma_x, \bar\sigma_\omega)$ 是全局常数，所有残基共享同一个桥宽。RNA 骨架的物理柔性是**残基级**的 —— 保守区 stiff、loop 区 flexible —— 这种柔性由实验 B-factor 直接测量 ($B \propto \langle u^2 \rangle$)。Level 3 引入一个**序列条件、残基级、时间无关**的 flexibility multiplier $\nu(s) > 0$，把 Level 2 的固定桥宽升级为 residue-wise 桥宽，再用 B-factor 做监督。

**两条强设计原则**：

1. **Corruption + training 层面严格嵌套 (L2 ⊂ L3)**：令 $\nu \equiv 1$（对所有残基），Level 3 的 **corruption** 张量 $(x_t, R_t)$ 和 **SFM velocity target** 都和 Level 2 逐 bit 相同（不是"approximately similar"，是严格退化）。推断端三级共享同一 ODE 采样器（§2.8）。**注意**：完整的 Level 3 训练目标是 $\mathcal{L}_{\text{level3}} = \mathcal{L}_{\text{bridge}} + \lambda_B \mathcal{L}_B + \lambda_\nu \mathcal{L}_\nu$，比 Level 2 多两项；这两项的梯度通过 `nu.detach()` stop-grad firewall 只回到 $\psi$，不回到 FlowModel，所以 FlowModel 的训练动力学和 Level 2 一致。这是"严格嵌套于 corruption + SFM target 层面，objective 层面通过 stop-grad 实现等价"的表述，而不是"objective 也逐 bit 相同"。
2. **Identifiability**：每个残基只有一个标量 B-factor，因此只应该学一个标量 $\nu$。把 translation width 和 rotation width 通过**同一个** $\nu$ 联合缩放，避免 "translation 变大/rotation 变小" 这种不可辨识的自由度。

### 3.2 Flexibility head

给定 per-residue 条件特征 $s \in \mathbb{R}^{c_\text{single}}$ 和 pair 特征 $p \in \mathbb{R}^{L \times L \times c_\text{pair}}$（直接用 dataset 的 `single_embedding` / `pair_embedding`，不和 FlowModel backbone 共享），定义一个 head $g_\psi: (s, p) \mapsto \mathbb{R}^L$ 和 per-sample 归一化：

$$a = \mathrm{softplus}(g_\psi(s, p)) + \varepsilon, \qquad \nu = \frac{a}{\frac{1}{L_R}\sum_{k\in\mathcal{R}} a_k}$$

其中 $\varepsilon > 0$ 是数值稳定常数，$\mathcal{R}$ 是当前样本的 resolved residues 集合 (`res_mask == 1`)，$L_R = |\mathcal{R}|$；$\sum_{k\in\mathcal{R}} a_k$ 中的 $k$ 是遍历 resolved 残基的 dummy index。

> **Mask contract（与 corruption / 推断对齐）**：head 内部的 mean-1 归一化用 **resolved mask** (`res_mask`)。理由：bridge corruption 只对 `res_mask == 1` 的位置真正注入噪声（unresolved 位置在最后一步乘 `res_mask` 被零掉），所以"$\nu$ 实际生效的位置集"就是 resolved 子集；让 head 在同一子集上做 mean-1 归一化，bridge 的有效 $\bar\sigma\sqrt{\nu}$ 在该子集上的均值就严格等于 Level 2 的 $\bar\sigma$（保 scale invariance）。在 inference / `predict_step` 上，输入序列里**不存在 unresolved**（一切位置都是要生成的），所以传 all-ones mask 等价于"全是 resolved"，与训练 contract 完全一致；validation 也走同一 contract（剥离 padding 后用 `eval_mask = res_mask[na_mask]`）。代码上 `flow_module.training_step` / `validation_step` / `predict_step` 三处统一遵守这条。

**关键性质**：

- **Mean-1 归一化**：$\frac{1}{L}\sum_{k=1}^L \nu_k \equiv 1$ 对任意 $\psi$、任意 batch 恒成立，不仅是 init 时
- **退化条件**：$a$ 在所有残基上取同一常数 (例如 $\psi$ 的最后一层 weight 全零) $\Rightarrow$ $\nu \equiv 1$。Level 3 在 init 时严格等同于 Level 2
- **全局尺度消除**：$\nu$ 只刻画 "residue 之间的相对 flexibility"，全局 sigma scale 仍由 $(\bar\sigma_x, \bar\sigma_\omega)$ 控制，避免 gauge 自由度

Head 实现：`LayerNorm + Linear` 把 $s$ 投到 hidden 维，**一个 block 的 pair-biased multi-head self-attention**（pair embedding 经过 `LayerNorm + Linear` 投成 per-head 的 additive attention bias，类似 IPA trunk 消费 pair feature 的方式）混合残基信息，最后接一个 scalar Linear 头读出 $g_\psi$。最后一层 `Linear` **零初始化 weight**、bias 设 `log(e-1) ≈ 0.5413`（这样 softplus 输出 1），保证 $\nu \equiv 1$ 在 init 严格成立——无论 attention block 的随机初始化是什么，因为零权重的最后一层让 attention 的输出根本进不来。

### 3.3 Residue-wise bridge

**平动桥**：Level 2 的固定 $\bar\sigma_x$ 换成 $\bar\sigma_x\sqrt{\nu}$：

$$x_t = (1-t)x_0 + t\,x_1 + \bar\sigma_x \sqrt{\nu}\sqrt{t(1-t)}\,\varepsilon^x,\quad \varepsilon^x \sim \mathcal{N}(0, I_3)$$

**转动桥**：同理用 $\bar\sigma_\omega\sqrt{\nu}$：

$$\omega = \mathrm{Log}(R_0^\top R_1)^\vee, \quad \Omega_t = t\,\omega + \bar\sigma_\omega\sqrt{\nu}\sqrt{t(1-t)}\,\varepsilon^\omega, \quad R_t = R_0\cdot\mathrm{Exp}(\hat\Omega_t)$$

Level 3 相对 Level 2 的**唯一结构性改动**：

$$\bar\sigma_x^2 \mapsto \nu\, \bar\sigma_x^2, \quad \bar\sigma_\omega^2 \mapsto \nu\, \bar\sigma_\omega^2$$

Bridge family 不变，endpoint 严格性不变。SFM velocity target 的公式形式与 Level 2 相同（§3.4），$\nu$ 的影响通过 bridge noise $\delta$ 自动编码。

### 3.4 Velocity target：SFM correction 自动包含 $\nu$

Level 3 使用与 Level 2 完全相同的 SFM velocity target 形式（§2.5）：

$$u_t^x = (x_1 - x_0) + \frac{1-2t}{2t(1-t)}\cdot\delta^x_t, \qquad u_t^\Omega = \omega + \frac{1-2t}{2t(1-t)}\cdot\delta^\Omega_t$$

$\nu$ 的影响**已编码在 bridge noise $\delta$ 中**：Level 3 的 corruption 使用 $\bar\sigma\sqrt{\nu}$ 生成噪声，因此 $\delta^x_t, \delta^\Omega_t$ 自动按残基缩放。SFM correction $\delta/(2t)$ 继承了这一缩放——高 $\nu$ 残基的 correction 大，低 $\nu$ 残基的 correction 小。

不需要为 Level 3 引入额外的 target 修改；Level 2 和 Level 3 共享同一套 SFM loss 代码路径。$\nu \equiv 1$ 时 Level 3 的 bridge noise 与 Level 2 逐 bit 相同，SFM correction 也相同（严格嵌套）。

> **关于 Jr-correction**。exact SFM target 的完整形式为 $a_t^\star = J_r(\Omega_t)[\omega + \tfrac{1-2t}{2t(1-t)}\delta^\Omega_t]$（§2.5.3）。代码使用 BCH 一阶近似 $J_r \approx I$（§2.5.4），与 FrameFlow / FoldFlow 基线一致。

### 3.5 B-factor alignment loss

**Bridge-induced MSD 推导**（appendix proof sketch D）：设残基的局部 frame-defining 模板原子集合为 $\mathcal{A}$，原子 $j$ 的 local 坐标 $q_j\in\mathbb{R}^3$。小角度近似下：

$$\delta y_j(t) \approx \delta x(t) + \delta\theta(t) \times q_j$$

其中 $\delta x(t) \sim \mathcal{N}(0,\, \nu\bar\sigma_x^2\, t(1-t) I_3)$，$\delta\theta(t) \sim \mathcal{N}(0,\, \nu\bar\sigma_\omega^2\, t(1-t) I_3)$。取原子平均与时间平均 ($\int_0^1 t(1-t)\,dt = 1/6$)：

$$\overline{\mathrm{MSD}} = \frac{\nu}{6}\Big(3\bar\sigma_x^2 + 2\bar\sigma_\omega^2\, c\Big), \qquad c = \frac{1}{|\mathcal{A}|}\sum_{j\in\mathcal{A}}|q_j|^2$$

（$j$ 是遍历该残基的 frame atoms 的 dummy index。）用标准各向同性关系 $B = \frac{8\pi^2}{3}\mathrm{MSD}$ 定义 bridge uncertainty proxy：

$$\widehat{B}^{\text{bridge}} = \frac{4\pi^2}{9}\nu\Big(3\bar\sigma_x^2 + 2\bar\sigma_\omega^2\, c\Big) \equiv \kappa\, \nu$$

**代码简化**：对 RNA frame atoms $[\text{C3}', \text{C4}', \text{O4}']$，从 `nucleotide_constants.nttype_rigid_group_default_frame` 读出的 local 坐标得到

$$c \approx \tfrac{1}{3}\Big(|\text{C3}'|^2 + |\text{C4}'|^2 + |\text{O4}'|^2\Big) \approx 1.473\ \text{Å}^2$$

**对所有 residue 是同一个常数**（A/U/G/C 的差别 $<10^{-3}$ Å，完全可忽略）。量级检查：$3\bar\sigma_x^2 = 12\ \text{Å}^2$ vs $2\bar\sigma_\omega^2\, c \approx 0.18\ \text{Å}^2$，translation 贡献主导。因此 $\kappa$ 也是与残基无关的全局常数。

**归一化 B-factor target**（per-sample，消除 resolution/refinement scale 差异）。在 resolved 子集 $\mathcal{R}$ 上写第 $k$ 个残基的归一化 B-factor $b_k$：

$$b_k = \frac{B_k}{\frac{1}{L_R}\sum_{j\in\mathcal{R}} B_j}, \qquad \widehat{b}_k = \frac{\kappa\,\nu_k}{\frac{1}{L_R}\sum_{j\in\mathcal{R}} \kappa\,\nu_j} = \frac{\nu_k}{\frac{1}{L_R}\sum_{j\in\mathcal{R}} \nu_j} = \nu_k$$

（最后一步用到 $\kappa$ 常数 + $\nu$ 已经在 $\mathcal{R}$ 上 mean-1 归一化，§3.2）

**Loss 最终形式（对 valid-B mask 重归一化的 asymptotic surrogate）**：

设 $\mathcal V$ = valid-B set（resolved AND $b > b_{\min}$ AND 当前 sample 通过 std gate $\mathrm{std}(b_{\text{valid}}) > s_{\min}$），在 loss 内部对 $\nu$ 在 $\mathcal V$ 上重做 mean-1 归一化：

$$\nu^{(B)}_i = \frac{\nu_i}{\frac{1}{|\mathcal V|}\sum_{j\in\mathcal V}\nu_j}, \qquad \hat b_i = \frac{b_i}{\frac{1}{|\mathcal V|}\sum_{j\in\mathcal V} b_j}$$

$$\mathcal{L}_B = \mathbb E_{\text{good samples}}\Bigl[\frac{1}{|\mathcal V|}\sum_{i\in\mathcal V}\bigl(\log\nu^{(B)}_i - \log\hat b_i\bigr)^2\Bigr]$$

**关于 $\hat b_i = \nu^{(B)}_i$ 的精度**：在 constant-$\kappa$ 假设下，这个 identity 是 $m_R$ 的 asymptotic limit，带 $O(1/m_R)$ 修正。修正系数在 `ensemble_dataset2` 长度（$m_R \in [30, 300]$）上远低于晶体学 B-factor 的 refinement noise floor ($\pm 10\text{--}20\%$)。

代码上：bridge corruption 用 resolved-mask (`res_mask`) 归一化的 $\nu$（与 §3.2 一致）；`L_B` 内部的双重归一化 (`nu_validB`) 是纯 loss-internal 操作，不影响 corruption。$c$ 和 $\kappa$ 在 loss 里不出现，只用于 writeup 的物理动机。

### 3.6 总 loss

$$\mathcal{L}_{\text{level3}} = \mathcal{L}_{\text{bridge}} + \lambda_B \mathcal{L}_B + \lambda_\nu \mathcal{L}_\nu$$

其中：

- $\mathcal{L}_{\text{bridge}}$ = Level 2 的 SE(3) SFM loss（含 bridge noise correction 的 translation + rotation VF + backbone atom + dist mat + torsion，参见 §2.7）
- $\mathcal{L}_B$ = §3.5 的 B-factor alignment
- $\mathcal{L}_\nu = \dfrac{1}{N_{\text{resolved-batch}}}\sum_{i\in\text{resolved}} (\log\nu_i)^2$ = **batch-pooled**（不是 per-sample 平均）的 $\log$-space 正则，分母是整个 batch 上的 resolved residue 数，分子在所有 resolved 残基上求和。很弱的正则，防止 $\nu$ 过度尖锐；因为 $\nu$ 已经 mean-1 归一化，这项也自动中心化

### 3.7 Training protocol：joint training + always stop-grad

**单阶段联合训练**：解冻所有参数，目标 $\mathcal{L}_{\text{level3}} = \mathcal{L}_{\text{bridge}} + \lambda_B\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$。可以从 Level 2 checkpoint 热启动，也可以从零起 —— 因为 $\nu \equiv 1$ 在 init 严格成立（§3.2 零初始化），FlowModel 在前几步看到的就是 Level 2 的 corruption。

**Stop-gradient 协议（始终启用）**：在 `corrupt_batch` 里使用 `nu.detach()` 作为 bridge width，因此 $\mathcal{L}_{\text{bridge}}$ 不会通过 bridge sampling 路径回传到 $\psi$。$\psi$ 只由 $\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$ 更新。这避免了 velocity loss "偷偷缩小 $\nu$ 让 MSE 变小" 的退化。代码上的体现：

```python
# nu mean-1 归一化用 res_mask（与 corruption / 推断对齐，§3.2）
nu_raw = self.flexibility_net(
    batch["single_embedding"],
    batch["pair_embedding"],
    batch["res_mask"].to(batch["single_embedding"].dtype),
)  # grad → ψ

# bridge sampling 用 detached nu: L_bridge 不回传 ψ
noisy_batch = self.interpolant.corrupt_batch(batch, nu=nu_raw.detach())

# flow model forward → L_bridge
batch_losses = self.model_step(noisy_batch)           # grad → FlowModel
train_loss = batch_losses[loss_key] + batch_losses["auxiliary_loss"]

# 用非 detached nu 算 L_B / L_nu：只更新 ψ
L_B, L_nu, n_good = self._compute_flex_losses(nu_raw, batch["res_mask"], batch["b_factors"])
train_loss = train_loss + λ_B · L_B + λ_ν · L_nu
```

这样 FlowModel 只受 $\mathcal{L}_{\text{bridge}}$ 梯度，FlexibilityNet 只受 $\mathcal{L}_B + \lambda_\nu\mathcal{L}_\nu$ 梯度。`configure_optimizers` 把两组参数都加进同一个 AdamW group，无需 phase 切换。

### 3.8 Inference：bridge-consistent SDE（与 Level 2 共享）

Level 3 走 §2.8 的 bridge-consistent Euler-Maruyama，$\nu$ 通过 FlexibilityNet 显式调制 per-residue 噪声振幅：

$$\alpha_i(t) = \sqrt{\nu_i}\,\sqrt{t(1-t)}, \quad x_{k+1,i} = \cdots + \bar\sigma_x\,\alpha_i(t_k)\sqrt{\Delta t_k}\,\tilde\xi^x_{k,i}, \quad \Omega_{k+1,i} = \cdots + \bar\sigma_\omega\,\alpha_i(t_k)\sqrt{\Delta t_k}\,\xi^R_{k,i}$$

其中 $\tilde\xi^x$ 是 per-residue 独立的高斯噪声（与 corruption 一致，不做 CoM 去均值）。FlexibilityNet 在采样开始前 forward 一次得到 $\nu$（缓存，不进逐步循环）；padding-stripped 后所有输入位置都视为 resolved（all-ones mask），与 §3.2 contract 一致。

**与 Level 2 的区别**：仅在于 $\nu_i$ 是否非 1。Level 2 调用 `Interpolant.sample(nu=None)`，等价于 $\nu_i \equiv 1$；Level 3 把 FlexibilityNet 的输出传进去，让噪声振幅按残基调制。两级共用 `_bridge_sde_step`，无单独的"模式选择"。

**与 Level 1 的区别**：Level 1 走纯 ODE（不进 `_bridge_sde_step`），无任何噪声注入。`Interpolant.sample` 内部按 `self._level` 切分支。

### 3.9 B-factor 数据与 NMR/cryo-EM fallback

- **数据源**：`build_datasets.py` 从 PDB `atom.bfactor` 解析，聚合到 C4' 原子得到 residue-level `b_factors`，通过 `conformer_collate` 以 `[B, N]` 形式进入 batch
- **unresolved residues**：`b_factors[unsolved] = 0.0`，通过 `res_mask == 0` 隐式标识
- **NMR / cryo-EM fallback（三层 sample-level gate）**：现代 PDB 里 NMR 结构的 b-factor 通常全为 0；某些 cryo-EM 结构则有 refined-but-uniform 的 B-factor。Level 3 在 `L_B` 计算前做 per-residue + per-sample gating（对应 code `_compute_flex_losses`）：
  - **Residue 级**：只对 `res_mask == 1 AND b_factors > 0` 的 residue 计 loss（unresolved 位置由数据管线写成 0，因此 `> 0` 自然把它们筛掉，无需额外阈值参数）
  - **Sample 级 (a)**：valid residues $\ge 2$
  - **Sample 级 (b)**：$\mathrm{mean}(b_{\text{valid}}) > 0$
  - **Sample 级 (c)**：$\mathrm{std}(b_{\text{valid}}) > \texttt{min\_std\_bfactor}$ —— 这条 std gate 把 uniform-positive 的 cryo-EM 样本筛掉，避免它们把 $\nu$ 的监督信号稀释成 $\nu \to 1$。参数 `bfactor.min_std_bfactor`，缺省值 `0.5` $\text{Å}^2$（通过 `getattr` 读取，老 config 不会 break）
  - 任何一条 sample-level 检查不过，该样本的 `L_B = 0`，不进 backward
- **缺失环（missing-loop）残基**：`res_mask = 0` 的位置在训练中不计入 `L_B`，但 inference 时 head 仍会从 `(single_embedding, pair_embedding)` 输出 $\nu_i$。Pair 上下文比纯 sequence-only 多一些信息，但这些位置在训练时仍未被 $\mathcal{L}_B$ 监督——本质上仍是 extrapolation，不是 supervised interpolation，需要在 limitations 里说清楚
- **各向异性 B-factor**：晶体学的 anisotropic refinement 在数据 pipeline 阶段已经被 collapse 成 isotropic；这是数据层面的限制，不是 anchor 设计的限制
- **$\lambda_B$ 标定**：手动通过 single-batch dry-run 调整，目标是 Stage 1 末 $\lambda_B \mathcal{L}_B \approx 0.1\,\mathcal{L}_{\text{bridge}}$。**不存在** dataset-level $\lambda_B$ warm-start 自动机制（早期草案曾设想过，但 code 里没有实现）

### 3.10 与 Level 2 的嵌套关系（corruption + SFM target 层面 strict，objective 层面 stop-grad 等价）

- $\psi$ 最后一层 weight $= 0$ + bias $= \log(e-1)$ $\Rightarrow$ $\mathrm{softplus}(\mathrm{bias}) = 1$ $\Rightarrow$ $a_i \equiv 1 + \varepsilon$ $\Rightarrow$ $\nu_i \equiv 1$ exactly after mean-1 normalization
- 此 init 状态下，**bridge corruption** 张量 $(x_t, R_t)$ 与 Level 2 bit-identical（相同 seed 下逐元素相等）
- 此 init 状态下，**SFM velocity target**（含 $\delta/(2t)$ 修正）与 Level 2 bit-identical
- 此 init 状态下，**推断** 三级共享同一 ODE 采样器，输出 bit-identical
- 此 init 状态下 $\log \nu = 0$，$\mathcal{L}_\nu = 0$
- $\mathcal{L}_B$ 在此 init 状态下 $= \dfrac{1}{|\mathcal V|}\sum_{i\in\mathcal V}(\log\hat b_i)^2$（因为 $\nu^{(B)} \equiv 1$），只驱动 $\psi$，不影响 FlowModel
- **objective 层面**：Level 3 的训练目标比 Level 2 多 $\lambda_B \mathcal{L}_B + \lambda_\nu \mathcal{L}_\nu$ 两项 —— 这两项**不**和 Level 2 逐 bit 相同，但通过 `nu.detach()` 路由后，FlowModel 看到的 effective gradient 与 Level 2 一致（FlowModel 只受 $\mathcal{L}_{\text{bridge}}$ 训练）。因此 FlowModel 的训练动力学与 Level 2 完全相同
- 若进一步 $\lambda_B = 0$ 且 $\psi$ 从零 init，Level 3 整体训练动力学完全等同 Level 2

> **总结**：strict bit-for-bit 嵌套是 *corruption + SFM target* 层面的；推断端三级共享 ODE 采样器；objective-level 通过 stop-grad firewall 实现 effective-gradient 等价，而不是逐 bit 相同。

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
   │     δ_x = x_t - [(1-t)·x_0 + t·x_1]        [bridge noise, stored in batch]
   │     ω   = Log(R_0^T · R_1)
   │     Ω_t = t·ω + σ_ω · √(t(1-t)) · ε_ω
   │     δ_Ω = Ω_t - t·ω                          [bridge noise, stored in batch]
   │     R_t = R_0 · Exp(hat(Ω_t))
   │
   ▼
[self-condition, 50% 概率, 复用 Level 1 逻辑]
   │
   ▼
FlowModel(x_t, R_t, t, single_emb, pair_emb) → (x̂_1, R̂_1, pred_torsions, ...)
   │
   ├─ L_trans    = w_x · MSE(x̂_1, x_1 + δ_x/(2t))   [SFM corrected target]
   ├─ L_rots_vf  = w_ω · MSE(v_θ^ω, Log(R_t^T R_1) + δ_Ω/(2t))  [SFM corrected]
   ├─ L_se3_vf   = L_trans + L_rots_vf
   ├─ L_bb_atom  = MSE(backbone atoms, ATOM37)       [gated: t > 0.25]
   ├─ L_dist_mat = MSE(pairwise distances)           [gated: t > 0.25]
   ├─ L_torsion  = MSE(torsion sin/cos)              [gated: t > 0.25]
   │
   └─ train_loss = L_se3_vf + (L_bb_atom + L_dist_mat + L_torsion) · aux_loss_weight
```

**与 Level 1 的差别**：(a) `corrupt_batch` 里多了两项 BB 噪声 + 存储 bridge noise $\delta$；(b) L_trans / L_rots_vf 的 target 包含 SFM correction $\delta/(2t)$。**无需任何 detach 处理**（因为没有 bridge-dependent 辅助网络）。

### 推断阶段（Level 2-3, ODE）

```
x_0 ~ N(0, I),  R_0 ~ U(SO(3))
   │
   for k = 0 … num_timesteps - 1:
       FlowModel(x_k, R_k, t_k) → (x̂_1, R̂_1)
       │
       u_x = (x̂_1 - x_k) / (1 - t_k)
       u_ω = Log(R_k^T · R̂_1) / (1 - t_k)
       │
       x_{k+1} = x_k + Δt · u_x               [deterministic Euler, 无噪声]
       R_{k+1} = R_k · Exp(hat(Δt · u_ω))      [deterministic Euler, 无噪声]
       │
       t_{k+1} = t_k + Δt
   │
   最后一步的 (x̂_1, R̂_1, pred_torsions) 用于生成 ATOM37 构象
```

**与 Level 1 推断形式完全一致**。Level 2-3 的差别仅在训练端（SFM loss + bridge corruption），推断端共享同一 ODE 采样器。

---

## 参数量

| 模块 | 参数量 | 备注 |
|------|--------|------|
| FlowModel（漂移） | 14,248,510 (~14.25M) | IPA 架构，5 blocks，三个 Level 共享 |
| **Level 1 / Level 2 总计** | **14,248,510 (~14.25M)** | Level 2 相对 Level 1 **零额外参数** |
| FlexibilityNet（Level 3 新增） | ~117,249 | single 投影 + pair-biased MHA(h=128, n_heads=4) + 零初始化 scalar 头 |
| **Level 3 总计** | ~14,365,759 | +0.82% over Level 2 |

Level 2 相对 Level 1 **没有新增任何参数**，全部随机性由 BB 噪声引入。Level 3 新增的 FlexibilityNet 只有约 117K 参数（<1%），且在初始化时严格等价于 Level 2（$\nu_i \equiv 1$）——零权重最后一层让 attention block 的随机初始化无法影响输出。

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

1. **Inference 噪声调度**：固定 $\sigma$ vs 按 $t$ 退火 vs 独立 hyperparameter。目前 Levels 2/3 的 inference $\sigma$ 与 training 一致。
2. **Level 2 → Level 3 热启动的 $\lambda_B$ 调度**：一开始 $\lambda_B$ 应该小还是大？是否需要渐进 warm-up？
3. **NMR 结构的 B-factor fallback**：当前做法是按 per-residue + per-sample 有效性门控跳过 $\mathcal{L}_B$，是否可以改用 ensemble-derived B-factor 作为替代？

---

> **历史说明**：旧 Level 2（BridgeWidthNet-based heteroscedastic NLL）和旧 Level 3（NLL + B-factor MSE anchor）设计的完整描述已从此文档移除。如需查阅，参见 git log 中 commit `bee0f44 Level 2 heteroscedastic bridge: BridgeWidthNet + NeurIPS draft` 之前的版本。
