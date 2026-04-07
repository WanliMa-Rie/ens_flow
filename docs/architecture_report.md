# 架构报告：TD-Flow — 基于 Feynman-Kac 框架的 B-factor 监督 SE(3) 随机桥

## 概述

本报告描述了 TD-Flow 模型的实现细节。该模型在 RNA 骨架生成的 SE(3) 流匹配框架之上，引入了随机扩散模块，利用实验 B-factor 监督每个核苷酸在生成轨迹中的柔性预算。

核心设计思想：将 B-factor 视为生成过程中从噪声到结构的**路径积分总不确定度**，而非单步快照。通过 TD（时序差分）学习风格的训练，避免了昂贵的完整轨迹展开。

---

## 设计原则

1. **各向同性随机桥** — B-factor 本身是各向同性量（单标量），无法区分平动/转动方向，因此采用各向同性参数化 (σ_x, σ_ω)，由模型自主分配 R³/SO(3) 柔性预算
2. **路径积分累积** — B-factor 对应完整生成轨迹的累计不确定度，而非某一时刻的瞬时扰动
3. **TD 风格训练** — 通过时序一致性约束训练价值函数，无需完整展开采样轨迹
4. **简洁优先** — 相比 TLS 各向异性分解，当前设计更简洁有效，避免了过度启发式的晶体学分解

---

## 关键公式

**瞬时柔性率（instantaneous flexibility rate）：**

$$q_i(t) = \sigma_{x,i}^2(t) + \sigma_{\omega,i}^2(t)$$

模型自主决定 σ_x（平动）与 σ_ω（转动）的比例，总和对应 B-factor 所表达的整体柔性。

**Feynman-Kac 价值函数：**

$$U_i(t) = \int_t^1 q_i(s)\, ds$$

表示从时刻 t 到终点仍剩余的柔性预算。初始条件：$U_i(0) = B_i^{\text{norm}}$；终止条件：$U_i(1) = 0$。

---

## 修改的文件

### `rna_backbone_design/models/amplitude_net.py`

**AmplitudeNet** — 每个核苷酸预测两个各向同性扩散幅度：

- 输入：`trans_t [B,N,3]`、`rotmats_t [B,N,3,3]`、`single_emb [B,N,384]`、`t [B,1]`
- 输出：`sigma_x [B,N]`（平动幅度）、`sigma_omega [B,N]`（转动幅度），均为 softplus 正值
- B-factor 仅作为监督目标，不作为输入

**UncertaintyNet** — 预测每个核苷酸的剩余累计不确定度 $U_i(t)$：

- 与 AmplitudeNet 相同骨干网络（FiLM + ResConvBlock）
- 输出：`U [B,N]`（每核苷酸 1 个标量，无正值约束）

**共享骨干架构：**

```
输入特征（trans=3 + rotmat上三角=6 + single_emb=384）
    → Linear 投影 → hidden_dim=128
    → FiLM 时序调制（正弦嵌入 → MLP → scale/shift）
    → 4× ResConvBlock（LayerNorm → Conv1D(k=5) → FiLM → SiLU → Conv1D）
    → LayerNorm → Linear → 输出
```

### `rna_backbone_design/data/fk_utils.py`

```python
def compute_flexibility_rate(sigma_x, sigma_omega):
    """q_i = σ_x² + σ_ω²，[B,N]"""
    return sigma_x ** 2 + sigma_omega ** 2

def normalize_b_factors_positive(b_factors, mask, eps=1e-6):
    """正值归一化：log → 平移至 min=0 → 缩放至 mean=1
    必须保持正值，因为 U(t) ≥ 0 由 q ≥ 0 保证。"""
```

**注意**：z-score 归一化在此处数学上不一致——因为 $q \geq 0$ 意味着 $U$ 单调不增，所以 $U(0) \geq 0$，不能有负的目标值。

### `rna_backbone_design/models/flow_module.py`

**`_feynman_kac_step()`** — 每步训练计算三项 FK 损失：

| 损失项 | 公式 | 含义 |
|--------|------|------|
| $L_{\text{TD}}$ | $\|U(t_1) - \text{sg}[U(t_2) + \bar{q} \cdot \Delta t]\|^2$ | 时序一致性（半梯度 TD） |
| $L_{\text{term}}$ | $\|U(x_1, t{=}1)\|^2$ | 终止条件：生成结束不确定度为零 |
| $L_{\text{init}}$ | $\|U(x_0, t{\approx}0) - B^{\text{norm}}\|^2$ | 初始条件：起点不确定度等于归一化 B-factor |

**时间采样策略：**
- $t_1$ 复用 `corrupt_batch` 已采样的插值时刻（省一次前向计算）
- $t_2$ 在 $(t_1, 1)$ 均匀采样
- $t_2$ 时刻状态通过测地线插值从 $(x_{t_1}, R_{t_1})$ 向 $(x_1, R_1)$ 推进

**半梯度 TD（训练稳定性关键）：**
```python
td_target = U_t2.detach() + q_bar.detach() * dt
L_TD = ||U_t1 - td_target||²
```
梯度仅通过 $U(t_1)$ 回传至 UncertaintyNet；AmplitudeNet 通过 $L_{\text{init}}$ 和 $L_{\text{term}}$ 间接受约束（若分配的总预算与 B-factor 不符，UncertaintyNet 无法同时满足所有三项条件）。

**`configure_optimizers()`** — 三个参数组：

| 参数组 | 学习率 |
|--------|--------|
| 漂移模型（FlowModel） | `1e-4 × 0.1 = 1e-5` |
| AmplitudeNet | `3e-4` |
| UncertaintyNet | `3e-4` |

### `rna_backbone_design/data/interpolant.py`

- `sample_stochastic` / `rollout_terminal`：解包 AmplitudeNet 的两个输出 `(sigma_x, sigma_omega)`
- `_rots_euler_maruyama_step`：各向同性标量噪声注入：
  ```python
  total_rotvec = scaling * d_t * rot_vf + a_rots[..., None] * noise * d_t.sqrt()
  ```

### `configs/config.yaml`

```yaml
fk_flow:
  enabled: true
  amplitude_net:
    c_single_in: 384       # 与主模型 single_emb 维度一致
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
  uncertainty_net:         # 与 amplitude_net 相同架构
    c_single_in: 384
    timestep_embed_dim: 64
    hidden_dim: 128
    num_layers: 4
    kernel_size: 5
  loss:
    td_weight: 1.0         # L_TD 权重
    term_weight: 1.0       # L_term 权重
    init_weight: 1.0       # L_init 权重
  optimizer:
    lr: 0.0003
  drift_lr_scale: 0.1      # 漂移模型使用缩减学习率
```

---

## 数据流

### 训练阶段

```
batch (x_0, x_1, B-factors)
         │
    ┌────┴──────────────┐
    ▼                   ▼
corrupt_batch        _feynman_kac_step
(得到 x_t, t)              │
    │              ┌────┴─────────────┐
    ▼              ▼                  ▼
 model_step    AmplitudeNet       UncertaintyNet
(流匹配损失)  (σ_x, σ_ω)@t1,t2    U@t1,t2,t≈0,t=1
                   │                  │
                   ▼                  ▼
             q = σ_x²+σ_ω²   L_TD + L_term + L_init
                   │                  │
         ┌─────────┴──────────────────┘
         ▼
    total_loss = L_flow + L_FK
```

### 推断阶段

```
x_0 ~ 先验（中心高斯平动 + 均匀 SO(3) 转动）
    │
    └─► Euler-Maruyama 循环（60步）
              │
         每步 k：
         ├─ FlowModel(x_t, t) → v_θ（漂移）
         ├─ AmplitudeNet(x_t, t) → (σ_x, σ_ω)
         ├─ x_{k+1} = x_k + Δt·v_θ + √Δt·σ_x·ξ_x
         └─ R_{k+1} = R_k · exp(Δt·ω + √Δt·σ_ω·ξ_ω)
              │
         x_1（生成结构）
```

---

## 参数量估算

| 模块 | 参数量 | 备注 |
|------|--------|------|
| FlowModel（漂移） | ~5M | 现有 IPA 架构 |
| AmplitudeNet | ~270K | 4层 ResConvBlock，hidden=128 |
| UncertaintyNet | ~270K | 与 AmplitudeNet 相同结构 |
| **新增合计** | **~540K** | 约为漂移模型的 10% |

---

## 训练开销

每步训练需要：
- 2次 AmplitudeNet 前向（t1 和 t2 时刻）
- 4次 UncertaintyNet 前向（t1、t2、t≈0、t=1）
- 与之前 L_cov（需要 M 条完整轨迹）相比，开销大幅降低（约 2× 基础流匹配开销）

---

## 关键设计决策

**1. 各向同性而非 TLS 各向异性分解**

B-factor 是各向同性量（3 个帧原子，4 个 TLS 未知量，欠定）。非刚体运动（扭转角、晶体堆积）也会污染 B-factor 信号，使方向信息不可靠。因此放弃 TLS 分解，改为由模型自主学习 R³/SO(3) 柔性分配，保持设计简洁。

**2. 以 L_init 为主要监督入口**

AmplitudeNet 本身不直接接受 B-factor 监督。其柔性分配的合理性来自约束：UncertaintyNet 必须同时满足 $U(0)=B$、$U(1)=0$ 和时序一致性。若 AmplitudeNet 给出的 q 总量与 B-factor 不一致，UncertaintyNet 的三项条件无法同时成立。

**3. L_init 的先验采样**

初始条件在 $t \approx 0$ 处评估，状态为纯噪声：中心化高斯平动（尺度 10Å）+ 均匀 SO(3) 转动，与 `corrupt_batch` 使用的先验完全一致。

**4. 复用插值时刻**

$t_1$ 直接复用 `corrupt_batch` 计算的 $x_{t_1}$，避免额外的插值开销。

**5. B-factor 正值归一化的必要性**

由 $q \geq 0$ 可知 $U(t)$ 单调不增。结合 $U(1)=0$，必须保证 $U(0) \geq 0$，即 B-factor 目标值必须非负。z-score 归一化（均值为 0，存在负值）在此框架下数学上不一致，必须使用正值保持的归一化方法（log → 平移 → 缩放）。
