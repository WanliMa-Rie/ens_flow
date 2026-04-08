# 基于 Feynman-Kac 累积不确定性的 B-factor 监督方法

## 1. 问题描述

RNA 结构的实验 B-factor 反映了每个原子在晶体中的位移方差：

$$
B_j = \frac{8\pi^2}{3} \langle u_j^2 \rangle
$$

我们的目标是：在 $SE(3)$ flow matching 生成过程中，让沿生成路径**累积的不确定性**对齐实验 B-factor。具体而言，柔性区域（高 B-factor）应在生成过程中积累更多随机性，从而在多次采样时产生更大的结构多样性。

核心问题在于：如何定义一个数学上严格、可训练的"累积不确定性"量，并将其与 B-factor 关联？

---

## 2. Constant-rate 方法的退化性

一种直观的做法是：在训练时刻 $t$，用 $(1-t)$ 缩放瞬时不确定性 trace 来近似剩余累积量，得到 tail loss：

$$
\mathcal{L}_{\text{tail}} = \sum_j \left| (1-t) \cdot q_j - (1-t) \cdot B_j \right|^2 = (1-t)^2 \sum_j \left| q_j - B_j \right|^2
$$

$(1-t)^2$ 对所有原子一致，直接消掉。这意味着该 loss 退化为**瞬时 trace 匹配**，没有任何真正的路径积分。

更根本的矛盾在于：如果扩散系数 $\sigma(t)$ 随时间变化（AmplitudeNet 以 $t$ 为输入），constant-rate 假设 $U_j^{\text{rem}}(t) = (1-t) \cdot U_j^{\text{tot}}$ 就不再成立。

---

## 3. $\mathfrak{se}(3)$ 协方差与 TLS 模型

### 3.1 Frame 几何

RNA backbone frame 由 C4', O4', C3' 三个原子定义，其在 local frame 中的坐标对所有核苷酸类型是固定的：

| 原子 | 局部坐标 $a_j$ | $\|a_j\|^2$ |
|------|----------------|--------------|
| C4'（原点） | $(0, 0, 0)$ | 0 |
| O4'（x 轴） | $(1.45, 0, 0)$ | 2.10 |
| C3'（xy 平面） | $(-0.38, 1.48, 0)$ | 2.33 |

几何特征：
- C4' 在 frame 原点 → 对任何旋转扰动的贡献为零
- O4' 在 x 轴上 → 对绕 x 轴的旋转免疫
- C3' 在 xy 平面内 → 旋转灵敏度与 O4' 完全不同
- 三个原子共面（$a_{j,3} = 0$）

### 3.2 各向同性模型的局限

若旋转扰动各向同性 $\Sigma_\omega = \sigma_\omega^2 I$，则 per-atom trace 为 $q_j = 3\sigma_x^2 + 2\|a_j\|^2 \sigma_\omega^2$。对三个 frame 原子：

$$
q_{C4'} = 3\sigma_x^2, \quad q_{O4'} = 3\sigma_x^2 + 4.20\,\sigma_\omega^2, \quad q_{C3'} = 3\sigma_x^2 + 4.67\,\sigma_\omega^2
$$

O4' 和 C3' 的 lever arm 极为接近（$\|a\|^2 = 2.10$ vs $2.33$），导致比值 $q_{O4'}/q_{C3'}$ 被锁死在 0.9–1.0 之间。各向同性模型**无法独立调节** O4' 和 C3' 的 B-factor。

### 3.3 从 $SE(3)$ 刚体运动到 per-atom 位移协方差

本节从刚体运动学出发，逐步推导每个原子的位移如何由 $SE(3)$ 上的扰动决定，最终得到 per-atom B-factor 与 Lie algebra 协方差之间的精确关系。

#### Step 1：全局坐标的刚体重建

每个残基 $i$ 的 backbone frame 是一个刚体变换 $(x_i, R_i) \in SE(3)$，其中 $x_i \in \mathbb{R}^3$ 是 C4' 的全局坐标（frame 平移），$R_i \in SO(3)$ 是 frame 旋转。

Frame 内的原子 $j$（C4', O4', C3'）在**局部坐标系**中有固定的位置向量 $a_j \in \mathbb{R}^3$（见 §3.1 表格，这些是糖环的理想化几何常数，不随 flow 状态变化）。原子 $j$ 的全局坐标通过刚体变换得到：

$$
r_j = x_i + R_i \, a_j
$$

对于 C4'，$a_{C4'} = 0$，所以 $r_{C4'} = x_i$，即 C4' 的全局坐标就是 frame 的平移。O4' 和 C3' 的全局坐标则额外依赖旋转 $R_i$。

#### Step 2：$SE(3)$ 上的无穷小扰动

考虑 frame $(x_i, R_i)$ 受到一个无穷小扰动。在 Lie algebra $\mathfrak{se}(3) \cong \mathbb{R}^6$ 中，扰动可分解为：

- 平移扰动 $\delta x \in \mathbb{R}^3$：frame 整体平移
- 旋转扰动 $\delta\omega \in \mathbb{R}^3$（角速度向量）：frame 绕原点旋转

扰动后，frame 变为 $(x_i + \delta x, \; R_i \, \exp([\delta\omega]_\times))$。将 $r_j = x_i + R_i \, a_j$ 对扰动求导，原子 $j$ 的全局位移为：

$$
\delta r_j = \delta x + \delta\omega \times (R_i \, a_j)
$$

其中 $\delta\omega \times v = -[v]_\times \,\delta\omega$（叉积的反对称矩阵形式）。由于我们在 **body frame** 中分析（即 $R_i = I$ 时的局部扰动），此式简化为：

$$
\delta r_j = \delta x + \delta\omega \times a_j = \delta x - [a_j]_\times \,\delta\omega
$$

这说明：**平移扰动对所有原子产生相同位移；旋转扰动通过叉积 $\delta\omega \times a_j$ 产生依赖于原子位置的位移。**

#### Step 3：灵敏度矩阵

将上式写成矩阵形式。定义 6 维 Lie algebra 扰动向量 $\xi = (\delta x, \delta\omega) \in \mathbb{R}^6$，则：

$$
\delta r_j = \underbrace{\bigl[\,I_3 \;\big|\; -[a_j]_\times\,\bigr]}_{A_j \in \mathbb{R}^{3 \times 6}} \,\xi
$$

$A_j$ 是原子 $j$ 的**灵敏度矩阵**，将 $\mathfrak{se}(3)$ 扰动线性映射为原子的全局位移。对于在原点的 C4'（$a_{C4'} = 0$），$A_{C4'} = [I_3 \mid 0]$，即只感受平移。

#### Step 4：从 Lie algebra 协方差到 per-atom 位移协方差

假设 $\mathfrak{se}(3)$ 扰动 $\xi$ 服从零均值分布，协方差为 $\Sigma^{\mathfrak{se}(3)} \in \mathbb{R}^{6\times 6}$（半正定）。由于 $\delta r_j = A_j \,\xi$ 是线性变换，per-atom 位移协方差为：

$$
\text{Cov}(\delta r_j) = A_j \,\Sigma^{\mathfrak{se}(3)}\, A_j^\top \in \mathbb{R}^{3 \times 3}
$$

#### Step 5：与 B-factor 的关系

各向同性 B-factor 正比于位移协方差的迹（即均方位移的 3 个方向之和）：

$$
B_j \propto \text{tr}\bigl(\text{Cov}(\delta r_j)\bigr) = \text{tr}\bigl(A_j \,\Sigma^{\mathfrak{se}(3)}\, A_j^\top\bigr) = \text{tr}\bigl(\Sigma^{\mathfrak{se}(3)} \cdot A_j^\top A_j\bigr)
$$

最后一步利用了迹的循环不变性。至此，**per-atom B-factor 是 $\Sigma^{\mathfrak{se}(3)}$ 的线性泛函**——形式简洁、可微分、且在 $SE(3)$ 几何上精确。

### 3.4 与晶体学 TLS 模型的对应

上述公式恰好就是晶体学中的 **Translation-Libration-Screw (TLS)** 模型：

$$
\Sigma^{\mathfrak{se}(3)} = \begin{pmatrix} T & S \\ S^\top & L \end{pmatrix}
$$

其中 $T$（平移）、$L$（旋振）、$S$（螺旋耦合）是标准 TLS 张量。TLS 模型已在数千个晶体结构上得到验证，是刚体位移不确定性的标准物理模型。

### 3.5 参数化选择：各向同性平移 + 各向异性旋转

完整的 $6\times 6$ 协方差有 21 个自由参数，对 3 个 B-factor 目标过于冗余。我们采用 block-diagonal 参数化：

$$
\Sigma^{\mathfrak{se}(3)} = \begin{pmatrix} \sigma_x^2 I_3 & 0 \\ 0 & \text{diag}(s_1, s_2, s_3) \end{pmatrix}
$$

各向同性平移（1 参数）+ 各向异性旋转（3 参数）= 每残基 4 个参数。per-atom trace 为：

$$
q_j = 3\sigma_x^2 + (a_{j2}^2 + a_{j3}^2)\,s_1 + (a_{j1}^2 + a_{j3}^2)\,s_2 + (a_{j1}^2 + a_{j2}^2)\,s_3
$$

三个 frame 原子对各旋转轴的灵敏度：

| 原子 | $s_1$ 系数（绕 x 旋转） | $s_2$ 系数（绕 y 旋转） | $s_3$ 系数（绕 z 旋转） |
|------|------------------------|------------------------|------------------------|
| C4' | 0 | 0 | 0 |
| O4' | 0 | 2.10 | 2.10 |
| C3' | 2.19 | 0.14 | 2.33 |

$s_1$（绕 x 轴旋转）**仅影响 C3'，不影响 O4'**（因为 O4' 在 x 轴上）。这彻底解决了各向同性模型的退化问题，赋予模型对每个 frame 原子 B-factor 的独立控制能力。

---

## 4. Feynman-Kac 累积不确定性框架

### 4.1 核心思想

定义 per-atom **剩余不确定性**为沿生成轨迹的路径积分：

$$
U_j(t) = \int_t^1 q_j(x_s, s) \, ds
$$

其中 $q_j$ 使用 §3.5 中的各向异性 $\mathfrak{se}(3)$ trace 公式，$\Sigma_i^{\mathfrak{se}(3)}(t)$ 由 AmplitudeNet 在时刻 $t$ 预测。

### 4.2 ODE 结构

沿确定性 flow 轨迹 $\dot{x}_t = v_\theta(x_t, t)$，剩余不确定性满足：

$$
\frac{dU_j}{dt}\bigg|_{\text{along trajectory}} = -q_j(t)
$$

边界条件：

$$
U_j(1) = 0 \quad (\text{终点无剩余不确定性}), \qquad U_j(0) = B_j^{\text{norm}} \quad (\text{总累积} = \text{实验 B-factor})
$$

物理含义：不确定性沿生成路径**单调递减**，递减速率由瞬时扩散振幅决定。实验 B-factor 是全路径的**总预算**。

### 4.3 网络架构

| 网络 | 输入 | 输出 | 角色 |
|------|------|------|------|
| **AmplitudeNet** $\sigma_\psi$ | $(x_t, R_t, c, t)$ | $(\sigma_x, s_1, s_2, s_3)_i$ per residue | 瞬时 $\mathfrak{se}(3)$ 扩散率 |
| **UncertaintyNet** $U_\phi$ | $(x_t, R_t, c, t)$ | $U_j$ per frame atom | 剩余累积不确定性 |

两个网络协同工作：AmplitudeNet 决定如何在时间轴上**分配**不确定性预算；UncertaintyNet 跟踪**还剩多少**预算。

### 4.4 训练损失

每一步训练采样两个时刻 $t_1 < t_2$，构造对应的插值状态：

$$
x_{t_k} = (1-t_k)\,x_0 + t_k\,x_1, \qquad k = 1, 2
$$

**损失 1 — 时间一致性（TD 风格）：**

$$
\mathcal{L}_{\text{TD}} = \sum_j \left\| U_\phi(x_{t_1}, t_1)_j - U_\phi(x_{t_2}, t_2)_j - \bar{q}_j \cdot (t_2 - t_1) \right\|^2
$$

其中 $\bar{q}_j = \frac{1}{2}\bigl(q_j(x_{t_1}, t_1) + q_j(x_{t_2}, t_2)\bigr)$ 为梯形近似。该项确保：$t_1$ 到 $t_2$ 之间剩余不确定性的减少量等于该区间内瞬时扩散率的积分。

**损失 2 — 终端条件：**

$$
\mathcal{L}_{\text{term}} = \sum_j \left\| U_\phi(x_1, 1)_j \right\|^2
$$

**损失 3 — 初始条件（B-factor 监督）：**

$$
\mathcal{L}_{\text{init}} = \sum_j \left\| U_\phi(x_0, 0)_j - B_j^{\text{norm}} \right\|^2
$$

**总损失：**

$$
\mathcal{L}_{\text{FK}} = \mathcal{L}_{\text{TD}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}} + \lambda_{\text{init}} \mathcal{L}_{\text{init}}
$$

### 4.5 与主 flow matching loss 的联合训练

总训练目标为：

$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{FK}} \mathcal{L}_{\text{FK}}
$$

其中 $\mathcal{L}_{\text{flow}}$ 是原有的 SE(3) velocity field loss + auxiliary losses。两组损失共享 backbone 表征但分别更新各自的网络参数。

### 4.6 采样过程

推理时使用 Euler-Maruyama 积分，噪声在 local frame 中各向异性注入：

$$
x_{k+1} = x_k + \Delta t \cdot v_\theta + \sqrt{\Delta t} \cdot \sigma_x \cdot \xi_x, \qquad \xi_x \sim \mathcal{N}(0, I_3)
$$

$$
R_{k+1} = R_k \,\exp\!\bigl(\Delta t \cdot \omega_\theta + \sqrt{\Delta t} \cdot L_\omega \cdot \xi_\omega\bigr), \qquad \xi_\omega \sim \mathcal{N}(0, I_3)
$$

其中 $L_\omega = \text{diag}(\sqrt{s_1}, \sqrt{s_2}, \sqrt{s_3})$。各向异性的 $L_\omega$ 确保不同旋转轴上的噪声量级不同——例如绕 hinge 轴的旋转可以有更大的扰动。

---

## 5. 方法特性

**无 constant-rate 假设**：$\sigma_\psi(t)$ 可以随时间任意变化。模型可以学到"前期大幅探索、后期精细调整"等非均匀策略。

**真正的路径积分**：$U_\phi$ 学到的是从当前状态到终点的积分值，而非某一时刻的快照。

**B-factor 作为总预算约束**：实验 B-factor 约束的是全路径的累积不确定性总量，而非每一步都逐点匹配。

**$SE(3)$ 几何精确**：通过 $\mathfrak{se}(3)$ 协方差和灵敏度矩阵 $A_j$，per-atom 不确定性的计算严格遵循刚体运动学。各向异性旋转使得模型能区分 lever arm 长度相近但方向不同的原子。

**与 TLS 模型一致**：参数化方案对应晶体学中经过广泛验证的 TLS 刚体位移模型，具有物理合理性。

**计算开销可控**：每步训练约 2 倍开销（两个时间点、两次网络前向传播），远低于需要 $M$ 次完整 rollout 的终端方差匹配方法。

---

## 6. 实现步骤

1. 修改 AmplitudeNet：输出 $(\sigma_x, s_1, s_2, s_3)$ per residue（4 个标量，softplus 保证正性）
2. 实现 UncertaintyNet（与 AmplitudeNet 相似架构，输出 3 个标量/残基 = 每个 frame 原子一个）
3. 实现各向异性 $q_j$ 公式，使用固定的 local coordinates $a_j$
4. 修改 `training_step`：每步采样 $(t_1, t_2)$，计算 $\mathcal{L}_{\text{TD}} + \mathcal{L}_{\text{term}} + \mathcal{L}_{\text{init}}$
5. 更新 Euler-Maruyama 采样：使用各向异性旋转噪声 $L_\omega = \text{diag}(\sqrt{s_1}, \sqrt{s_2}, \sqrt{s_3})$
6. 验证：检查 $U_\phi(x_0, 0)$ 的空间分布是否匹配实验 B-factor
