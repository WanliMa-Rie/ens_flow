# FK-Flow: 基于 Feynman-Kac 路径积分的 B-factor 引导 SE(3) Flow Matching

## 3.1 问题设定与符号

**SE(3) Flow Matching 回顾。** 给定 RNA 序列长度 $L$，每个核苷酸残基 $i$ 由一个刚体 frame $(x_i, R_i) \in SE(3)$ 表示，其中 $x_i \in \mathbb{R}^3$ 为 C4' 原子坐标（frame 平移），$R_i \in SO(3)$ 为 frame 旋转矩阵。模型学习一个条件向量场 $v_\theta: SE(3)^L \times [0,1] \to \mathfrak{se}(3)^L$，将先验分布 $p_0$（各向同性高斯平移 + 均匀 SO(3) 旋转）推向目标分布 $p_1$。训练时，对每个样本均匀采样 $t \sim \mathcal{U}[0,1]$，构造插值状态
$$
x_t = (1-t)\,x_0 + t\,x_1, \quad R_t = \text{Exp}\bigl(t \cdot \text{Log}(R_0^\top R_1)\bigr) R_0
$$
并回归向量场 $v_\theta(Z_t, t)$ 到 ground truth。

**B-factor 与热位移。** 实验 B-factor 量化晶体中原子位置的热涨落：
$$
B_j = \frac{8\pi^2}{3} \langle \|\delta r_j\|^2 \rangle
$$
其中 $\delta r_j$ 是原子 $j$ 的位移偏差。高 B-factor 区域（如 loop、末端）对应结构柔性高的部位。我们的核心假设是：**实验 B-factor 可以作为生成过程沿路径累积的总不确定性预算**——柔性区域应在 flow 轨迹中积累更多随机性。

**目标。** 我们引入一个可学习的、per-residue 的随机扩散场，使得 flow matching 从确定性 ODE 升级为 SDE：
$$
dZ_t = v_\theta(Z_t, t)\,dt + \sigma_\psi(Z_t, t)\,dW_t
$$
其中 $\sigma_\psi$ 的空间分布和时间演化受实验 B-factor 约束。


## 3.2 刚体运动学与 TLS 参数化

### 3.2.1 Frame 原子几何

RNA backbone frame 由 C4'、O4'、C3' 三个原子定义。每个原子在局部坐标系中的位置 $a_j$ 是核苷酸几何常数（与核苷酸类型和 flow 状态无关）：

| 原子 | 局部坐标 $a_j$ | 物理意义 |
|------|----------------|----------|
| C4' | $(0, 0, 0)$ | Frame 原点 |
| O4' | $(1.45, 0, 0)$ | x 轴方向 |
| C3' | $(-0.38, 1.48, 0)$ | xy 平面内 |

给定 frame $(x_i, R_i)$，原子 $j$ 的全局坐标由刚体变换给出：
$$
r_j = x_i + R_i \, a_j
$$

### 3.2.2 SE(3) 灵敏度分析

考虑 frame 受到无穷小 Lie algebra 扰动 $\xi = (\delta x, \delta \omega) \in \mathfrak{se}(3) \cong \mathbb{R}^6$，原子 $j$ 的全局位移为：
$$
\delta r_j = \delta x - [a_j]_\times \,\delta\omega = A_j \,\xi
$$
其中 $A_j = [I_3 \mid -[a_j]_\times] \in \mathbb{R}^{3\times 6}$ 为灵敏度矩阵。若 $\xi$ 的协方差为 $\Sigma^{\mathfrak{se}(3)} \in \mathbb{R}^{6\times 6}$，则 per-atom 位移协方差为 $\text{Cov}(\delta r_j) = A_j \,\Sigma^{\mathfrak{se}(3)} A_j^\top$，对应的 B-factor 正比于：
$$
B_j \propto \text{tr}(\text{Cov}(\delta r_j)) = \text{tr}(\Sigma^{\mathfrak{se}(3)} \cdot A_j^\top A_j)
$$
此关系恰好对应晶体学中的 **Translation-Libration-Screw (TLS)** 刚体位移模型 [Schomaker & Trueblood, 1968]，已在数千个晶体结构精修中得到验证。

### 3.2.3 各向异性对角参数化

完整 $6 \times 6$ 协方差有 21 个自由参数，对 3 个 B-factor 目标严重过参数化。我们采用各向同性平移 + 各向异性旋转的 block-diagonal 参数化：
$$
\Sigma^{\mathfrak{se}(3)}_i(t) = \begin{pmatrix} \sigma_{x,i}^2(t) \, I_3 & 0 \\ 0 & \text{diag}(s_{1,i}(t),\, s_{2,i}(t),\, s_{3,i}(t)) \end{pmatrix}
$$
每残基 4 个正参数 $(\sigma_x, s_1, s_2, s_3)$，由 AmplitudeNet $\sigma_\psi$ 预测。代入灵敏度公式，per-atom 瞬时扩散率（trace）为：
$$
q_j(t) = 3\sigma_x^2 + (a_{j2}^2 + a_{j3}^2)\,s_1 + (a_{j1}^2 + a_{j3}^2)\,s_2 + (a_{j1}^2 + a_{j2}^2)\,s_3
$$

代入 §3.2.1 中的固定几何，三个 frame 原子的旋转灵敏度系数为：

| | $s_1$（绕 x） | $s_2$（绕 y） | $s_3$（绕 z） |
|---|---|---|---|
| C4' | 0 | 0 | 0 |
| O4' | 0 | 2.10 | 2.10 |
| C3' | 2.19 | 0.14 | 2.33 |

关键性质：$s_1$（绕 x 轴旋转）**仅影响 C3'、不影响 O4'**——因为 O4' 恰好位于 x 轴上，绕 x 轴的旋转不改变其位置。各向同性模型 $\Sigma_\omega = \sigma_\omega^2 I$ 无法解耦 O4' 与 C3'（$\|a\|^2$ 仅差 10%），而各向异性参数化为每个 frame 原子的 B-factor 提供了独立控制。


## 3.3 Feynman-Kac 累积不确定性

### 3.3.1 核心定义

我们定义 per-atom **剩余不确定性** $U_j(t)$ 为沿 flow 轨迹从时刻 $t$ 到终点 $t{=}1$ 的瞬时扩散率路径积分：
$$
U_j(t) \triangleq \int_t^1 q_j(Z_s, s)\, ds
$$
其中 $q_j$ 是 §3.2.3 中的各向异性 trace。沿确定性 flow 轨迹 $\dot{Z}_t = v_\theta(Z_t, t)$，$U_j$ 满足 ODE：
$$
\frac{dU_j}{dt}\bigg|_{\text{along } v_\theta} = -q_j(Z_t, t), \qquad U_j(1) = 0
$$

此结构与 Feynman-Kac 公式的联系在于：$U_j$ 可视为 SDE 路径上"running cost"的条件期望（value function），$q_j$ 是瞬时 cost。

### 3.3.2 边界条件与 B-factor 对齐

边界条件赋予 $U_j$ 物理含义：

- **终端条件** $U_j(1) = 0$：生成结束时无剩余不确定性。
- **初始条件** $U_j(0) = B_j^+$：全路径累积的总不确定性等于（正规化后的）实验 B-factor。

其中 $B_j^+$ 为保持正性的归一化 B-factor（log-shift-scale），确保与 $U_j \geq 0$（$q_j \geq 0$ 决定了 $U$ 单调非增）的数学约束一致。

### 3.3.3 与 constant-rate 方法的对比

一种朴素做法是假设扩散率恒定，令 $U_j^{\text{rem}}(t) = (1-t) \cdot q_j$。然而由于 $(1-t)$ 对所有原子相同，所得 loss 退化为瞬时匹配 $\|q_j - B_j\|^2$，不含任何路径积分。更本质地，AmplitudeNet 以 $t$ 为输入，$\sigma_\psi(t)$ 随时间变化，constant-rate 假设不成立。Feynman-Kac 框架允许网络自由学习非均匀的时间分配策略（如"前期探索、后期精修"）。


## 3.4 网络架构

### 3.4.1 AmplitudeNet $\sigma_\psi$

AmplitudeNet 预测每个残基在时刻 $t$ 的瞬时 $\mathfrak{se}(3)$ 扩散参数。

**输入：** per-residue 特征向量，由三部分拼接：噪声平移 $x_t \in \mathbb{R}^3$、旋转矩阵上三角 $\text{utri}(R_t) \in \mathbb{R}^6$、结构 embedding $c_i \in \mathbb{R}^{384}$，共 393 维。时间步 $t$ 通过正弦编码嵌入为 $\mathbb{R}^{64}$。

**架构：** 输入经线性层投影至 $\mathbb{R}^{128}$，经 4 层残差卷积块（kernel=5，FiLM 时间调制）提取局部序列上下文，最后通过两个并行输出头：
- 平移头：$\text{Linear}(128 \to 1) + \text{softplus} \to \sigma_x \in \mathbb{R}_+$
- 旋转头：$\text{Linear}(128 \to 3) + \text{softplus} \to (s_1, s_2, s_3) \in \mathbb{R}_+^3$

### 3.4.2 UncertaintyNet $U_\phi$

UncertaintyNet 预测每个 frame 原子从当前时刻到终点的剩余累积不确定性 $U_j(t)$。

**架构：** 与 AmplitudeNet 相同的 backbone（FiLM + ResConvBlocks），输出头为 $\text{Linear}(128 \to 3)$，对应 C4'、O4'、C3' 三个 frame 原子各一个标量。无 softplus 约束（$U_j$ 在 $t=1$ 时为零，通过损失函数隐式约束非负性）。

### 3.4.3 协同机制

两个网络角色互补：
- **AmplitudeNet** 决定 *如何在时间轴上分配* 不确定性预算（"花钱"的速率）
- **UncertaintyNet** 追踪 *还剩多少* 预算（"余额"）

训练通过时间一致性损失将两者耦合：AmplitudeNet 的瞬时输出 $q_j$ 必须在积分意义上与 UncertaintyNet 的 $U_j$ 保持一致。


## 3.5 训练

### 3.5.1 Feynman-Kac 损失

每步训练，在已有的 flow matching 插值时刻 $t_1$（来自 `corrupt_batch`）之外，额外采样 $t_2 \sim \mathcal{U}(t_1, 1)$，并通过线性/测地线插值构造 $Z_{t_2}$。三项损失：

**TD 时间一致性（semi-gradient）：**
$$
\mathcal{L}_{\text{TD}} = \frac{1}{|\mathcal{M}|} \sum_{i,j \in \mathcal{M}} \Big\| U_\phi(Z_{t_1}, t_1)_j - \underbrace{\big[U_\phi(Z_{t_2}, t_2)_j + \bar{q}_j \cdot (t_2 - t_1)\big]}_{\text{detached target}} \Big\|^2
$$
其中 $\bar{q}_j = \frac{1}{2}(q_j(Z_{t_1}, t_1) + q_j(Z_{t_2}, t_2))$ 为梯形积分近似，$\mathcal{M}$ 为有效 frame 原子集合。遵循强化学习中的 semi-gradient TD(0) 范式，**target 项（包括 $U_\phi(Z_{t_2}, t_2)$ 和 $\bar{q}_j$）均从计算图中 detach**，仅通过 $U_\phi(Z_{t_1}, t_1)$ 反传梯度，以稳定训练。

**终端条件：**
$$
\mathcal{L}_{\text{term}} = \frac{1}{|\mathcal{M}|} \sum_{i,j \in \mathcal{M}} \big\| U_\phi(x_1, R_1, t{=}1)_j \big\|^2
$$

**初始条件（B-factor 对齐）：**
$$
\mathcal{L}_{\text{init}} = \frac{1}{|\mathcal{M}|} \sum_{i,j \in \mathcal{M}} \big\| U_\phi(Z_0, t{=}0)_j - B_j^+ \big\|^2
$$
其中 $Z_0$ 从实际先验分布采样（centered Gaussian $\times$ 10Å，均匀 SO(3)），而非重用插值状态。

**联合损失：**
$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_{\text{TD}} \mathcal{L}_{\text{TD}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}} + \lambda_{\text{init}} \mathcal{L}_{\text{init}}
$$

### 3.5.2 梯度流分析

各损失项对各网络的梯度路径：

| 损失 | $v_\theta$（drift） | $\sigma_\psi$（AmplitudeNet） | $U_\phi$（UncertaintyNet） |
|------|-----|------|------|
| $\mathcal{L}_{\text{flow}}$ | ✓ | | |
| $\mathcal{L}_{\text{TD}}$ | | ✗（detached） | ✓（仅 $U_{t_1}$） |
| $\mathcal{L}_{\text{term}}$ | | | ✓ |
| $\mathcal{L}_{\text{init}}$ | | | ✓ |

AmplitudeNet 不直接接收 FK 损失的梯度。其约束是**间接的**：若 $\sigma_\psi$ 分配的噪声预算 $\int_0^1 q_j \, dt$ 不匹配 $B_j^+$，则 $U_\phi$ 无法同时满足 $U(0) = B_j^+$、$U(1) = 0$ 和 TD 一致性三个条件，训练信号通过 TD target 中 $\bar{q}_j$ 的数值（而非梯度）间接影响 AmplitudeNet 的参数更新方向。这一机制类似于 actor-critic 框架中 critic 对 actor 的隐式约束。

### 3.5.3 优化设置

三组参数使用不同学习率：drift 网络 $v_\theta$（主学习率 $\times 0.1$），AmplitudeNet 和 UncertaintyNet（$3 \times 10^{-4}$），采用 AdamW 优化器。降低 drift 学习率是因为 FK 监督不应破坏已收敛的 flow matching 质量。


## 3.6 推理：各向异性 Euler-Maruyama 采样

推理时，给定先验样本 $Z_0 \sim p_0$，使用 Euler-Maruyama 格式积分 SDE：

**平移更新（各向同性）：**
$$
x_{k+1} = x_k + \Delta t \cdot v_\theta^x(Z_{t_k}, t_k) + \sqrt{\Delta t} \cdot \sigma_{x}(Z_{t_k}, t_k) \cdot \xi_k, \quad \xi_k \sim \mathcal{N}(0, I_3)
$$

**旋转更新（各向异性，local frame）：**
$$
R_{k+1} = R_k \cdot \text{Exp}\Big(\Delta t \cdot \omega_\theta(Z_{t_k}, t_k) + \sqrt{\Delta t} \cdot L_\omega(Z_{t_k}, t_k) \cdot \zeta_k\Big), \quad \zeta_k \sim \mathcal{N}(0, I_3)
$$
其中
$$
L_\omega = \text{diag}\big(\sqrt{s_1},\; \sqrt{s_2},\; \sqrt{s_3}\big)
$$
是旋转噪声的 Cholesky 因子。$L_\omega$ 在 local frame 中定义，因此不同旋转轴的噪声量级独立——例如，若结构在某个 hinge 方向有较大柔性，模型可以学到沿该方向注入更多旋转噪声。

多次采样同一序列会产生结构 ensemble，其 per-atom 位置方差应与实验 B-factor profile 相关。
