# Ensemble-Supervised Variational Latent Flow Matching

## 1. 问题定义

同一条 RNA sequence 可能对应多个真实 conformers。如果直接训练一个 deterministic flow matching 模型

$$
v(x_t, t, c),
$$

其中 $c$ 表示 sequence condition，那么在多模态目标下，模型容易学到不同构象 mode 的平均速度场。结果通常不是落在某个真实 mode 上，而是落在几个 mode 之间。

这里的核心改造是引入一个 latent variable $z$，把速度场改写为

$$
v(x_t, t, c, z).
$$

$z$ 不是最终结构，而是一个 mode-level 条件变量。它告诉 decoder 当前应该沿哪一种 transport mode 前进。于是模型学习的不再是单一速度场，而是一族 latent-conditioned velocity fields。

## 2. 核心思路

训练时，我们有同一条 sequence 的多个 conformers。它们的作用不是直接作为多个预测目标并列回归，而是用来推断一个 posterior latent：

$$
q_\phi(z \mid \mathcal{X}_t, c, t).
$$

这里：

- $c$ 是 sequence condition。
- $\mathcal{X} = \{x_1^{(i)}\}_{i=1}^K$ 是同一 sequence 的 conformer set。
- $x_1^{(i)}$ 是第 $i$ 个真实构象。
- $x_0$ 是 flow 的 base state。
- $x_t$ 是中间插值状态。

推断时没有 ensemble，因此必须另外学习一个 sequence-only prior：

$$
p_\theta(z \mid c).
$$

训练目标由两部分组成：

1. 用 posterior latent $z$ 条件化 flow decoder，回归 velocity。
2. 用 KL 项把 posterior 压到 prior 上，使测试时的 prior sampling 可用。

## 3. 数据组织

训练样本不应再被理解为单个 conformer，而应被理解为一个 sequence-centered ensemble item：

$$
\mathcal{X} = \{x_1^{(i)}\}_{i=1}^K.
$$

一个样本对应：

- 一条 sequence $c$
- 一个 conformer set $\mathcal{X}$

为了让 posterior 真正从构象差异里提取 mode 信息，而不是被不同噪声扰乱，训练时对同一个 ensemble item 应共享同一个 $x_0$ 和同一个 $t$。然后对每个 conformer 构造：

$$
x_t^{(i)} = (1 - t)x_0 + t x_1^{(i)},
$$

$$
u_t^{(i)} = x_1^{(i)} - x_0.
$$

这样做的意义是：

- 所有 conformers 都在同一个 interpolation stage 上比较
- 所有 conformers 都相对于同一个 base reference
- posterior 看到的差异主要来自真实构象差异，而不是采样噪声差异

## 4. 训练时的主前向过程

给定一个 sequence-centered ensemble item，训练时的前向过程为：

1. 采样 $t \sim \mathrm{Uniform}(0,1)$
2. 采样 base state $x_0 \sim p_0$
3. 对每个 conformer 构造 $x_t^{(i)}$ 和目标 velocity $u_t^{(i)}$
4. 用整个 noisy conformer set 推断 posterior latent $z$
5. 用某一个具体 conformer 的 $(x_t^{(j)}, u_t^{(j)})$ 做 flow matching 回归

这里形成两个分支：

- posterior inference branch
- velocity regression branch

## 5. Posterior 分支

posterior 分支的输入是整个 noisy conformer set：

$$
\mathcal{X}_t = \{x_t^{(i)}\}_{i=1}^K,
$$

以及 sequence condition $c$ 和时间 $t$。它的目标是输出：

$$
q_\phi(z \mid \mathcal{X}_t, c, t).
$$

我建议把这条分支拆成三步。

### 5.1 Sequence Encoder

先对 sequence 做编码，得到 residue-level token：

$$
s_r \in \mathbb{R}^d, \quad r = 1, \dots, L.
$$

时间 $t$ 应显式注入这一分支。最简单的方式是把 time embedding 加到 sequence token 上，得到：

$$
s_{r,t}.
$$

### 5.2 Shared Structure Encoder

对于每个 conformer 的 noisy state $x_t^{(i)}$，用同一个 structure encoder 编码成 residue token：

$$
h_{r,t}^{(i)} \in \mathbb{R}^d.
$$

关键要求只有一个：所有 conformers 共用同一个 encoder，输出维度一致。

### 5.3 Conformer-Level Cross-Attention Pooling

这一步的作用不是直接预测 dynamics label，而是做 sequence-conditioned residue-wise set pooling。

对于每个 residue $r$，用 sequence token 作为 query，从这一 residue 在不同 conformers 上的 token 集合中读出一个 evidence embedding：

$$
q_{r,t} = W_Q s_{r,t},
$$

$$
k_{r,t}^{(i)} = W_K h_{r,t}^{(i)},
$$

$$
v_{r,t}^{(i)} = W_V h_{r,t}^{(i)}.
$$

注意力权重为

$$
\alpha_{i,r,t} =
\mathrm{softmax}_i \left(
\frac{q_{r,t}^{\top} k_{r,t}^{(i)}}{\sqrt{d}}
\right).
$$

然后得到 residue-level evidence：

$$
e_{r,t} = \sum_{i=1}^K \alpha_{i,r,t} v_{r,t}^{(i)}.
$$

$e_{r,t}$ 的含义是：posterior 从 ensemble 中提炼出的、对 mode disambiguation 有用的 residue-level heterogeneity evidence。

它不是显式物理动力学标签。它只是帮助模型区分“当前该走哪一种构象 mode”。

### 5.4 Global Posterior Latent

在得到 residue-level evidence 后，再做一次全局汇聚，得到 posterior latent 的高斯参数：

$$
g_t = \mathrm{Pool}(\{e_{r,t}\}_{r=1}^L),
$$

$$
(\mu_q, \log \sigma_q) = \mathrm{MLP}(g_t).
$$

于是：

$$
q_\phi(z \mid \mathcal{X}_t, c, t)
=
\mathcal{N}(z; \mu_q, \mathrm{diag}(\sigma_q^2)).
$$

训练时通过 reparameterization trick 采样：

$$
z = \mu_q + \sigma_q \odot \epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I).
$$

这一步把局部异质性证据压缩成一个全局 mode code。

## 6. Prior 分支

测试时没有 ensemble，因此不能使用 posterior。必须学习一个只依赖 sequence 的 prior：

$$
p_\theta(z \mid c).
$$

最简单的参数化方式是：

$$
g_c = \mathrm{Pool}(\{s_r\}_{r=1}^L),
$$

$$
(\mu_p, \log \sigma_p) = \mathrm{MLP}(g_c),
$$

$$
p_\theta(z \mid c)
=
\mathcal{N}(z; \mu_p, \mathrm{diag}(\sigma_p^2)).
$$

这里不建议让 prior 依赖 $x_t$。否则测试时会出现循环依赖：你需要先知道当前 trajectory 才能决定 latent，而当前 trajectory 又取决于 latent。

## 7. Velocity Decoder

真正生成结构的是 latent-conditioned flow decoder：

$$
\hat{u}_t = v_\psi(x_t, t, c, z).
$$

其中：

- $x_t$ 是当前 noisy state
- $t$ 是时间
- $c$ 是 sequence condition
- $z$ 是 posterior 或 prior 给出的 mode latent

这里最重要的问题不是再堆更多 block，而是明确 $z$ 如何进入 decoder。

最合适的方式是：

- 不把 $z$ 直接拼到坐标上
- 把 $z$ 作为 trunk blocks 的 conditioning signal
- 通过 FiLM、AdaLN 或 gated modulation 影响各层更新

这样 decoder 学到的是一个 conditioned family of vector fields，而不是一个统一场上最后临时加一点偏置。

## 8. 训练时 decoder 用哪些 conformers

训练时有两种可行方案。

### 方案 A

posterior 用整个 ensemble，decoder 对所有 conformers 都做 velocity regression，然后平均损失。

### 方案 B

posterior 用整个 ensemble，但 decoder 只随机选一个 conformer $j$ 做回归：

$$
\hat{u}_t^{(j)} = v_\psi(x_t^{(j)}, t, c, z),
$$

$$
u_t^{(j)} = x_1^{(j)} - x_0.
$$

对于初版实现，更推荐方案 B。原因很简单：显存更省，主链路更容易先跑通。

## 9. 损失函数

整套方法的训练目标可以写成：

$$
\mathcal{L}
=
\underbrace{
\mathbb{E}_{z \sim q_\phi(z \mid \mathcal{X}_t, c, t)}
\Bigl(
\left\|
v_\psi(x_t^{(j)}, t, c, z) - u_t^{(j)}
\right\|^2
\Bigr)
}_{\text{latent-conditioned flow matching}}
+
\beta
\underbrace{
\mathrm{KL}\bigl(
q_\phi(z \mid \mathcal{X}_t, c, t)
\;\|\;
p_\theta(z \mid c)
\bigr)
}_{\text{posterior-to-prior transfer}}.
$$

两项各自负责的事情非常明确：

- 第一项让 posterior 提取对 velocity mode selection 有用的信息
- 第二项让这些信息可以被 sequence-only prior 拟合，从而在测试阶段采样

## 10. 这套方法到底学到了什么

这里需要把叙述说准确。

cross-attention 没有显式 dynamics label 监督，因此不能说它“学到了真实动力学”。更准确的说法是：

它学到了对 multi-modal velocity disambiguation 有用的 conformational heterogeneity signal。

这已经足够，因为当前方法的目标并不是恢复真实时间尺度或速率常数，而是避免 deterministic FM 在多构象条件下发生 conditional averaging。

## 11. 推断阶段

推断时的流程应与训练时严格分离。

给定 sequence condition $c$：

1. 从 prior 计算 $p_\theta(z \mid c)$
2. 采样 latent：$z \sim p_\theta(z \mid c)$
3. 采样初始 base state：$x_0 \sim p_0$
4. 用 latent-conditioned velocity field 从 $t = 0$ 积分到 $t = 1$

动力系统写成：

$$
\frac{d x_t}{d t} = v_\psi(x_t, t, c, z).
$$

最终得到一个结构样本 $x_1$。

重复采样即可得到 sequence-conditioned ensemble。

## 12. 多样性的两种来源

这套方法生成的多样性有两层：

### 12.1 跨 latent 的多样性

不同的 $z$ 对应不同的 mode-level transport pattern。这部分多样性反映的是 mode diversity。

### 12.2 同一 latent 下的多样性

即使固定同一个 $z$，不同的初始 $x_0$ 也可能带来不同样本。这部分多样性更接近 intra-mode sample variation。

后续评估时，最好把这两类多样性区分开看。

## 13. 对当前代码重构的直接含义

如果按这个设计重构代码，那么关键变化不是“多生成几个样本”，而是下面四件事：

1. 数据单位从单 conformer 扩展到 sequence-centered ensemble
2. 新增 posterior encoder，用 ensemble 推断 $q_\phi(z \mid \mathcal{X}_t, c, t)$
3. 新增 prior encoder，用 sequence 推断 $p_\theta(z \mid c)$
4. 修改 flow decoder，使其显式接收 latent conditioning

如果只是在现有模型上重复采样不同的 $x_0$，但没有 latent posterior/prior 这条链路，那么得到的仍然不是这里定义的 ensemble-supervised variational latent flow matching。

## 14. 总结

整套方法可以概括成一句话：

训练时，用同一 sequence 的 conformer ensemble 推断一个 mode latent；测试时，用 sequence-only prior 采样该 latent；然后让 latent-conditioned flow decoder 生成该 sequence 的结构 ensemble。

这条设计的目标不是让模型直接输出一个 conformer set，而是先学会“构象 mode 的条件化”，再通过 flow sampling 生成对应 mode 的结构样本。
