最合适的写法是把 level 2 定义成一个明确的方法名：

**Conditional Heteroscedastic SE(3) Stochastic Bridge**

这样写，方法的核心就不再是“把 ODE 换成 stochastic bridge”，而是：**在 SE(3) stochastic bridge 里，引入由 sequence context 和当前几何状态共同决定的条件异方差扩散强度**。这和 FoldFlow-SFM 的关系也很清楚：FoldFlow 已经说明了，SE(3) 上可以把 deterministic bridge 替换为 stochastic bridge；translation 端用 Brownian bridge，rotation 端用 SO(3) 上的 Brownian bridge，并在训练时用 IGSO(3) 做 simulation-free approximation。它们的实验还显示，stochasticity 会提升 novelty 和 diversity，但同时可能牺牲一部分 designability。你的方法不是重复它，而是在它的 bridge family 上进一步引入 condition-dependent bridge width。 

下面我直接按“论文里的 methods 小节”来写。

---

## 3.2 Conditional Heteroscedastic SE(3) Stochastic Bridge

### Problem setup

给定 RNA 序列 (a=(a_1,\dots,a_L))，记其条件表征为
[
c=\mathrm{Enc}(a).
]
我们将第 (i) 个 residue 表示为一个 rigid frame
[
G_i=(x_i,R_i)\in SE(3),
]
其中 (x_i\in\mathbb R^3) 为平移，(R_i\in SO(3)) 为旋转。整条 RNA 的 backbone 构象记为
[
G={(x_i,R_i)}_{i=1}^L \in SE(3)^L.
]
与 level 1 中的 deterministic flow matching 不同，这里我们不再假设从 source 到 target 的条件路径是单一确定性轨迹，而是将其建模为一族满足端点约束的随机桥。这样做的动机是，ensemble generation 本质上对应一族可行的随机运输路径，而不是唯一的一条平均轨迹。FoldFlow-SFM 也是沿这个方向，把 SE(3) 上的 deterministic dynamics 替换成 stochastic bridge，并指出 stochasticity 有助于跳出训练分布支撑、提升 novelty。 

我们的关键改动在于：桥的随机强度不是全局固定的时间函数，而是**条件的、state-dependent 的、residue-wise 的**。换言之，同一条 RNA 序列在不同当前状态 (G_t) 下，桥的宽度可以不同；因此模型能够学习“哪些位置需要保持确定性，哪些位置需要保留分叉能力”。

### Conditional bridge construction

记 (G_0=(X_0,R_0)) 为目标构象，(G_1=(X_1,R_1)) 为 source sample。训练时，我们在 ((G_0,G_1)) 之间构造条件随机桥，并在任意 (t\in(0,1)) 采样中间状态 (G_t)。

对于 translation 部分，我们采用高斯桥：
[
x_{i,t} = \mu^x_{i,t} + \beta^x_{i,t},\varepsilon^x_i,
\qquad
\varepsilon^x_i \sim \mathcal N(0,I_3),
]
其中条件均值为
[
\mu^x_{i,t} = (1-t)x_{i,0} + t x_{i,1}.
]

对于 rotation 部分，我们采用 (SO(3)) 上的各向同性高斯近似桥：
[
R_{i,t} \sim \mathrm{IGSO}(3)!\left(\mu^R_{i,t}, (\beta^\omega_{i,t})^2\right),
]
其中旋转均值为 geodesic interpolation
[
\mu^R_{i,t}
===========

R_{i,0}\exp!\big(t,\log(R_{i,0}^{\top}R_{i,1})\big).
]

这里的写法直接继承了 FoldFlow-SFM 的基本桥构造：translation 端采用 Brownian bridge 形式，rotation 端采用 SO(3) 上的 stochastic bridge，并使用 IGSO(3) 作为 simulation-free 近似。区别仅在于，FoldFlow 中的噪声尺度主要是时间函数，而这里我们让它变成条件函数。

### Condition-dependent bridge width

为了让 bridge width 对序列与当前状态敏感，我们将
[
\beta^x_{i,t},\quad \beta^\omega_{i,t}
]
写成“固定 base schedule 与可学习 modulation”的乘积：
[
\beta^x_{i,t} = \sigma_x^0(t),\alpha^x_{i,t},\qquad
\beta^\omega_{i,t} = \sigma_\omega^0(t),\alpha^\omega_{i,t}.
]
其中 (\sigma_x^0(t)) 和 (\sigma_\omega^0(t)) 是预先给定的 endpoint-safe schedule，满足
[
\sigma_x^0(0)=\sigma_x^0(1)=0,\qquad
\sigma_\omega^0(0)=\sigma_\omega^0(1)=0.
]
最简单的选择是
[
\sigma_x^0(t)=\lambda_x\sqrt{t(1-t)},\qquad
\sigma_\omega^0(t)=\lambda_\omega\sqrt{t(1-t)}.
]
这样可保证桥在两个端点退化为确定性端点约束，不破坏 bridge 的基本性质。

真正的学习对象是 modulation factor
[
\alpha^x_{i,t},\quad \alpha^\omega_{i,t},
]
它们由一个小网络预测。关键是，输入不能写成“序列表征和 (G_t) 的直接差”，因为两者不在同一空间。更合理的做法是先构造一个**sequence-aware reference state**，再计算几何残差。

具体地，给定 ((c,G_t,t))，我们用一个 reference head 预测当前时刻对应的 clean estimate
[
\hat G_{0|t}={(\hat x_{i,0|t},\hat R_{i,0|t})}*{i=1}^L.
]
然后定义 translation 和 rotation 的残差
[
\Delta^x*{i,t}=x_{i,t}-\hat x_{i,0|t},
\qquad
\Delta^\omega_{i,t}=\log(\hat R_{i,0|t}^{\top}R_{i,t})\in\mathfrak{so}(3).
]
于是 modulation network 的输入为
[
z_{i,t} = [,c_i,\ \Delta^x_{i,t},\ \Delta^\omega_{i,t},\ t,].
]
最后得到
[
\alpha^x_{i,t} = \mathrm{Map}*x(f_x(z*{i,t})),
\qquad
\alpha^\omega_{i,t} = \mathrm{Map}*\omega(f*\omega(z_{i,t})),
]
其中 (\mathrm{Map}) 是一个有界映射，例如
[
\mathrm{Map}(u)=\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})\cdot \mathrm{sigmoid}(u).
]
这样设计有三个作用。第一，test time 一致，因为输入只依赖 sequence、当前状态和模型自己的 reference prediction，不依赖不可见的真实 target。第二，(\alpha) 只是在 base schedule 上做有界调制，不会破坏端点条件。第三，它把“随机性该放在哪、放多少”变成了一个可学习但受约束的问题，而不是完全自由的噪声场。

### Stochastic transport view

在连续时间上，上述条件桥可写成一个 residue-wise 的随机运输过程。为了简洁，我们分别写 translation 与 rotation：

[
d x_{i,t}
=========

u^x_\theta(G_t,c,t),dt
+
\beta^x_{i,t},dW^x_{i,t},
]
[
d R_{i,t}
=========

R_{i,t}\exp!\Big(
u^\omega_\theta(G_t,c,t),dt
+
\beta^\omega_{i,t},dW^\omega_{i,t}
\Big),
]
其中 (W^x_{i,t}) 是 (\mathbb R^3) 上的 Brownian motion，(W^\omega_{i,t}) 是 (\mathfrak{so}(3)) 上的 isotropic noise。严格地说，rotation 部分可以视为 SO(3) 上的随机过程，其离散实现通过李代数扰动再用指数映射回群上。

从建模角度看，level 1 学的是单一确定性 drift；level 2 则是在 drift 之外引入 residue-wise 的条件扩散项。并且这里的扩散不是“均匀撒噪声”，而是依赖 sequence context 与当前局部几何偏差，因此它更像一个**heteroscedastic uncertainty field on (SE(3)^L)**。

### Network parameterization

我们将主网络写成两个输出头：

其一是 clean-state predictor
[
\hat G_{0|t}=h_\theta(c,G_t,t),
]
用于给出当前状态对应的目标几何估计；

其二是 bridge-width head
[
(\alpha^x_{i,t},\alpha^\omega_{i,t}) = g_\phi(c_i,\Delta^x_{i,t},\Delta^\omega_{i,t},t).
]

为了与 flow matching 主干兼容，我们仍然将 drift 参数化为“预测 clean endpoint，再诱导 velocity”的形式。即
[
u^x_\theta(i,t)=\frac{\hat x_{i,0|t}-x_{i,t}}{t+\epsilon},
\qquad
u^\omega_\theta(i,t)=\frac{1}{t+\epsilon}\log(R_{i,t}^{\top}\hat R_{i,0|t}).
]
这和 FoldFlow 采用的“先预测 (\hat x_0)，再由 (\hat x_0) 构造 velocity”是同一类参数化思想。

### Training objective

训练时，我们先采样
[
t\sim \mathrm{Uniform}(0,1),
]
再从上述条件异方差随机桥中采样 (G_t)。随后用 flow-matching 形式监督主干 drift。对于第 (i) 个 residue，其 target 写为
[
u^{x,*}*{i,t}=\frac{x*{i,0}-x_{i,t}}{t+\epsilon},
\qquad
u^{\omega,*}*{i,t}=\frac{1}{t+\epsilon}\log(R*{i,t}^{\top}R_{i,0}).
]
于是主损失为
[
\mathcal L_{\mathrm{bridge}}
============================

\mathbb E\Big[
\sum_{i=1}^L
|u^x_\theta(i,t)-u^{x,*}_{i,t}|*2^2
+
|u^\omega*\theta(i,t)-u^{\omega,*}_{i,t}|_2^2
\Big].
]

需要注意的是，(\alpha^x,\alpha^\omega) 没有直接真值监督。如果不加约束，它们会出现可识别性不足的问题，最坏情况下会退化到把桥宽压得很小，重新接近 deterministic path。因此我们加入一个轻量正则项：
[
\mathcal L_{\sigma}
===================

\lambda_{\mathrm{mag}},
\mathbb E!\left[(\alpha^x-1)^2+(\alpha^\omega-1)^2\right]
+
\lambda_{\mathrm{time}},
\mathbb E!\left[|\partial_t \alpha^x|^2+|\partial_t \alpha^\omega|^2\right]
+
\lambda_{\mathrm{sp}},
\mathbb E!\left[\sum_{(i,j)\in \mathcal N}
(\alpha^x_{i,t}-\alpha^x_{j,t})^2
+
(\alpha^\omega_{i,t}-\alpha^\omega_{j,t})^2
\right].
]
这里第一项限制 modulation 不要偏离 base schedule 过远，第二项鼓励时间平滑，第三项鼓励局部空间平滑。

如果希望更直接地约束 (\sigma) 不要塌陷，还可以额外加一个很弱的 variance floor regularizer，例如
[
\mathcal L_{\mathrm{floor}}
===========================

\lambda_{\mathrm{floor}}
\mathbb E!\left[
\mathrm{ReLU}(\alpha_{\mathrm{low}}-\alpha^x)^2
+
\mathrm{ReLU}(\alpha_{\mathrm{low}}-\alpha^\omega)^2
\right].
]

最终损失写为
[
\mathcal L
==========

\mathcal L_{\mathrm{bridge}}
+
\mathcal L_{\sigma}
+
\lambda_{\mathrm{aux}}\mathcal L_{\mathrm{aux}},
]
其中 (\mathcal L_{\mathrm{aux}}) 可以继续沿用你 level 1 里的 backbone coordinate loss 或 atom-level auxiliary loss。

### Inference

推理时，从 source distribution 采样初始状态 (G_1)，然后从 (t=1) 积分到 (t=0)。离散化后，对 translation 我们使用 Euler–Maruyama：
[
x_{i,t-\Delta t}
================

## x_{i,t}

\Delta t,u^x_\theta(i,t)
+
\sqrt{\Delta t},\beta^x_{i,t},\xi^x_i,
\qquad
\xi^x_i\sim\mathcal N(0,I_3).
]

对 rotation，我们在李代数里加扰动后再映射回 SO(3)：
[
R_{i,t-\Delta t}
================

R_{i,t}\exp!\Big(
-\Delta t,u^\omega_\theta(i,t)
+
\sqrt{\Delta t},\beta^\omega_{i,t},\xi^\omega_i
\Big),
\qquad
\xi^\omega_i\sim\mathcal N(0,I_3).
]

这个离散采样器有两个清楚的退化关系。如果令
[
\alpha^x_{i,t}\equiv 1,\qquad \alpha^\omega_{i,t}\equiv 1,
]
那么它退化为普通的 SE(3) stochastic bridge。如果再令
[
\beta^x_{i,t}\equiv 0,\qquad \beta^\omega_{i,t}\equiv 0,
]
就进一步退化为 level 1 的 deterministic SE(3) flow matching。这个层级关系在论文叙事上很干净。

---

## Why this should help ensemble generation

这个方法之所以适合拿来讲 ensemble，不是因为“加了噪声所以能多样化”，而是因为它把随机性变成了**条件化的局部生成自由度**。

如果桥宽是固定的时间函数，那么模型只能在全局层面决定“现在整体噪声大还是小”。这对于单结构生成够用，但对 ensemble generation 不够，因为 RNA 的多模态通常不是全局均匀发生的，而是局部发生的：某些 stem 很稳定，某些 loop 或 junction 更容易出现 basin-to-basin transition。固定 (\sigma(t)) 没法表达这种 residue-wise 的差异。

而在这里，
[
\beta^x_{i,t},\ \beta^\omega_{i,t}
]
是 sequence-aware、state-aware 的。于是模型学到的不是“整个分子在时刻 (t) 有多随机”，而是“在当前 sequence context 与当前几何状态下，第 (i) 个 residue 是否正处在需要分叉的区域”。这就把 stochastic bridge 从一个全局随机过程，提升为一个**局部自适应的 ensemble transport process**。

更具体地说，这个方法缓解 ensemble generation 的逻辑是：

第一，deterministic flow 往往倾向于学习单一平均运输方向，因此容易把多个构象 basin 压成一条 mean trajectory。FoldFlow 也观察到，加入 stochasticity 后 novelty 和 diversity 提升，说明随机路径本身就在帮助生成过程逃离单一平均解。

第二，plain stochastic bridge 虽然引入了多样性，但它不知道“哪里该随机、哪里不该随机”。因此它的 stochasticity 仍然是粗粒度的。

第三，你这里的 condition-dependent bridge width 恰好补上了这一点：同一条序列在不同中间状态上，局部桥宽可以不同，从而允许模型只在潜在 mode-transition 区域放大扩散，而在稳定骨架区域保持较强确定性。于是它不只是“更随机”，而是“更有选择地随机”。

所以，如果要把这一层的 story 写成一句话，我会写成：

> Level 2 lifts deterministic transport on (SE(3)^L) to a conditional heteroscedastic stochastic bridge, enabling the model to allocate uncertainty adaptively across residues and time, and thereby represent ensemble generation as a state-dependent stochastic transport process rather than a single averaged trajectory.

---

## 这一层的 novelty 应该怎么讲

这部分你需要讲得很准，不要讲成“我们首次在 SE(3) 上做 stochastic bridge”，那是不成立的。更稳的说法是：

1. 现有 SE(3) stochastic bridge 使用固定的时间噪声 schedule；
2. 我们提出在 (SE(3)^L) 上学习一个**conditioned heteroscedastic bridge width field**；
3. 因而随机性不再是全局预设的，而是由 sequence context 与当前几何状态共同决定；
4. 这使得 stochastic bridge 从“通用多样性机制”变成“面向 ensemble 的局部不确定性分配机制”。

换句话说，你的 novelty 不在 “stochastic bridge” 这四个字，而在 **conditional heteroscedasticity on SE(3)**。

---

## 如果这节写成论文中的开头段落，我会这样写

We model RNA ensemble generation as a stochastic transport problem on (SE(3)^L). Unlike deterministic flow matching, which learns a single transport trajectory between source and target structures, our level-2 model introduces a conditional stochastic bridge whose diffusion strength is modulated by both the sequence context and the current geometric state. Concretely, we parameterize a residue-wise bridge width field over translations and rotations, allowing the model to inject uncertainty only where the current state suggests local conformational branching is needed. This yields a conditional heteroscedastic stochastic bridge on (SE(3)^L), which preserves endpoint constraints while enabling state-dependent diversification during generation.

---