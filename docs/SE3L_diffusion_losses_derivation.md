# 从 Stratonovich 型条件随机微分方程到 diffusion term 的两个 loss 的完整推导

先把结论放前面：

从我们定义的 Stratonovich 型条件随机微分方程出发，**第一个 diffusion loss（quadratic-variation loss）是可以严格推出来的**；它直接对应“生成过程中到底向第 $i$ 个 residue 累计注入了多少二阶不确定性”。而**第二个 diffusion loss（terminal empirical covariance loss）不是第一个 loss 的重复，而是另一个层面的约束**：它约束的是“这些中间注入的不确定性，经过漂移场传播、压缩、放大以后，在终点真正留下了多大的 ensemble spread”。前者约束“注入机制”，后者约束“终态结果”。这两个 loss 都从同一条 SDE 出发，但对应的是二阶统计的两个不同层面。

---

## 1. 起点：条件 Stratonovich 随机运输方程

我们的方法从这条方程出发：

$$
dG_t
=
V_\theta(G_t,t,c)\,dt
+
\Sigma_{\theta,\psi}(G_t,t,c,\widetilde B)\circ dW_t,
\qquad
G_t\in \mathrm{SE}(3)^L.
$$

这里：

- $c$ 是序列条件；
- $\widetilde B$ 是由 B-factor 归一化得到的 residue-wise uncertainty field；
- $V_\theta$ 是漂移项；
- $\Sigma_{\theta,\psi}$ 是扩散项。

这个式子的意思是：结构不是沿一条完全确定的轨迹走，而是在“平均折叠方向”之外，还允许一个受 $\widetilde B$ 调制的随机展开。

因为 B-factor 对应的是**位置不确定性 / 原子位移**，所以从这条群值 SDE 推 loss 时，最自然的是先投影到平移通道，而不是直接对整条 $\mathrm{SE}(3)^L$ 方程做抽象几何运算。记

$$
X_t := \Pi_x(G_t)\in \mathbb R^{3L},
$$

也就是把每个 residue 的平移向量 $x_t^{(i)}$ 拼起来。于是平移通道满足一个普通的 Stratonovich SDE：

$$
dX_t
=
b^{\mathrm S}(X_t,t,c)\,dt
+
\sigma(X_t,t,c,\widetilde B)\circ dW_t.
$$

其中 $\sigma\in\mathbb R^{3L\times r}$ 是平移扩散矩阵。后面出现的所有二阶量，都会从 $\sigma$ 推出来。

---

## 2. 第一步：把 Stratonovich 改写成 Itô

为了推二阶统计，我们先把上式变成 Itô 形式：

$$
dX_t
=
b^{\mathrm I}(X_t,t,c,\widetilde B)\,dt
+
\sigma(X_t,t,c,\widetilde B)\,dW_t.
$$

其中 Itô 漂移与 Stratonovich 漂移的关系是

$$
b_k^{\mathrm I}
=
b_k^{\mathrm S}
+
\frac12
\sum_{a=1}^r\sum_{\ell=1}^{3L}
\sigma_{\ell a}\,\partial_{x_\ell}\sigma_{ka}.
$$

也就是说，Stratonovich 与 Itô 的扩散矩阵 $\sigma$ 是同一个，变化只发生在漂移项上：当噪声是状态依赖的 multiplicative noise 时，会多出一个额外 drift correction。

接下来定义整条链平移通道的扩散张量

$$
D_t := \sigma_t \sigma_t^\top \in \mathbb R^{3L\times 3L}.
$$

把它按 residue 切成 $L\times L$ 个 $3\times 3$ block，其中第 $i$ 个对角块记作

$$
D_{x,i}(t) \in \mathbb R^{3\times 3}.
$$

这个 $D_{x,i}(t)$ 就是第 $i$ 个 residue 在时刻 $t$ 的**局部平移扩散张量**。如果它大，说明该 residue 在这一时刻允许更大的随机 spread；如果它小，说明该 residue 被约束得更紧。

---

## 3. 第一个 loss：从 quadratic variation 严格推出

这一步最干净，也最严格。

对 Itô 过程

$$
dX_t = b_t\,dt+\sigma_t\,dW_t,
$$

其 quadratic variation 只来自随机项，不来自漂移项。直观原因是：漂移增量是 $O(dt)$，平方后是 $O(dt^2)$，极限里消失；随机增量是 $O(\sqrt{dt})$，平方后是 $O(dt)$，极限里留下来。

于是整条平移过程的 quadratic variation 满足

$$
[X]_t = \int_0^t D_s\,ds.
$$

取第 $i$ 个 residue 的 $3\times 3$ 对角块，就得到

$$
[X^{(i)}]_t = \int_0^t D_{x,i}(s)\,ds.
$$

再取 trace，把矩阵变成一个标量：

$$
q_i(\omega)
:=
\mathrm{Tr}[X^{(i)}]_1
=
\int_0^1 \mathrm{Tr}\!\bigl(D_{x,i}(s)\bigr)\,ds.
$$

这个 $q_i(\omega)$ 的意义非常明确：它是**沿着单条生成轨迹，模型向第 $i$ 个 residue 累计注入的总随机不确定性能量**。它完全由 diffusion term 决定，和漂移项本身无关。

但训练标签 $\widetilde B_i$ 是一个确定性的 residue-wise 标量，而 $q_i(\omega)$ 沿单条轨迹仍然是随机的。所以自然地，我们取条件期望：

$$
Q_i^{\theta,\psi}(c,\widetilde B)
:=
\mathbb E\!\left[
q_i(\omega)\mid c,\widetilde B
\right]
=
\mathbb E\!\left[
\int_0^1
\mathrm{Tr}\!\bigl(D_{x,i}^{\theta,\psi}(G_t,t,c,\widetilde B)\bigr)\,dt
\;\middle|\;
c,\widetilde B
\right].
$$

这一步的含义是：从随机轨迹的 pathwise 二次变差，转成给定条件下的可学习、可比较的 deterministic uncertainty scalar。

由于原始 B-factor 会受到整体尺度、实验条件、分辨率等影响，直接比较绝对数值并不稳妥，所以我们不直接让 $Q_i$ 去拟合原始 $B_i$，而是先做和 B-factor 相同风格的稳定化处理，例如

$$
\bar q_i
=
\mathrm{Norm}\!\bigl(\log(Q_i^{\theta,\psi}+\varepsilon)\bigr),
\qquad
\widetilde B_i
=
\mathrm{Norm}\!\bigl(\log(B_i+\varepsilon)\bigr).
$$

然后定义第一个 diffusion loss：

$$
\mathcal L_{\mathrm{qv}}
=
\sum_{i=1}^L
(\bar q_i-\widetilde B_i)^2.
$$

这就是第一个 loss 的完整来源。它不是 heuristic，而是直接从 SDE 的 quadratic variation 推出来的。这里真正被监督的是：**扩散项沿路径累计注入的不确定性，是否在 residue 维度上和实验 B-profile 的相对分布一致。**

如果要压缩成一句话：

$$
\boxed{
\mathcal L_{\mathrm{qv}}
\text{ 约束的是 diffusion term 沿生成路径累计注入的 uncertainty budget。}
}
$$

---

## 4. 为什么光有 $\mathcal L_{\mathrm{qv}}$ 不够

因为 $Q_i$ 只约束“中间一共注入了多少随机性”，但**并不保证终点真的留下了对应大小的 ensemble spread**。

原因是：扩散在中间时刻注入的随机性，之后会被漂移场继续运输、压缩、放大、旋转和耦合。也就是说，路径上注入的噪声，与终点的实际方差，不是同一个东西。即使两个模型有相同的 $Q_i$，只要漂移场不同，它们的最终 ensemble spread 也可能不一样。

所以，如果只用 $\mathcal L_{\mathrm{qv}}$，模型可能出现一种“取巧”行为：路径中间注入了足够的 noise，因此 $Q_i$ 看起来对了；但后续漂移又把这些 spread 全部收缩掉，最后生成 ensemble 仍然很窄。为了避免这种现象，需要第二个 loss 去直接约束**终态结果**。

---

## 5. 第二个 loss：从二阶矩演化到终态协方差

还是从 Itô 形式开始：

$$
dX_t = b_t\,dt+\sigma_t\,dW_t.
$$

现在考虑二阶矩 $X_tX_t^\top$。对它应用 Itô 公式，有

$$
d(X_tX_t^\top)
=
X_t\,dX_t^\top
+
dX_t\,X_t^\top
+
dX_t\,dX_t^\top.
$$

把 $dX_t$ 代进去：

$$
d(X_tX_t^\top)
=
\bigl(X_t b_t^\top + b_t X_t^\top + D_t\bigr)\,dt
+
X_t \sigma_t\,dW_t^\top
+
\sigma_t dW_t\,X_t^\top.
$$

这里最关键的是最后的二阶项：

$$
dX_t\,dX_t^\top = \sigma_t\sigma_t^\top\,dt = D_t\,dt.
$$

也就是说，**终态二阶矩中的“源项”仍然正是 diffusion tensor**。

对上式取条件期望，随机积分项消失，得到二阶矩演化方程

$$
\frac{d}{dt}M_t
=
\mathbb E[X_t b_t^\top + b_t X_t^\top + D_t \mid c,\widetilde B],
\qquad
M_t:=\mathbb E[X_tX_t^\top\mid c,\widetilde B].
$$

再记均值

$$
\mu_t := \mathbb E[X_t\mid c,\widetilde B],
$$

则协方差

$$
C_t := M_t - \mu_t\mu_t^\top
$$

满足精确关系

$$
\frac{d}{dt}C_t
=
\mathbb E[(X_t-\mu_t)b_t^\top + b_t(X_t-\mu_t)^\top \mid c,\widetilde B]
+
\mathbb E[D_t\mid c,\widetilde B].
$$

这个式子说明：

- 第一项来自漂移，它决定已有不确定性如何被运输、拉伸、收缩；
- 第二项来自 diffusion，它是真正往系统里“注入”新方差的源项。

如果进一步在均值轨迹附近做局部线性化，

$$
b(x,t)\approx b(\mu_t,t)+J_t(x-\mu_t),
\qquad
J_t:=\partial_x b(\mu_t,t),
$$

并把扩散张量近似成 $\bar D_t$，就得到熟悉的 Lyapunov 型近似：

$$
\dot C_t \approx J_t C_t + C_t J_t^\top + \bar D_t.
$$

这个式子把第二个 loss 的动机讲得非常清楚：**最终协方差并不等于路径上的 quadratic variation；它是 quadratic variation 经过漂移场传播后的终态结果。** 因此，若想让 ensemble 的终态 spread 也符合 B-profile，必须直接约束终态协方差。

---

## 6. 从理论协方差到可训练的 empirical covariance loss

理论上，我们想匹配的是终态条件协方差

$$
C_1 = \mathrm{Cov}(X_1\mid c,\widetilde B).
$$

但训练时没有 closed form，所以只能用 Monte Carlo rollout 来估计。

对同一个 $(c,\widetilde B)$，采样 $M$ 次独立随机生成轨迹，得到终态样本

$$
\{X_1^{(m)}\}_{m=1}^M.
$$

其中第 $i$ 个 residue 的终态位置均值为

$$
\bar x_1^{(i)}
=
\frac1M\sum_{m=1}^M x_1^{(m,i)},
$$

经验协方差为

$$
\widehat C_i
=
\frac1M
\sum_{m=1}^M
\bigl(x_1^{(m,i)}-\bar x_1^{(i)}\bigr)
\bigl(x_1^{(m,i)}-\bar x_1^{(i)}\bigr)^\top.
$$

再取一个标量 summary，例如 trace：

$$
\hat s_i^2 := \mathrm{Tr}(\widehat C_i).
$$

这表示第 $i$ 个 residue 在终态 ensemble 中实际表现出来的 positional spread。然后和前面一样做稳定化：

$$
\bar c_i
=
\mathrm{Norm}\!\bigl(\log(\hat s_i^2+\varepsilon)\bigr).
$$

于是第二个 diffusion loss 就定义成

$$
\mathcal L_{\mathrm{cov}}
=
\sum_{i=1}^L
(\bar c_i-\widetilde B_i)^2.
$$

这就是第二个 loss 的完整来源：它来自终态协方差，而终态协方差又是由同一条 SDE 的二阶矩演化式推出来的。这里它不再只看 diffusion “注入了多少”，而是看**终点实际还剩下多少可见 spread**。

如果压成一句：

$$
\boxed{
\mathcal L_{\mathrm{cov}}
\text{ 约束的是 diffusion 注入的 uncertainty 在经过漂移传播后，终点真正表现出来的 ensemble spread。}
}
$$

---

## 7. 两个 loss 的关系：不是重复，而是“机制 + 结果”

从同一条 Stratonovich SDE 出发，经过 Itô 化以后，diffusion term 的核心对象始终是

$$
D_t=\sigma_t\sigma_t^\top.
$$

由它可以推出两类二阶统计：

第一类是**路径级**的 quadratic variation，

$$
Q_i
=
\mathbb E\!\left[\int_0^1 \mathrm{Tr}(D_{x,i}(t))dt\mid c,\widetilde B\right],
$$

对应第一个 loss

$$
\mathcal L_{\mathrm{qv}}.
$$

第二类是**终态级**的 covariance，

$$
C_1 = \mathrm{Cov}(X_1\mid c,\widetilde B),
$$

对应第二个 loss

$$
\mathcal L_{\mathrm{cov}}.
$$

前者监督“diffusion mechanism”；后者监督“diffusion outcome”。前者严格从 quadratic variation 得到；后者严格从 Itô 二阶矩演化得到，但在训练中要用 Monte Carlo 经验协方差去近似。二者都源自同一个 diffusion tensor $D_t$，但不是同一个量。

因此，完整的 diffusion 部分目标可以写成

$$
\mathcal L_{\mathrm{diff}}
=
\lambda_{\mathrm{qv}}\,\mathcal L_{\mathrm{qv}}
+
\lambda_{\mathrm{cov}}\,\mathcal L_{\mathrm{cov}}.
$$

于是可以非常干净地说：

**我们不是直接让模型回归 B-factor，而是让 B-factor 去约束 diffusion term 所诱导的二阶统计：一路上的累计不确定性，以及终点的实际 ensemble spread。**

---

## 8. 可直接放进论文的方法小结

从条件 Stratonovich 随机运输方程

$$
dG_t = V_\theta(G_t,t,c)\,dt + \Sigma_{\theta,\psi}(G_t,t,c,\widetilde B)\circ dW_t
$$

出发，我们仅对其平移投影 $X_t=\Pi_x(G_t)$ 进行 uncertainty calibration，因为 B-factor 本质上表征的是 positional uncertainty。将其改写为 Itô 形式

$$
dX_t = b_t\,dt+\sigma_t\,dW_t
$$

后，定义扩散张量 $D_t=\sigma_t\sigma_t^\top$。对第 $i$ 个 residue 的平移块 $D_{x,i}(t)$，其路径级二次变差满足

$$
[X^{(i)}]_1 = \int_0^1 D_{x,i}(t)\,dt,
$$

因此 residue-wise 累计不确定性可定义为

$$
Q_i = \mathbb E\!\left[\int_0^1 \mathrm{Tr}(D_{x,i}(t))dt \mid c,\widetilde B\right].
$$

这给出 quadratic-variation calibration loss。另一方面，对二阶矩 $X_tX_t^\top$ 应用 Itô 公式可得，终态协方差由漂移传播项与扩散源项共同决定，因此我们进一步通过多次随机 rollout 的经验终态协方差来构造 covariance calibration loss。前者约束 diffusion 沿路径累计注入的不确定性，后者约束这些不确定性在终态真正表现出来的 ensemble spread。
