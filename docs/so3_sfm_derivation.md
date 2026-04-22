# Level 2 on $SO(3)$：从完备推导到一阶近似

本文档给出一套**自洽的、可直接写进 methods / appendix** 的推导，回答下面这个问题：

> 在当前 Level 2 方案中，我们把旋转前向过程定义为一个 **tangent-space-lifted stochastic bridge**
> $$
> \Omega_t = t\,\omega + \sigma_\omega\sqrt{t(1-t)}\,\varepsilon, \qquad
> R_t = R_0\,\Exp(\widehat{\Omega_t}),
> $$
> 其中
> $$
> \omega = \Log(R_0^\top R_1)^\vee.
> $$
> 那么，训练时 $SO(3)$ 上的 **理论完备** target 应该是什么？
> 进一步，代码里更友好的一阶近似 target 又是如何从完备式推出的？

下文的结论是：

1. **理论完备的 exact target** 是先在李代数 $\mathfrak{so}(3)\cong\mathbb R^3$ 上写出 stochastic interpolant 的条件速度，再通过右 Jacobian $J_r(\Omega_t)$ 推到群上；
2. **代码友好的一阶近似** 则是把 $J_r(\Omega_t)$ 近似成单位阵，并把不同切空间之间的精确变换用一阶 BCH 近似替代，最终得到你当前写法里的
   $$
   \Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}.
   $$

这份推导严格以你当前架构报告中的 Level 2 定义为起点：Level 2 的旋转桥是在 $\mathfrak{so}(3)$ 中定义闭式 Brownian bridge，再通过指数映射推前到 $SO(3)$，而不是 intrinsic Brownian bridge；同时，Level 2–3 的 loss 已经被你改写成 stochastic flow matching (SFM) 形式，而不是旧的 deterministic FM 形式。fileciteturn15file0

---

## 1. 当前 Level 2 的前向过程

按照当前方案，旋转部分的前向过程是：

$$
\omega = \Log(R_0^\top R_1)^\vee \in \mathbb R^3,
$$

$$
\Omega_t = t\omega + \sigma_\omega\sqrt{t(1-t)}\,\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I_3),
$$

$$
R_t = R_0\Exp(\widehat{\Omega_t}).
$$

并且定义 bridge noise

$$
\delta_t^\Omega \triangleq \Omega_t - t\omega.
$$

这就是你报告里 Level 2 的“切空间 BB + exp pushforward”定义。它有闭式采样、严格端点约束，但不是 intrinsic $SO(3)$ Brownian bridge，而是一个 chart-induced / tangent-space-lifted stochastic bridge。fileciteturn15file0

因此，**如果我们要理论完备地写 loss，就必须从这个 forward path 本身出发推导 target，而不能直接借用 FoldFlow-SFM 对另一种 $SO(3)$ bridge 的 loss 形式。** FoldFlow-SFM 的 $SO(3)$ 部分使用的是它自己定义的 guided diffusion bridge / approximation；你的桥和它的桥不是同一个对象。fileciteturn15file0fileciteturn0file0

---

## 2. 在 $\mathfrak{so}(3)\cong\mathbb R^3$ 上的 exact SFM 速度场

### 2.1 写成 stochastic interpolant 的标准形式

把 $\Omega_t$ 写成一般的 stochastic interpolant 形式：

$$
\Omega_t = \mu_t + \sigma_t\,\varepsilon,
$$

其中

$$
\mu_t = t\omega,
\qquad
\sigma_t = \sigma_\omega\sqrt{t(1-t)}.
$$

于是

$$
\delta_t^\Omega = \Omega_t - \mu_t = \sigma_t\varepsilon.
$$

对于一般的 Euclidean stochastic interpolant

$$
z_t = \mu_t + \sigma_t\varepsilon,
$$

其 exact conditional velocity target 为

$$
u_t^{\text{exact}}(z_t)
= \dot\mu_t + \frac{\dot\sigma_t}{\sigma_t}(z_t-\mu_t).
$$

这是你已经在 $\mathbb R^3$ 部分采用的 SFM 逻辑；当前架构报告也把这条思路明确移植到了旋转桥的切空间表示上。fileciteturn15file0

### 2.2 代入当前 $\Omega_t$

先计算

$$
\dot\mu_t = \omega.
$$

再计算

$$
\sigma_t = \sigma_\omega\sqrt{t(1-t)}
\quad\Longrightarrow\quad
\frac{\dot\sigma_t}{\sigma_t} = \frac{1-2t}{2t(1-t)}.
$$

因此，李代数中的 exact conditional velocity 就是

$$
\boxed{
 u_t^\Omega
 = \omega + \frac{1-2t}{2t(1-t)}\,\delta_t^\Omega
}
$$

这正是你当前架构报告中写出的切空间 SFM velocity。fileciteturn15file0

这一步非常关键，因为它说明：

- 如果你已经决定 **$\mathbb R^3$ 上用 SFM exact target**；
- 那么 **$SO(3)$ 上也应该先在 $\mathfrak{so}(3)$ 中用完全同样的 SFM exact target**。

否则 translational part 和 rotational part 又会重新变成两套不同理论。

---

## 3. 从李代数速度到 $SO(3)$ 上的 exact body velocity

上一步得到的 $u_t^\Omega$ 仍然是 **李代数坐标 $\Omega_t$ 的时间导数**。但模型在旋转部分真正对应的是群上的角速度 / tangent vector，因此还需要把这条速度通过指数映射推到 $SO(3)$ 上。

### 3.1 曲线定义

考虑群上的轨迹

$$
R_t = R_0\Exp(\widehat{\Omega_t}).
$$

由于 $R_0$ 是常数，真正需要处理的是

$$
Q_t = \Exp(\widehat{\Omega_t}).
$$

对 Lie group 上的指数曲线，若使用 body-frame 角速度表示，则有标准关系

$$
Q_t^\top \dot Q_t = \widehat{J_r(\Omega_t)\,\dot\Omega_t},
$$

其中 $J_r(\Omega_t)$ 是 $SO(3)$ 的右 Jacobian。

因此，群上的 exact body velocity 为

$$
a_t^\star = J_r(\Omega_t)\,\dot\Omega_t.
$$

把上节得到的 exact SFM velocity 代入，就得到

$$
\boxed{
 a_t^\star
 =
 J_r(\Omega_t)
 \left[
 \omega + \frac{1-2t}{2t(1-t)}\,\delta_t^\Omega
 \right]
}
$$

这就是当前 Level 2 方案下，**理论完备**、与 forward path 完全一致的 $SO(3)$ target。

---

## 4. 这是“最终正确”的版本，为什么？

因为它满足三个一致性条件：

### 4.1 与当前 forward corruption 一致

它直接从

$$
\Omega_t = t\omega + \sigma_\omega\sqrt{t(1-t)}\varepsilon
$$

推出，而不是从别的桥推出。

### 4.2 与 $\mathbb R^3$ 的 SFM 逻辑一致

$\mathbb R^3$ 上你已经接受：

$$
u_t^x = \dot\mu_t + \frac{\dot\sigma_t}{\sigma_t}(x_t-\mu_t).
$$

那么旋转在切空间里自然也应写成同样形式：

$$
u_t^\Omega = \dot\mu_t + \frac{\dot\sigma_t}{\sigma_t}(\Omega_t-\mu_t).
$$

### 4.3 与群几何一致

你没有直接在 $SO(3)$ 上定义 intrinsic bridge，而是先在李代数里定义路径，再通过 $\Exp$ 推前到群上。那么 target 也必须经过同样的群几何映射，这就是 $J_r(\Omega_t)$ 出现的原因。

因此，**如果你问“哪个才是正确的、理论完备的？”答案就是这一条：**

$$
\boxed{
 a_t^\star
 =
 J_r(\Omega_t)
 \left[
 \omega + \frac{1-2t}{2t(1-t)}\,\delta_t^\Omega
 \right]
}
$$

---

## 5. 如何把 exact target 改写成 endpoint-parameterized loss

你当前代码不是直接预测 body velocity，而是用 endpoint parameterization：

$$
v_\theta^\omega(t,R_t)
=
\frac{1}{1-t}\Log(R_t^\top \hat R_{1|t})^\vee.
$$

因此，若要与 exact target 对齐，最自然的 exact loss 是

$$
\mathcal L_{SO(3)}^{\text{exact}}
=
 w_\omega
 \,\mathbb E
 \left[
 \left\|
 \frac{1}{1-t}\Log(R_t^\top \hat R_{1|t})^\vee
 - a_t^\star
 \right\|_2^2
 \right].
$$

等价地，也可以把 $(1-t)$ 乘到 target 上，写成更像 endpoint regression 的形式：

$$
\mathcal L_{SO(3)}^{\text{exact}}
=
 w_\omega
 \,\mathbb E
 \left[
 \left\|
 \Log(R_t^\top \hat R_{1|t})^\vee
 - (1-t)a_t^\star
 \right\|_2^2
 \right].
$$

这时 target 是

$$
(1-t)a_t^\star
=
(1-t)
J_r(\Omega_t)
\left[
\omega + \frac{1-2t}{2t(1-t)}\delta_t^\Omega
\right].
$$

这是 **代码仍然保留 endpoint head，但 target 使用 exact theory** 的写法。

---

## 6. 从 exact target 推到你当前的一阶近似

现在来说明：为什么你当前文档里的

$$
\Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}
$$

本质上就是上面 exact target 的一阶近似。

### 6.1 第一步：忽略 Jacobian 的高阶修正

当 $\|\Omega_t\|$ 不大时，

$$
J_r(\Omega_t) = I + O(\|\Omega_t\|).
$$

在 first-order 近似下取

$$
J_r(\Omega_t) \approx I.
$$

于是

$$
a_t^\star
\approx
\omega + \frac{1-2t}{2t(1-t)}\delta_t^\Omega.
$$

再乘上 $(1-t)$：

$$
(1-t)a_t^\star
\approx
(1-t)\omega + \frac{1-2t}{2t}\delta_t^\Omega.
$$

### 6.2 第二步：把 $(1-t)\omega$ 改写成 $(\omega-\Omega_t)+\text{noise term}$

由

$$
\Omega_t = t\omega + \delta_t^\Omega
$$

可得

$$
\omega - \Omega_t = (1-t)\omega - \delta_t^\Omega.
$$

所以

$$
(1-t)\omega
;=
(\omega-\Omega_t)+\delta_t^\Omega.
$$

代回去：

$$
(1-t)a_t^\star
\approx
(\omega-\Omega_t)+\delta_t^\Omega + \frac{1-2t}{2t}\delta_t^\Omega.
$$

合并系数：

$$
1 + \frac{1-2t}{2t} = \frac{1}{2t}.
$$

因此

$$
(1-t)a_t^\star
\approx
(\omega-\Omega_t)+\frac{\delta_t^\Omega}{2t}.
$$

### 6.3 第三步：把 $(\omega-\Omega_t)$ 替换成 endpoint-friendly 量

现在的问题是，$(\omega-\Omega_t)$ living in the tangent at $R_0$，而代码里最方便拿到的是

$$
\Log(R_t^\top R_1)^\vee,
$$

它 living in the body frame of $R_t$。

对

$$
R_t = R_0\Exp(\widehat{\Omega_t}),
\qquad
R_1 = R_0\Exp(\widehat{\omega}),
$$

有

$$
R_t^\top R_1
=
\Exp(-\widehat{\Omega_t})\Exp(\widehat{\omega}).
$$

利用 BCH 在一阶截断下，

$$
\Log\big(\Exp(-\widehat{\Omega_t})\Exp(\widehat{\omega})\big)^\vee
\approx
\omega-\Omega_t.
$$

于是得到

$$
\boxed{
(1-t)a_t^\star
\approx
\Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}
}
$$

这正是你现在文档里写的 code-friendly target。fileciteturn15file0

---

## 7. 因此，这两个版本的关系是什么？

现在可以非常清楚地总结：

### 7.1 理论完备版

$$
\boxed{
 a_t^\star
 =
 J_r(\Omega_t)
 \left[
 \omega + \frac{1-2t}{2t(1-t)}\,\delta_t^\Omega
 \right]
}
$$

这是与当前 Level 2 forward path **完全严格对齐** 的 exact target。

### 7.2 代码友好一阶近似版

$$
\boxed{
(1-t)a_t^\star
\approx
\Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}
}
$$

这是把 exact target 经过

1. $J_r(\Omega_t)\approx I$；
2. BCH 一阶截断；
3. 不同切空间之间用 identity-style approximation 识别

之后得到的实现形式。

所以，**它们不是两套不同方法，而是同一个 target 的 exact form 与 first-order approximation。**

---

## 8. 为什么旧的 drift-style 版本不再是最合适的主公式

你之前还出现过一个版本：

$$
J_r(\Omega_t)\frac{\omega-\Omega_t}{1-t}.
$$

这个式子并不是完全错，但它对应的是 **bridge drift / Brownian-bridge SDE** 的视角，也就是

$$
d\Omega_t = \frac{\omega-\Omega_t}{1-t}dt + \sigma_\omega dW_t.
$$

在那个视角下，它是自然的。

但你现在已经把 translational part 改成了 **SFM / stochastic interpolant exact target**，即

$$
\dot\mu_t + \frac{\dot\sigma_t}{\sigma_t}(x_t-\mu_t).
$$

那么 rotational part 最干净的做法也必须跟着改成完全同一种 SFM 形式，否则 $\mathbb R^3$ 和 $SO(3)$ 又会重新变成两套理论。当前架构报告也已经把 Level 2–3 的主损失重新表述成 SFM loss，而不是旧的 deterministic FM loss。fileciteturn15file0

所以，**今后正文里最该保留的是 exact SFM 版，而不是旧的 drift-only 版。**

---

## 9. 最终推荐的论文写法

最稳的写法是把正文和实现分开写。

### 9.1 Methods 正文：给 exact form

你可以在正文里写：

> For the lifted stochastic bridge on $SO(3)$, we first define the Lie-algebra interpolant
> $$
> \Omega_t = t\omega + \sigma_\omega\sqrt{t(1-t)}\,\varepsilon,
> \qquad \omega = \Log(R_0^\top R_1)^\vee,
> $$
> and map it to the group via $R_t = R_0\Exp(\widehat{\Omega_t})$. The exact conditional velocity in the Lie algebra is
> $$
> u_t^\Omega = \omega + \frac{1-2t}{2t(1-t)}(\Omega_t-t\omega),
> $$
> which induces the exact body-frame target on $SO(3)$
> $$
> a_t^\star = J_r(\Omega_t)u_t^\Omega.
> $$

### 9.2 Implementation / Appendix：给一阶近似

然后说明：

> In implementation, we use the first-order approximation
> $$
> (1-t)a_t^\star \approx \Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t},
> $$
> obtained by approximating $J_r(\Omega_t)\approx I$ and truncating the BCH expansion to first order.

这样 reviewer 看起来会非常清楚：

- 理论上你是完备的；
- 实现上你没有假装 exact；
- 近似来源写得很诚实。

---

## 10. 最终可直接使用的 loss 形式

### 10.1 Exact theoretical loss

若旋转 head 预测 body velocity，则

$$
\boxed{
\mathcal L_{SO(3)}^{\text{exact}}
=
 w_\omega
 \,\mathbb E
 \left[
 \left\|
 v_\theta^\omega(t,R_t)
 -
 J_r(\Omega_t)
 \left(
 \omega + \frac{1-2t}{2t(1-t)}\delta_t^\Omega
 \right)
 \right\|_2^2
 \right]
}
$$

若旋转 head 仍用 endpoint parameterization，则

$$
\boxed{
\mathcal L_{SO(3)}^{\text{exact}}
=
 w_\omega
 \,\mathbb E
 \left[
 \left\|
 \Log(R_t^\top \hat R_{1|t})^\vee
 -
 (1-t)
 J_r(\Omega_t)
 \left(
 \omega + \frac{1-2t}{2t(1-t)}\delta_t^\Omega
 \right)
 \right\|_2^2
 \right]
}
$$

### 10.2 First-order implementation loss

$$
\boxed{
\mathcal L_{SO(3)}^{\text{1st-order}}
=
 w_\omega
 \,\mathbb E
 \left[
 \left\|
 \Log(R_t^\top \hat R_{1|t})^\vee
 -
 \left(
 \Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}
 \right)
 \right\|_2^2
 \right]
}
$$

这就是你当前方案在理论与实现上的最清楚版本。

---

## 11. 一句话总结

你的当前 code-friendly 写法

$$
\Log(R_t^\top R_1)^\vee + \frac{\delta_t^\Omega}{2t}
$$

**不是另一套方法**，而是下面这个 exact target 的一阶近似：

$$
J_r(\Omega_t)
\left[
\omega + \frac{1-2t}{2t(1-t)}\delta_t^\Omega
\right].
$$

前者适合实现，后者适合论文里的理论主公式。
