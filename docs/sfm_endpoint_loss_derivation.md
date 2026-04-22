# SFM Endpoint Loss Derivation for the Stochastic Bridge

本文推导为什么在当前 $\mathbb R^3$ 和 $SO(3)$ 的 stochastic bridge 框架下，如果 FlowModel 输出的是 endpoint prediction，则 Level 2/3 的主 loss target 需要包含

$$
\frac{\delta_t}{2t}.
$$

核心结论是：

$$
\frac{\delta_t}{2t}
$$

不是普通 clean denoising 的修正，而是把 SFM velocity matching 改写成 endpoint matching 后得到的等价 target。

---

## 1. Bridge 定义

对第 $i$ 个 residue，Level 3 的平动 bridge 为

$$
x_{i,t}
=
(1-t)x_{i,0}
+t x_{i,1}
+\delta^x_{i,t},
$$

其中

$$
\delta^x_{i,t}
=
\bar\sigma_x
\sqrt{\nu_i}
\sqrt{t(1-t)}
\varepsilon^x_i,
\qquad
\varepsilon^x_i\sim\mathcal N(0,I_3).
$$

旋转部分先在李代数中定义 bridge。令

$$
\omega_i
=
\log(R_{i,0}^{\top}R_{i,1})^\vee.
$$

定义

$$
\Omega_{i,t}
=
t\omega_i+\delta^\Omega_{i,t},
$$

其中

$$
\delta^\Omega_{i,t}
=
\bar\sigma_\omega
\sqrt{\nu_i}
\sqrt{t(1-t)}
\varepsilon^\Omega_i,
\qquad
\varepsilon^\Omega_i\sim\mathcal N(0,I_3).
$$

然后 push forward 到 $SO(3)$：

$$
R_{i,t}
=
R_{i,0}\operatorname{Exp}(\widehat{\Omega_{i,t}}).
$$

当 $\nu_i\equiv 1$ 时，Level 3 退化为 Level 2。下面省略 residue 下标 $i$。

---

## 2. $\mathbb R^3$ 平动推导

写成一般 stochastic interpolant：

$$
x_t=\mu_t+\sigma_t\varepsilon,
$$

其中

$$
\mu_t=(1-t)x_0+t x_1,
$$

$$
\sigma_t=\bar\sigma_x\sqrt{\nu}\sqrt{t(1-t)}.
$$

因为 $\nu$ 不依赖 $t$，所以

$$
\dot\mu_t=x_1-x_0,
$$

并且

$$
\frac{\dot\sigma_t}{\sigma_t}
=
\frac{1-2t}{2t(1-t)}.
$$

stochastic interpolant 的 conditional velocity 为

$$
u_t^x
=
\dot\mu_t
+
\frac{\dot\sigma_t}{\sigma_t}(x_t-\mu_t).
$$

又因为

$$
x_t-\mu_t=\delta_t^x,
$$

所以

$$
u_t^x
=
x_1-x_0
+
\frac{1-2t}{2t(1-t)}\delta_t^x.
$$

FlowModel 不直接输出 $u_t^x$，而是输出 endpoint prediction $\hat x_{1\mid t}$。这个参数化诱导出的速度为

$$
u_\theta^x
=
\frac{\hat x_{1\mid t}-x_t}{1-t}.
$$

为了让 endpoint parameterization 等价于 SFM velocity matching，需要满足

$$
\frac{x_{1\mid t}^{\star}-x_t}{1-t}
=
u_t^x.
$$

因此

$$
x_{1\mid t}^{\star}
=
x_t+(1-t)u_t^x.
$$

代入 $u_t^x$：

$$
x_{1\mid t}^{\star}
=
x_t
+
(1-t)(x_1-x_0)
+
\frac{1-2t}{2t}\delta_t^x.
$$

由 bridge 定义

$$
x_t=(1-t)x_0+t x_1+\delta_t^x,
$$

可得

$$
x_t+(1-t)(x_1-x_0)
=
x_1+\delta_t^x.
$$

所以

$$
x_{1\mid t}^{\star}
=
x_1+\delta_t^x
+
\frac{1-2t}{2t}\delta_t^x.
$$

合并系数：

$$
1+\frac{1-2t}{2t}
=
\frac{1}{2t}.
$$

最终得到

$$
\boxed{
x_{1\mid t}^{\star}
=
x_1+\frac{\delta_t^x}{2t}
}.
$$

因此，平动部分的 endpoint-form SFM loss 为

$$
\mathcal L_x
=
\mathbb E
\left[
\lambda_x
\frac{
\left\|
\hat x_{1\mid t}
-
\left(
x_1+\frac{\delta_t^x}{2t}
\right)
\right\|^2
}{
s(t)^2
}
\right].
$$

代码中使用的归一化因子为

$$
s(t)=1-\min(t,t_{\mathrm{clip}}).
$$

---

## 3. $SO(3)$ 旋转推导

当前旋转 bridge 不是 intrinsic Brownian bridge on $SO(3)$，而是 tangent-space lifted bridge：

$$
\Omega_t=t\omega+\delta_t^\Omega,
$$

$$
R_t=R_0\operatorname{Exp}(\widehat{\Omega_t}).
$$

在李代数 $\mathfrak{so}(3)\cong\mathbb R^3$ 中，和上一节完全一样，SFM velocity 为

$$
u_t^\Omega
=
\omega
+
\frac{1-2t}{2t(1-t)}\delta_t^\Omega.
$$

严格推到 $SO(3)$ 的 body-frame angular velocity，需要右 Jacobian $J_r$：

$$
a_t^\star
=
J_r(\Omega_t)
\left[
\omega
+
\frac{1-2t}{2t(1-t)}\delta_t^\Omega
\right].
$$

因此，严格的 velocity-form rotation loss 可以写为

$$
\mathcal L_R^{\mathrm{exact}}
=
\mathbb E
\left[
\lambda_R
\left\|
\frac{
\log(R_t^\top \hat R_{1\mid t})^\vee
}{1-t}
-
J_r(\Omega_t)
\left[
\omega
+
\frac{1-2t}{2t(1-t)}\delta_t^\Omega
\right]
\right\|^2
\right].
$$

当前代码采用 endpoint tangent vector：

$$
v_\theta^\Omega
=
\log(R_t^\top \hat R_{1\mid t})^\vee.
$$

因此 exact endpoint target 是

$$
v_{1\mid t}^{\Omega,\star}
=
(1-t)a_t^\star.
$$

为了得到当前实现使用的一阶近似，先令

$$
J_r(\Omega_t)\approx I.
$$

于是

$$
(1-t)u_t^\Omega
=
(1-t)\omega
+
\frac{1-2t}{2t}\delta_t^\Omega.
$$

又由

$$
\Omega_t=t\omega+\delta_t^\Omega
$$

得到

$$
\omega-\Omega_t
=
(1-t)\omega-\delta_t^\Omega.
$$

所以

$$
(1-t)\omega
=
(\omega-\Omega_t)+\delta_t^\Omega.
$$

代回：

$$
(1-t)u_t^\Omega
=
(\omega-\Omega_t)
+
\delta_t^\Omega
+
\frac{1-2t}{2t}\delta_t^\Omega.
$$

合并 $\delta_t^\Omega$ 的系数：

$$
1+\frac{1-2t}{2t}
=
\frac{1}{2t}.
$$

因此

$$
(1-t)u_t^\Omega
=
(\omega-\Omega_t)
+
\frac{\delta_t^\Omega}{2t}.
$$

另一方面，

$$
R_t^\top R_1
=
\operatorname{Exp}(-\widehat{\Omega_t})
\operatorname{Exp}(\widehat{\omega}).
$$

由一阶 BCH 近似：

$$
\log(R_t^\top R_1)^\vee
\approx
\omega-\Omega_t.
$$

所以当前代码使用的 rotation endpoint target 为

$$
\boxed{
v_{1\mid t}^{\Omega,\star}
\approx
\log(R_t^\top R_1)^\vee
+
\frac{\delta_t^\Omega}{2t}
}.
$$

对应的 rotation loss 是

$$
\mathcal L_R
=
\mathbb E
\left[
\lambda_R
\frac{
\left\|
\log(R_t^\top \hat R_{1\mid t})^\vee
-
\left(
\log(R_t^\top R_1)^\vee
+
\frac{\delta_t^\Omega}{2t}
\right)
\right\|^2
}{
s(t)^2
}
\right].
$$

---

## 4. 合并后的主 bridge loss

当前实现对应的 Level 2/3 主 loss 可以写成

$$
\boxed{
\mathcal L_{\mathrm{bridge}}
=
\mathbb E
\left[
\lambda_x
\frac{
\left\|
\hat x_{1\mid t}
-
\left(
x_1+\frac{\delta_t^x}{2t}
\right)
\right\|^2
}{
s(t)^2
}
+
\lambda_R
\frac{
\left\|
\log(R_t^\top \hat R_{1\mid t})^\vee
-
\left(
\log(R_t^\top R_1)^\vee
+
\frac{\delta_t^\Omega}{2t}
\right)
\right\|^2
}{
s(t)^2
}
\right].
}
$$

其中

$$
\delta_t^x
=
x_t-\bigl[(1-t)x_0+t x_1\bigr],
$$

$$
\delta_t^\Omega
=
\Omega_t-t\omega.
$$

在 Level 3 中，$\nu_i$ 只通过 bridge noise 进入：

$$
\delta_{i,t}^x\propto \sqrt{\nu_i},
\qquad
\delta_{i,t}^\Omega\propto \sqrt{\nu_i}.
$$

因此 Level 3 不需要额外改变 loss 的形式。

---

## 5. 与 clean denoising 的区别

如果目标是普通 clean endpoint denoising，则 target 是

$$
x_1,
\qquad
\log(R_t^\top R_1)^\vee.
$$

但这对应的不是当前 stochastic bridge 的 SFM velocity target。

如果目标是学习该 stochastic bridge 对应的 SFM velocity，并且网络输出采用 endpoint parameterization，则 target 必须是

$$
x_1+\frac{\delta_t^x}{2t},
\qquad
\log(R_t^\top R_1)^\vee+\frac{\delta_t^\Omega}{2t}.
$$

这就是代码中

$$
\mathrm{bridge\_noise}\times\frac{1}{2t}
$$

修正项的来源。
