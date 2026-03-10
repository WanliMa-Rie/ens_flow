# Gaussian Distribution as Score

RNA 相较于 protein 来说，柔性更大，因此在去噪时，需要更灵活的去噪方式。因此，本工作提出了将 score model 的输出看作是分布，而不是单纯的 $\nabla\log p_t(R_t,t)$，额外的方差可以用于表征柔性和刚性。



## 几何背景

- 流形 $\mathcal{M}$：RNA 的构象空间可以看作是 $SE(3)^N$，其中 $N$ 为氨基酸残基的数目。
- 切空间 $T_x\mathcal{M}$：Diffusion 的 score $s$ 实际上是切空间上的向量。损失也是在切空间中计算。
    - 平移部分 $\in \mathbb{R}^3$：
    - 旋转部分 $\in so(3)$（李代数），其同构与 $\mathbb{R}^3$
    - 因此，对于每一个残基 $n$, 其 score $s_n\in\mathbb{R}^6$。





## 建模目标

传统的 score matching 假设 $p_t(s|x_t)$ 是各向同性的 Gaussian (方差为常数 $\sigma^2I$，在预测时仅仅预测均值 $\mu_\theta(x_t)$). 而本工作则假设 $p_t(s|x_t)$ 是一个局部自适应的多元高斯分布：

$$
p_t(s|x_t) = \mathcal{N}(s|\mu_\theta(x_t), \Sigma_\theta(x_t))
$$
其中：
- $\mu_\theta(x_t)\in\mathbb{R}^{6N}$：预测的去噪方向（Score）。
- $\Sigma_\theta(x_t)\in\mathbb{R}^{6N\times6N}$：预测的局部协方差矩阵（Uncertainty）。

这其中 $N$ 为某个构象氨基酸残基的数目。然而，这样做的一个问题是，协方差矩阵的维度为 $6N\times6N$，这在计算上是不可行的，尤其是后面涉及到求逆运算，复杂度为 $O((6N)^3)$，这在计算上是不可行的。

有两种办法可以来解决：

### 方法一：参数化对角协方差矩阵
假设残基间独立（block diagonal），且每个维度独立。因此 $6N\times6N$ 的协方差矩阵可以参数化为 $6N$ 维的向量，并用模型来预测：
$h\in\mathbb{R}^{6N}$，用于参数化对角协方差 $\Sigma=\text{diag}(\exp(h))$

### 方法二：Cholesky 分解

模型需要三个输出头：
- Score head: $\hat{s}\in\mathbb{R}^{6N}$，作为均值输出，保持不变
- Local anisotropy head: $v_{local}\in\mathbb{R}^{N\times 21}$。对于每个残基 $i$，预测其局部协方差矩阵的 Cholesky 因子 $L_i$（有 $6\times (6+1)/2=21$个参数）。
- Global coupling head: $v_{global}\in\mathbb{R}^{N\times N}$。利用 pair representation 预测残基 $i,j$ 之间的标量耦合系数，通过一些约束（如 SPD，对角元素为 1）构建全局相关性矩阵（correlation matrix） $C_{global}$。

基于上述输出，能够通过 Kronecker 积得到完整的 $3N\times3N$ 的协方差矩阵 $\Sigma$：

$$
\Sigma=L_{local}(C_{global} \otimes I_6)L_{local}^T
$$
其中：
- $L_{local}=\text{blockdiag}(L_1,\dots,L_N)\in\mathbb{R}^{3N\times3N}$ 是非常稀疏的快对角矩阵
- $C_{global}\otimes I_6$ 表示将残基 $i,j$ 之间的标量耦合系数 $C_{ij}$ 扩展到 6 个自由度上.


## Loss function

我们的 loss function 主要由两部分组成：均值 loss 和方差 loss。


### 符号定义

- $\hat{s}\in\mathbb{R}^{B\times N\times 6}$：模型预测的均值


### mean loss

一般的 denoising score matching (DSM) 的 loss function 为：

$$
\mathcal{L}_{DSM}=\mathbb{E}_{x_t,s_t}\left[\left(s_t-\hat{s}_\theta(x_t)\right)^2\right]
$$
即为 $s_t$ 的均方误差。在当前的 context 下，为了引入残基的柔性，我们不再使用欧氏距离 MSE，而是用马氏距离（Mahalanobis distance）形式的 MSE，利用预测出的方差来加权这个单点误差：

$$
\mathcal{L}_{Anchor}=\frac{1}{2}(s^{(k^*)}-\hat{s})^T\Sigma^{-1}(s^{(k^*)}-\hat{s})
$$
物理含义：
- 如果模型预测某处（残基）的方差 $\Sigma$ 很小（刚性），那么 $\hat{s}$ 稍微偏离 $s^{(k^*)}$ 时，会带来很大的惩罚。
- 如果模型预测某处（残基）的方差 $\Sigma$ 很大（柔性），那么 $\hat{s}$ 稍微偏离 $s^{(k^*)}$ 时，会带来很小的惩罚。
-  这样使得模型在刚性区域必须学得很准，而在柔性区域可以有一定的容错性（即产生不同的 ensemble）。




### Ensemble loss

仅仅通过 Anchor loss 我们无法从中得到 $\Sigma$ 的信息，。因此，我们引入了 Ensemble loss，即对整个 cluster 的分布进行极大似然估计，即最小化 Negative Log Likelihood (NLL)：

$$
\begin{aligned}
\mathcal{L}_{dist}&=-\mathbb{E}_{s^{(k)}\sim\mathcal{S}^{GT}}\left[\log p_t(s^{(k)}|x_t)\right]\\
&=-\mathbb{E}_{s^{(k)}\sim\mathcal{S}^{GT}}\left[\log \mathcal{N}(s^{(k)}|\mu_\theta(x_t),\Sigma_\theta(x_t))\right]\\
&=\mathbb{E}_{s^{(k)}\sim\mathcal{S}^{GT}}\left[\frac{1}{2}(s^{(k)}-\hat{s})^T\Sigma_\theta^{-1}(s^{(k)}-\hat{s})+\frac{1}{2}\log\det|\Sigma_\theta|\right]\\
&=\frac{1}{K}\sum_{k=1}^K\left[\frac{1}{2}(s^{(k)}-\hat{s})^T\Sigma_\theta^{-1}(s^{(k)}-\hat{s})+\frac{1}{2}\log\det|\Sigma_\theta|\right]
\end{aligned}
$$

通过预测损失与不同 GT 结构的差值，能够得到哪些残基柔性高（对应大方差），哪些残基刚性高（对应小方差）。

> 这里有一个 idea，能否直接通过不同残基的柔性指标作为监督来进行学习？因为现在学习本质上是在学当前的 score 与 GT 结构的分别。


### 总 loss

$$
\mathcal{L}=\mathcal{L}_{Anchor}(s^{(k^*)},\hat{s},\Sigma)+\lambda_{ens}\mathcal{L}_{dist}(\mathcal{S}^{GT},\hat{s},\Sigma)
$$
不难发现，当 $K=1$ 时，$\mathcal{L}_{dist}=0$，则 $\mathcal{L}=\mathcal{L}_{Anchor}$，即退化为 Anchor loss。


> 为什么这能够区分刚性/柔性？我们可以通过分析 $\Sigma$ 的梯度来理解。考虑 $\mathcal{L}_{dist}$ 对某一个残差的对数方差 $h$ 求导：
> $$
\frac{\partial\mathcal{L}_{dist}}{\partial h}=\frac{1}{2K}\sum_{k=1}^K\left(1 - \frac{(s^{(k)}-\hat{s})^2}{e^h}\right)
> $$
> 令导数为 0，得到最优的方差估计 $\hat{\sigma}^2$：
> $$
\hat{\sigma}^2=\frac{1}{K}\sum_{k=1}^K(s^{(k)}-\hat{s})^2 = e^h
> $$
> 观察到，模型学到的预测方差 $\hat{\Sigma}$，将严格收敛于真实 cluster 内的经验方差。这即对应着刚性区域和柔性区域内结构相差不同导致的经验方差不同，而模型则能够学习到这些不同。



## 最终的 loss 表达式

为了避免对 $\Sigma$ 求逆，两种方案会有不同的解析表达式。

### 方案一：参数化对角协方差矩阵


1. **定义**

- 预测量：$h\in\mathbb{R}^{6N}$，用于参数化对角协方差
- 对角协方差：$\Sigma=\text{diag}(\exp(h))$
- 逆矩阵：$\Sigma^{-1}=\text{diag}(\exp(-h))$
- 矩阵行列式：$\log|\Sigma|=\Sigma \exp(h)=\sum_{n,d}h_{n,d}$

2. **Mahalanobis distance term**

对于任意样本 $k$，其加权平方误差为：
$$
\begin{aligned}
Q^{(k)}&=\frac{1}{2}(s^{(k)}-\hat{s})^T\Sigma^{-1}(s^{(k)}-\hat{s})\\
&=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^6(s^{(k)}_{ij}-\hat{s}_{ij})\cdot[\Sigma^{-1}]_{ij}\cdot(s^{(k)}_{ij}-\hat{s}_{ij})\\
&=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^6\exp(-h_{ij})\cdot(s^{(k)}_{ij}-\hat{s}_{ij})^2\\
\end{aligned}
$$

3. **Log-determinant term**

$$
\begin{aligned}
\log|\Sigma|&=\log\left(\prod_{i=1}^N\prod_{j=1}^6\exp(h_{ij})\right)\\
&=\sum_{i=1}^N\sum_{j=1}^6h_{ij}
\end{aligned}
$$


4. **最终 loss 解析式**

Anchor loss:

$$
\mathcal{L}_{Anchor}=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^6\exp(-h_{ij})\cdot(s^{(k^*)}_{ij}-\hat{s}_{ij})^2
$$

Ensemble loss:

$$
\mathcal{L}_{dist}=\frac{1}{K}\sum_{k=1}^K\left[\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^6\left(\exp(-h_{ij})(s^{(k)}_{ij}-\hat{s}_{ij})^2+h_{ij}\right)\right]
$$


### 方案二：Cholesky 分解
此方法引入了局部各向异性和全局耦合，利用 Kronecker 积结构避免大矩阵求逆。

#### 1. 定义与性质

- **协方差结构**：$\Sigma = L_{local} (C_{global} \otimes I_6) L_{local}^T$。
- **$L_{local}$**：块对角矩阵，对角块为 $L_i$（$6\times 6$ 下三角）。
- **$C_{global}$**：全局相关性矩阵（$N \times N$ SPD）。
- **辅助变量**：令 $U_{global}$ 为 $C_{global}$ 的 Cholesky 因子，即 $C_{global} = U_{global} U_{global}^T$，其中 $U_{global}$ 为 $N \times N$ 下三角矩阵。

#### 2. 关键数学工具 (避免 $O((6N)^3)$ 的核心)

我们需要计算 $\Sigma^{-1}$ 和 $\log \det \Sigma$。

A. 逆矩阵的分解

利用 Kronecker 积的性质 $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$：

$$\begin{aligned} \Sigma^{-1} &= \left[ L_{local} (C_{global} \otimes I_6) L_{local}^T \right]^{-1} \\ &= L_{local}^{-T} (C_{global} \otimes I_6)^{-1} L_{local}^{-1} \\ &= L_{local}^{-T} (C_{global}^{-1} \otimes I_6) L_{local}^{-1} \end{aligned}$$

代入二次型项 $\delta^T \Sigma^{-1} \delta$：

$$\delta^T L_{local}^{-T} (C_{global}^{-1} \otimes I_6) L_{local}^{-1} \delta$$

令中间变量 **$\mathbf{y} = L_{local}^{-1} \delta$**（局部白化），则上式变为：

$$\mathbf{y}^T (C_{global}^{-1} \otimes I_6) \mathbf{y}$$

由于 $C_{global}^{-1} = U_{global}^{-T} U_{global}^{-1}$，上式继续分解：

$$\mathbf{y}^T (U_{global}^{-T} U_{global}^{-1} \otimes I_6) \mathbf{y} = \mathbf{y}^T (U_{global}^{-T} \otimes I_6) (U_{global}^{-1} \otimes I_6) \mathbf{y}$$

令中间变量 **$\mathbf{z} = (U_{global}^{-1} \otimes I_6) \mathbf{y}$**（全局白化），则最终二次型退化为欧氏距离：

$$\mathbf{z}^T \mathbf{z} = \|\mathbf{z}\|^2$$

**B. 行列式的分解**

$$\begin{aligned} \log \det \Sigma &= \log \det (L_{local}) + \log \det (C_{global} \otimes I_6) + \log \det (L_{local}^T) \\ &= 2 \log \det (L_{local}) + \log \left( (\det C_{global})^6 \cdot (\det I_6)^N \right) \\ &= 2 \sum_{i=1}^N \log \det (L_i) + 6 \log \det (C_{global}) \end{aligned}$$

由于 $L_i$ 和 $U_{global}$ 都是下三角矩阵，行列式等于对角元素之积：

$$\log \det \Sigma = 2 \sum_{i=1}^N \sum_{j=1}^6 \log (L_{i})_{jj} + 12 \sum_{i=1}^N \log (U_{global})_{ii}$$

#### 3. 计算步骤详解 (Step-by-Step)

为了得到解析式，我们定义两个高效计算操作：

1. **局部解耦 (Local Uncoupling)**：计算 $\mathbf{y}^{(k)} = L_{local}^{-1} \delta^{(k)}$。
   - 由于 $L_{local}$ 是块对角，这等价于对每个残基 $i$ 求解 $6\times 6$ 的线性方程组：$L_i \cdot \mathbf{y}_i^{(k)} = \delta_i^{(k)}$。
   - 使用 `torch.linalg.solve_triangular` 并行计算。
2. **全局解耦 (Global Uncoupling)**：计算 $\mathbf{z}^{(k)} = (U_{global}^{-1} \otimes I_6) \mathbf{y}^{(k)}$。
   - 将 $\mathbf{y}^{(k)}$ 视为 $N \times 6$ 的矩阵。上述操作等价于在 $N$ 维度上求解线性方程组：$U_{global} \cdot \mathbf{Z}^{(k)} = \mathbf{Y}^{(k)}$。
   - 同样使用 `torch.linalg.solve_triangular`。

#### 4. 最终 Loss 解析式 (方法二)

令 $\mathbf{z}^{(k)}$ 为经过局部和全局两步解耦后的残差向量。

**Anchor Loss**:

$$\mathcal{L}_{Anchor} = \frac{1}{2} \|\mathbf{z}^{(k^*)}\|^2 = \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^6 (\mathbf{z}_{ij}^{(k^*)})^2$$

**Ensemble Loss**:

$$\begin{aligned} \mathcal{L}_{dist} = \frac{1}{K} \sum_{k=1}^K \bigg[ &\frac{1}{2} \|\mathbf{z}^{(k)}\|^2 \\ &+ \sum_{i=1}^N \sum_{j=1}^6 \log (L_{i})_{jj} \quad \text{(局部对数体积)} \\ &+ 6 \sum_{i=1}^N \log (U_{global})_{ii} \quad \text{(全局对数体积)} \bigg] \end{aligned}$$


