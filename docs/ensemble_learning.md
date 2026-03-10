现在的代码是一个 de novo RNA backbone generation 的问题. 这其实和我们的有些区别. 我想要做的是一个 RNA ensemble generation 问题. 想要通过学习 ensemble dataset, 来学习 ensemble 信息. 因此数据输入和验证的逻辑可能要进行修改.
首先，我们先来关注模型。我先陈述一些事实：
1. 现在的模型是一个基于 SE(3) flow matching 的 generative model。
2. 模型输入是加噪后的结构数据（在 `interpolant.py` 中可以看到，通过 `corrupt_batch` 来对输入的 ground truth batch $x_1$ 进行线性插值加噪，得到 noisy batch $x_t$）。
3. 这个代码是一个 do novo RNA backbone generation. 因此在推理时，其输入只需要时长度信息以及采样的噪声即可。模型在任何时候都不需要 sequence 的信息。并且生成的只是骨架，而不包含任何的核苷酸信息。
基于此，我们要对代码进行如下更改：

### 输入数据

1. 整体流程概览
目前的数据流水线设计为 "离线构建 + 直接加载" 模式：
1.  预处理 (Preprocessing): `rna_backbone_design/data/build_datasets.py`
    *   输入: 原始 .pdb 文件。
    *   操作: 通过 `build_datasets.py` 将每个 conformer 的几何特征、cluster 元信息整理成 `.pt` 记录，同时保留 cluster 目录下的 `.npy` embeddings 供运行时读取。
    *   存储: 保存为 `train_conformers.pt`、`val_ensemble_conformers.pt`、`val_single_conformers.pt`、`test_ensemble_conformers.pt`、`test_single_conformers.pt`。
2.  数据集加载 (Dataset Loading): `rna_conformer_dataset.py`
    *   输入: 预处理 `.pt` 记录 + cluster 下的 `.npy` embeddings。
    *   操作: 直接读取离线构建好的几何张量，并在运行时按需加载 embedding。
3.  模型输入 (Model Input): `rna_conformer_datamodule.py`
    *   操作: 将不同长度的数据进行 Batch 打包 (Padding)，并在 `val/test ensemble` 模式下按 cluster 重新组装 ensemble。
    *   输出: 也就是喂给模型的最终 batch 字典。
---
2. 模型接收到的输入 (Output to Model)
当你的模型在 training_step 接收到一个 batch 时，它是一个字典，包含以下核心 Tensor。
假设 B = Batch Size, L = Sequence Length (Padding 后的长度)。
A. 几何特征 (Geometry / Structure)
这是 Flow Matching 模型要预测或以此为条件的核心目标。
*   trans_1: [B, L, 3]
    *   含义: 每个残基的平移向量 (Translation)。
    *   来源: 通常取自 C4' 原子的坐标，代表该残基在空间中的中心位置。
*   rotmats_1: [B, L, 3, 3]
    *   含义: 每个残基的旋转矩阵 (Rotation Matrix)。
    *   来源: 根据 C4', C1', N1/N9 等原子构建的局部刚体坐标系。
*   torsion_angles_sin_cos: [B, L, 8, 2]
    *   含义: 8 个关键扭转角 (Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Chi 等) 的正弦和余弦值。
    *   格式: 最后一维是 2，分别存 [sin(angle), cos(angle)]。使用 sin/cos 是为了避免角度的周期性问题。
*   torsion_angles_mask: [B, L, 8]
    *   含义: 标记哪些扭转角是有效的（例如末端残基可能缺某些角，或者原子丢失导致无法计算）。
B. 序列与掩码 (Sequence & Masks)
*   aatype: [B, L] (Int64)
    *   含义: 氨基酸/核苷酸的类别索引 (0-3 对应 A, U, C, G，或者包含更多 Token)。
*   res_mask: [B, L] (Float32)
    *   含义: 残基掩码。1 表示真实存在的残基，0 表示 Padding 填充的部分。
*   is_na_residue_mask: [B, L] (Float32)
    *   含义: 区分是核酸(1)还是蛋白质(0)。在这个数据集中通常全为 1。
C. 嵌入特征 (Embeddings)
这些是预先计算好的上下文信息（例如来自 ESM 或 RNA-FM 等语言模型）。
*   single_embeds: [B, L, 640] (维度取决于具体的 Embedding 模型)
    *   含义: 每个残基的单体特征向量。
*   pair_embeds: [B, L, L, 128]
    *   含义: 残基对之间的特征向量（Pairwise features），捕捉长程依赖。
---


### 模型改进

1. Input (输入)
模型接收一个包含当前时刻（Timestep）结构信息的字典 input_feats。在训练时，这些输入是通过 Interpolant 对真实结构加噪生成的；在推理时，通过随机采样生成。
*   t: 扩散时间步 (Timestep)，形状 [B, 1]，范围 [0, 1]。
*   res_mask: 残基掩码，形状 [B, N]，标记哪些位置是有效的 RNA 核苷酸。
*   trans_t: 当前时刻 $t$ 的骨架平移向量（Translations），形状 [B, N, 3]。
*   rotmats_t: 当前时刻 $t$ 的骨架旋转矩阵（Rotation Matrices），形状 [B, N, 3, 3]。
*   trans_sc (Optional): Self-conditioning 的平移向量，用于提高模型一致性。
2. Output (输出)
模型直接预测 $t=1$ 时刻的去噪（Clean）结构，即真实的 RNA 骨架。
*   pred_trans: 预测的真实平移向量，形状 [B, N, 3]。
*   pred_rotmats: 预测的真实旋转矩阵，形状 [B, N, 3, 3]。
*   pred_torsions: 预测的扭转角（Torsion Angles），形状 [B, N, 8, 2] (8 个角度的 sin/cos 值)，用于重建全原子结构。
3. Pipeline (网络架构流程)
核心网络 FlowModel (rna_backbone_design/models/flow_model.py) 采用了类似 AlphaFold2 / OpenFold 的架构，但针对 Flow Matching 进行了适配。
1.  特征嵌入 (Embedding):
    *   Node Embedder: 将序列位置（Positional Encoding）和时间步 t 编码为节点特征 [B, N, c_s]。
    *   Edge Embedder: 基于节点特征、当前结构的相对距离（Distograms）和相对位置编码，生成边特征 [B, N, N, c_p]。
2.  主干网络 (Attention Trunk):
    *   由 $L$ 个 IPA Block (Invariant Point Attention) 堆叠而成。
    *   IPA 机制: 在更新节点特征时，显式利用了当前的 3D 几何信息（刚体变换），保证了 SE(3) 不变性。
    *   Transformer Encoder: 在 IPA 之后，通过标准的 Transformer 层处理序列信息。
    *   Backbone Update: 每个 Block 结束时，使用预测的更新量迭代更新当前的刚体位置 (curr_rigids)。这意味着结构是在网络层级间逐步精修的。
3.  预测头 (Heads):
    *   Structure Head: 直接输出主干网络最后一层更新后的刚体 (trans, rotmats)。
    *   Torsion Head: 一个简单的 MLP (torsion_net)，根据最终的节点特征预测 RNA 的扭转角。
4. Function & Mechanism (核心功能与机制)
该模型的设计核心是 Flow Matching（流匹配），特别是针对 SE(3) 流形（刚体变换群）的流匹配。
*   训练 (Flow Matching Training):
    *   插值 (Interpolant): 定义了一个概率路径，将噪声分布（$t=0$，高斯噪声+均匀旋转）平滑变换到数据分布（$t=1$，真实 RNA 结构）。
        *   平移: 使用 Optimal Transport (OT) 路径，即直线插值。
        *   旋转: 使用 SO(3) 上的测地线插值。
    *   损失函数:
        *   SE(3) Loss: 计算预测结构与真实结构在平移和旋转上的差异（Vector Field Loss）。
        *   Auxiliary Loss: 辅助损失，包括全原子坐标损失 (Backbone Atom Loss)、成对距离损失 (Distance Matrix Loss) 和扭转角损失。
*   推理 (Sampling / ODE Solver):
    *   从先验分布（高斯噪声 + 均匀随机旋转）采样 $x_0$。
    *   使用欧拉法（Euler method）求解常微分方程 (ODE)。在每一步 $t$，模型预测目标 $x_1$，利用该预测计算向量场 $v_t = (x_1 - x_t) / (1-t)$，从而推动结构向真实分布演化。
    *   最终通过 pred_trans、pred_rotmats 和 pred_torsions 重建出全原子 PDB 文件。
