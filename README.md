# RNA-FrameFlow

SE(3) flow matching for conditional RNA backbone generation.

当前仓库的训练与推理流程已经围绕“簇目录 + 预处理 `.pt` 文件 + 单体/成对 embedding 条件”重构。本文档只描述当前代码真实支持的工作流，不再保留早期基于原始 RNAsolo 流程的过时说明。

## 当前代码做什么

核心训练入口和职责如下：

- `train_se3_flows.py`: 读取 `configs/config.yaml`，实例化 `FlowModule(cfg)` 和 `RNAConformerDataModule(cfg.data_cfg)`，然后调用 Lightning `Trainer.fit(...)`。
- `inference_se3_flows.py`: 读取 `configs/inference.yaml`，从 checkpoint 所在目录合并训练时保存的 `config.yaml`，按 `test_clusters.json` 过滤测试簇，然后调用 `Trainer.predict(...)` 采样并写出 PDB。
- `rna_backbone_design/data/build_datasets.py`: 从簇目录中的 `structure/*.pdb` 离线构建预处理数据，输出每个 split 一个 `.pt` 文件。
- `rna_backbone_design/data/rna_conformer_dataset.py`: 运行时读取单条 conformer 记录，并从磁盘加载 `*_single.npy` / `*_pair.npy` embedding。
- `rna_backbone_design/data/rna_conformer_datamodule.py`: 构建 train/val/test dataloader，并在 `conformer_collate` 中把 batch pad 到同一长度。
- `rna_backbone_design/models/flow_module.py`: 负责训练 loss、验证指标、采样和 checkpoint 恢复。

## 环境安装

当前 `pyproject.toml` 要求：

- Python `>=3.11,<3.12`
- `uv` 作为推荐环境管理器
- PyTorch 源默认指向 `cu126`

最小安装流程：

```bash
git clone https://github.com/rish-16/rna-backbone-design.git
cd rna-backbone-design

pip install uv
uv venv
uv sync
```

如果你的机器不是 CUDA 12.6，需要先按你的环境调整 `pyproject.toml` 中的 `tool.uv.sources`。

训练默认使用 Weights & Biases 记录实验；若不想联网记录，可把 `configs/config.yaml` 中的 `experiment.wandb.mode` 改成 `offline`。

## 数据目录约定

当前代码假设 `data_cfg.data_dir` 指向按 cluster 组织的数据根目录。每个 cluster 至少包含：

```text
<data_dir>/
  cluster_xxx/
    structure/
      *.pdb
    embedding/
      *_single.npy
      *_pair.npy
```

其中：

- `structure/*.pdb` 用于离线构建几何特征。
- `embedding/*_single.npy` 和 `embedding/*_pair.npy` 会在训练/推理时动态加载。
- 当前模型前向依赖 `single_embedding` 和 `pair_embedding`，这两类条件输入不能省略。

默认配置位于 `configs/config.yaml`：

```yaml
data_cfg:
  data_dir: /projects/u6bk/wanli/rna_ensemble_data
  preprocessed_dir: ${data_cfg.data_dir}/preprocessed_data
```

## 离线预处理

`build_datasets.py` 会直接从 cluster 下的 PDB 文件生成五个预处理文件：

- `train_conformers.pt`
- `val_ensemble_conformers.pt`
- `val_single_conformers.pt`
- `test_ensemble_conformers.pt`
- `test_single_conformers.pt`

运行示例：

```bash
uv run -m rna_backbone_design.data.build_datasets \
  --data_dir /projects/u6bk/wanli/rna_ensemble_data \
  --output_dir /projects/u6bk/wanli/rna_ensemble_data/preprocessed_data \
  --cluster_split /projects/u6bk/wanli/rna_ensemble_data/split_cdhit80.json \
  --compute_rmsd
```

如果不传 `--cluster_split`，脚本会默认使用 `<data_dir>/split_cdhit80.json`，必要时基于 CD-HIT 重新构建 split manifest。

### `.pt` 文件内容

每个 `.pt` 文件都是 `List[Dict]`，每个元素对应一个 conformer。当前记录至少包含：

- `aatype`: `[L]`
- `trans_1`: `[L, 3]`
- `rotmats_1`: `[L, 3, 3]`
- `torsion_angles_sin_cos`: `[L, 8, 2]`
- `torsion_angles_mask`: `[L, 8]`
- `res_mask`: `[L]`
- `is_na_residue_mask`: `[L]`
- `c4_coords`: `[L, 3]`
- `bb_coords`: `[L, 3, 3]`
- 元数据：`conformer_name`、`cluster_name`、`seq_len`、`embedding_dir`、`cluster_conformer_names`、`cluster_size`
- 训练集可选字段：`rmsd_matrix`、`rmsd_names`

embedding 不会写进 `.pt`，而是在 `RNAConformerDataset` 中从 `embedding_dir` 读取。

## 运行时 batch 契约

`conformer_collate` 会把样本 pad 到 batch 内最大长度，生成的 clean batch 关键字段为：

- `aatype`: `[B, L]`
- `res_mask`: `[B, L]`
- `trans_1`: `[B, L, 3]`
- `rotmats_1`: `[B, L, 3, 3]`
- `torsion_angles_sin_cos`: `[B, L, 8, 2]`
- `single_embedding`: `[B, L, C_single]`
- `pair_embedding`: `[B, L, L, C_pair]`
- `seq_len`: `[B]`

默认配置下：

- `C_single = 384`
- `C_pair = 128`

真正送入 `FlowModel.forward(...)` 之前，`Interpolant.corrupt_batch(...)` 还会补充：

- `t`
- `trans_t`: `[B, L, 3]`
- `rotmats_t`: `[B, L, 3, 3]`
- 可选 `trans_sc`: `[B, L, 3]`

## 训练

训练命令：

```bash
uv run train_se3_flows.py
```

默认行为：

- 使用 `configs/config.yaml`。
- 实验配置会被保存到 `experiment.checkpointer.dirpath/config.yaml`。
- checkpoint 路径默认是：

```yaml
experiment:
  checkpointer:
    dirpath: /projects/u6bk/wanli/ckpt_flow/${experiment.wandb.project}/${experiment.wandb.name}/
```

训练前至少检查这些配置项：

- `data_cfg.data_dir`: cluster 数据根目录
- `data_cfg.preprocessed_dir`: 五个 `.pt` 文件所在目录
- `experiment.wandb.project` / `experiment.wandb.name`
- `experiment.trainer.accelerator` / `strategy`
- `experiment.checkpointer.dirpath`

当前默认配置还启用了：

- `val_ensemble_as_cluster: true`
- `test_ensemble_as_cluster: true`
- `experiment.ensemble_metrics.enabled: true`

这意味着 ensemble 验证/测试会按 cluster 聚合，而不是把每个 conformer 独立当作一个样本。

## 推理

推理命令：

```bash
uv run inference_se3_flows.py
```

`configs/inference.yaml` 中最重要的字段：

```yaml
inference:
  ckpt_path: /projects/u6bk/wanli/checkpoint/rnafm_ckpt/ensemble_flow_matching/last.ckpt
  output_dir: /projects/u6bk/wanli/inference/rnafm_ensemble
  test_clusters_json: /projects/u6bk/wanli/inference/test_clusters.json
  test_cluster_split: test_ensemble
  num_gpus: 4
```

当前推理逻辑有几个关键点：

- 如果 `inference.ckpt_path` 指向目录，该目录下必须且只能有一个 `.ckpt` 文件。
- 推理会优先读取 checkpoint 同目录下的 `config.yaml`；如果不存在，则尝试 `config_flashipa.yaml`。
- checkpoint 配置会与 `configs/inference.yaml` 合并，且 inference 配置优先生效。
- `test_clusters_json` 中支持的 split 名称是：`all`、`both`、`test_single`、`single`、`test_ensemble`、`ensemble`。
- 推理 dataloader 固定 `batch_size=1`，输出会写到 `inference.output_dir` 下。

`test_clusters.json` 预期至少包含如下键中的一个或多个：

```json
{
  "test_single": ["cluster_..."],
  "test_ensemble": ["cluster_..."]
}
```

## 当前目录流转

一个最常见的数据流如下：

```text
cluster directories
  -> build_datasets.py
  -> preprocessed_data/*.pt
  -> train_se3_flows.py
  -> checkpoint dir/config.yaml + *.ckpt
  -> inference_se3_flows.py
  -> sampled PDBs under inference.output_dir
```

## 常见约束

- 当前训练和推理都默认依赖 `single_embedding` 与 `pair_embedding`。
- 如果你修改数据 shape，需要联动更新：`build_datasets.py` -> `RNAConformerDataset` -> `conformer_collate` -> `Interpolant` / `FlowModule`。
- 验证和测试阶段会过滤名称形如 `msa_group_*` / `msa_*` 的 conformer。
- `train` split 在 `posterior_nearest_p > 0` 时会额外加载邻居 conformer 的几何信息。

## 论文与引用

论文信息：

- arXiv: <https://arxiv.org/abs/2406.13839>
- TMLR: <https://openreview.net/forum?id=wOc1Yx5s09>

```bibtex
@article{anand2025rnaframeflow,
  title={{RNA}-FrameFlow: Flow Matching for de novo 3D {RNA} Backbone Design},
  author={Rishabh Anand and Chaitanya K. Joshi and Alex Morehead and Arian Rokkum Jamasb and Charles Harris and Simon V Mathis and Kieran Didi and Rex Ying and Bryan Hooi and Pietro Lio},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=wOc1Yx5s09},
}
```
