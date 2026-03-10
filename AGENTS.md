# Repository Guidelines

## Working Contract

This repo is research code. Do not optimize for generic contributor guidance, generic abstractions, or defensive scaffolding. If the user request is unclear, stop and clarify the objective and success criterion. If the goal is clear, implement the shortest correct path. Always make tensor inputs, outputs, and shapes explicit, and verify them against code and configs before editing. For routine verification, do not default to smoke tests; prefer compile-level syntax checks such as `python -m compileall` unless the user explicitly asks for runtime tests.

## Main Files

- `train_se3_flows.py`: training entrypoint. Instantiates `FlowModule(cfg)` and `RNAConformerDataModule(cfg.data_cfg)`, then runs Lightning `Trainer.fit`.
- `inference_se3_flows.py`: inference entrypoint. Merges checkpoint config with `configs/inference.yaml`, filters clusters from `test_clusters.json`, and calls `FlowModule.predict_step`.
- `rna_backbone_design/data/build_datasets.py`: offline preprocessing from raw cluster folders into per-split `.pt` files.
- `rna_backbone_design/data/rna_conformer_dataset.py`: runtime dataset. Loads one conformer record, crops it to valid length, and loads embeddings from disk.
- `rna_backbone_design/data/rna_conformer_datamodule.py`: split setup and `conformer_collate`, which pads samples to the max length in the batch.
- `rna_backbone_design/models/flow_module.py`: training loss, validation logic, inference sampling, and PDB writing.
- `rna_backbone_design/models/flow_model.py`: actual network. Current forward path requires `single_embedding` and `pair_embedding`.

## Raw Data And Preprocessed Data

`data_cfg.data_dir` is cluster-based. Each cluster is expected to contain:

- `cluster_xxx/structure/*.pdb`
- `cluster_xxx/embedding/*_single.npy`
- `cluster_xxx/embedding/*_pair.npy`

`build_datasets.py` writes:

- `train_conformers.pt`
- `val_ensemble_conformers.pt`
- `val_single_conformers.pt`
- `test_ensemble_conformers.pt`
- `test_single_conformers.pt`

Each `.pt` file is `List[Dict]`, one dict per conformer. Current record schema:

- `aatype`: `[L]`
- `trans_1`: `[L, 3]`
- `rotmats_1`: `[L, 3, 3]`
- `torsion_angles_sin_cos`: `[L, 8, 2]`
- `torsion_angles_mask`: `[L, 8]`
- `res_mask`: `[L]`
- `is_na_residue_mask`: `[L]`
- `c4_coords`: `[L, 3]`
- `bb_coords`: `[L, 3, 3]`
- metadata: `conformer_name`, `cluster_name`, `seq_len`, `embedding_dir`, `cluster_conformer_names`, `cluster_size`
- optional train-only fields: `rmsd_matrix`, `rmsd_names`

## Batch And Model Input Contract

`conformer_collate` produces clean batches with:

- `aatype`: `[B, L]`
- `res_mask`: `[B, L]`
- `trans_1`: `[B, L, 3]`
- `rotmats_1`: `[B, L, 3, 3]`
- `torsion_angles_sin_cos`: `[B, L, 8, 2]`
- `single_embedding`: `[B, L, C_single]`
- `pair_embedding`: `[B, L, L, C_pair]`
- `seq_len`: `[B]`
- metadata such as `conformer_name`, `cluster_name`, `pdb_name`, `cluster_id`
- ensemble batches may also carry `gt_bb_ensemble` and `gt_c4_ensemble`

With the default config, `C_single=384` and `C_pair=128`.

`FlowModel.forward` does not consume this clean batch directly. `Interpolant.corrupt_batch(...)` adds the actual diffusion inputs:

- `t`: timestep tensor, broadcastable over residues
- `trans_t`: `[B, L, 3]`
- `rotmats_t`: `[B, L, 3, 3]`
- optional `trans_sc`: `[B, L, 3]`

The effective model inputs are `res_mask`, `t`, `trans_t`, `rotmats_t`, `single_embedding`, and `pair_embedding`.

## Important Assumptions

- Current training and inference paths assume single/pair conditioning. `predict_step` slices `batch["single_embedding"]` and `batch["pair_embedding"]` directly.
- If you change shapes, update the full chain together: `build_datasets.py` -> `RNAConformerDataset` -> `conformer_collate` -> `Interpolant`/`FlowModule` -> tests.
- Prefer explicit assertions or `ValueError` on violated invariants. Do not hide shape uncertainty inside wide `if/else` branches.
