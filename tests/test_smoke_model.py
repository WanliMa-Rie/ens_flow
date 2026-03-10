import torch
from omegaconf import OmegaConf

from rna_backbone_design.models.flow_module import FlowModule


def _make_cfg():
    return OmegaConf.create(
        {
            "data_cfg": {
                "data_dir": "data_ensemble/rna_ensemble_data",
                "batch_size": 2,
                "num_workers": 0,
                "max_len": 64,
            },
            "interpolant": {
                "min_t": 1e-2,
                "rots": {
                    "train_schedule": "linear",
                    "sample_schedule": "linear",
                    "exp_rate": 10,
                },
                "trans": {
                    "train_schedule": "linear",
                    "sample_schedule": "linear",
                },
                "sampling": {"num_timesteps": 5},
                "self_condition": True,
            },
            "model": {
                "use_flashipa": False,
                "mode": "orig_2d_bias",
                "node_embed_size": 32,
                "edge_embed_size": 16,
                "symmetric": False,
                "node_features": {
                    "c_s": 32,
                    "c_pos_emb": 8,
                    "c_timestep_emb": 8,
                    "c_single_in": 16,
                    "embed_diffuse_mask": False,
                    "max_num_res": 256,
                    "timestep_int": 1000,
                },
                "edge_features": {
                    "single_bias_transition_n": 2,
                    "c_s": 32,
                    "c_p": 16,
                    "c_pair_in": 8,
                    "relpos_k": 64,
                    "use_rbf": True,
                    "num_rbf": 32,
                    "feat_dim": 8,
                    "num_bins": 10,
                    "self_condition": True,
                },
                "ipa": {
                    "c_s": 32,
                    "c_z": 16,
                    "c_hidden": 16,
                    "no_heads": 2,
                    "no_qk_points": 2,
                    "no_v_points": 2,
                    "seq_tfmr_num_heads": 2,
                    "seq_tfmr_num_layers": 1,
                    "num_blocks": 1,
                },
            },
            "experiment": {
                "checkpointer": {"dirpath": ".cache_test/"},
                "training": {
                    "min_plddt_mask": None,
                    "loss": "se3_vf_loss",
                    "bb_atom_scale": 0.1,
                    "trans_scale": 0.1,
                    "translation_loss_weight": 2.0,
                    "t_normalize_clip": 0.9,
                    "rotation_loss_weights": 1.0,
                    "aux_loss_weight": 1.0,
                    "aux_loss_t_pass": 0.25,
                    "tors_loss_scale": 1.0,
                    "num_non_frame_atoms": 0,
                },
                "optimizer": {"lr": 1e-4},
                "batch_ot": {
                    "enabled": False,
                    "cost": "kabsch",
                    "noise_per_sample": 1,
                    "permute": False,
                },
            },
        }
    )


def _make_batch(B: int = 2, L: int = 8):
    res_mask = torch.ones(B, L, dtype=torch.float32)
    res_mask[1, -2:] = 0.0

    rotmats_1 = torch.eye(3, dtype=torch.float32)[None, None].repeat(B, L, 1, 1)
    trans_1 = torch.randn(B, L, 3, dtype=torch.float32)

    tors = torch.randn(B, L, 8, 2, dtype=torch.float32)
    tors = torch.nn.functional.normalize(tors, dim=-1)
    tors_mask = torch.ones(B, L, 8, dtype=torch.float32)

    single_embedding = torch.randn(B, L, 16, dtype=torch.float32)
    pair_embedding = torch.randn(B, L, L, 8, dtype=torch.float32)

    return {
        "aatype": torch.zeros(B, L, dtype=torch.long),
        "res_mask": res_mask,
        "is_na_residue_mask": res_mask > 0.5,
        "trans_1": trans_1,
        "rotmats_1": rotmats_1,
        "torsion_angles_sin_cos": tors,
        "torsion_angles_mask": tors_mask,
        "single_embedding": single_embedding,
        "pair_embedding": pair_embedding,
        "pdb_name": ["smoke_a", "smoke_b"],
    }


def test_flowmodule_model_step_smoke():
    cfg = _make_cfg()
    module = FlowModule(cfg)
    batch = _make_batch()
    module.interpolant.set_device(batch["res_mask"].device)
    noisy_batch = module.interpolant.corrupt_batch(batch)
    losses = module.model_step(noisy_batch)
    assert "se3_vf_loss" in losses
    assert "auxiliary_loss" in losses
    for v in losses.values():
        assert torch.isfinite(v).all()
