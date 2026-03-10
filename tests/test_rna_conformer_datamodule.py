import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rna_backbone_design.data.rna_conformer_datamodule import RNAConformerDataModule


def _write_pairformer_embeddings(data_dir: Path, cluster_name: str, seq_len: int) -> None:
    embedding_dir = data_dir / cluster_name / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        embedding_dir / f"{cluster_name}_single.npy",
        np.ones((seq_len, 4), dtype=np.float32),
    )
    np.save(
        embedding_dir / f"{cluster_name}_pair.npy",
        np.ones((seq_len, seq_len, 2), dtype=np.float32),
    )


def _make_record(
    cluster_name: str,
    conformer_name: str,
    seq_len: int,
    cluster_conformer_names: list[str],
) -> dict:
    rotmats = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(seq_len, 1, 1)
    return {
        "conformer_name": conformer_name,
        "cluster_name": cluster_name,
        "seq_len": seq_len,
        "embedding_dir": f"{cluster_name}/embedding",
        "aatype": torch.zeros(seq_len, dtype=torch.long),
        "trans_1": torch.zeros(seq_len, 3, dtype=torch.float32),
        "rotmats_1": rotmats,
        "torsion_angles_sin_cos": torch.zeros(seq_len, 8, 2, dtype=torch.float32),
        "torsion_angles_mask": torch.ones(seq_len, 8, dtype=torch.float32),
        "res_mask": torch.ones(seq_len, dtype=torch.float32),
        "is_na_residue_mask": torch.ones(seq_len, dtype=torch.bool),
        "c4_coords": torch.zeros(seq_len, 3, dtype=torch.float32),
        "bb_coords": torch.zeros(seq_len, 3, 3, dtype=torch.float32),
        "cluster_conformer_names": cluster_conformer_names,
        "cluster_size": len(cluster_conformer_names),
    }


def test_rna_conformer_datamodule_loads_preprocessed_pt_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "rna_data"
    preprocessed_dir = data_dir / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    _write_pairformer_embeddings(data_dir, "cluster_train_1", seq_len=4)
    _write_pairformer_embeddings(data_dir, "cluster_val_ensemble_1", seq_len=3)
    _write_pairformer_embeddings(data_dir, "cluster_val_single_1", seq_len=5)

    torch.save(
        [
            _make_record(
                "cluster_train_1",
                "train_conf_1",
                seq_len=4,
                cluster_conformer_names=["train_conf_1"],
            )
        ],
        preprocessed_dir / "train_conformers.pt",
    )
    torch.save(
        [
            _make_record(
                "cluster_val_ensemble_1",
                "val_ens_conf_1",
                seq_len=3,
                cluster_conformer_names=["val_ens_conf_1", "val_ens_conf_2"],
            ),
            _make_record(
                "cluster_val_ensemble_1",
                "val_ens_conf_2",
                seq_len=3,
                cluster_conformer_names=["val_ens_conf_1", "val_ens_conf_2"],
            ),
        ],
        preprocessed_dir / "val_ensemble_conformers.pt",
    )
    torch.save(
        [
            _make_record(
                "cluster_val_single_1",
                "val_single_conf_1",
                seq_len=5,
                cluster_conformer_names=["val_single_conf_1"],
            )
        ],
        preprocessed_dir / "val_single_conformers.pt",
    )

    cfg = OmegaConf.create(
        {
            "data_dir": str(data_dir),
            "preprocessed_dir": str(preprocessed_dir),
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "posterior_nearest_p": 0,
            "embedding_cache_size": 2,
            "val_ensemble_as_cluster": True,
            "test_ensemble_as_cluster": True,
        }
    )

    datamodule = RNAConformerDataModule(cfg)
    datamodule.setup("fit")

    assert datamodule.embedding_dim == 4

    train_batch = next(iter(datamodule.train_dataloader()))
    assert train_batch["single_embedding"].shape == (1, 4, 4)
    assert train_batch["pair_embedding"].shape == (1, 4, 4, 2)
    assert train_batch["pdb_name"] == ["train_conf_1"]

    val_ensemble_batch = next(iter(datamodule.val_dataloader()[0]))
    assert val_ensemble_batch["cluster_name"] == ["cluster_val_ensemble_1"]
    assert val_ensemble_batch["gt_c4_ensemble"][0].shape == (2, 3, 3)

    split_path = datamodule.export_split_indices(str(tmp_path / "splits"))
    split_payload = json.loads(Path(split_path).read_text())
    assert split_payload["train"]["cluster_names"] == ["cluster_train_1"]
    assert split_payload["val_ensemble"]["cluster_names"] == ["cluster_val_ensemble_1"]
    assert split_payload["val_single"]["cluster_names"] == ["cluster_val_single_1"]
