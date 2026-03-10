from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rna_backbone_design.data import utils as du
from rna_backbone_design.data import nucleotide_constants as nc


@dataclass(frozen=True)
class CDHitSplitConfig:
    identity_threshold: float = 0.8
    word_length: int = 5
    val_fraction: float = 0.1
    seed: int = 123
    cdhit_bin: str = "cd-hit-est"
    threads: int = 0
    memory_mb: int = 0
    cache_filename: str = "split_cdhit80.json"


def _stable_hash_percent(text: str, seed: int) -> int:
    h = hashlib.md5(f"{seed}:{text}".encode()).hexdigest()
    return int(h, 16) % 100


def _cluster_num_structures(cluster_dir: pathlib.Path) -> int:
    feature_dir = cluster_dir / "features"
    return len(list(feature_dir.glob("*.pkl")))


def collect_valid_cluster_dirs(data_dir: pathlib.Path) -> List[pathlib.Path]:
    potential_clusters = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("cluster_")]
    )
    valid: List[pathlib.Path] = []
    for cluster_dir in potential_clusters:
        feature_dir = cluster_dir / "features"
        embedding_dir = cluster_dir / "embedding"
        if not feature_dir.is_dir() or not embedding_dir.is_dir():
            continue
        pkl_files = list(feature_dir.glob("*.pkl"))
        single_files = list(embedding_dir.glob("*_single.npy"))
        pair_files = list(embedding_dir.glob("*_pair.npy"))
        if pkl_files and single_files and pair_files:
            valid.append(cluster_dir)
    return valid


def cluster_dir_to_sequence(cluster_dir: pathlib.Path) -> str:
    feature_dir = cluster_dir / "features"
    pkl_files = sorted(list(feature_dir.glob("*.pkl")))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {feature_dir}")
    raw_feats = du.read_pkl(str(pkl_files[0]), verbose=False)
    raw_feats = du.parse_complex_feats(raw_feats)
    aatype = raw_feats.get("aatype", None)
    if aatype is None:
        raise ValueError(f"Missing 'aatype' in {pkl_files[0]}")
    aatype = np.asarray(aatype)

    modeled_idx = raw_feats.get("modeled_idx", None)
    if modeled_idx is not None:
        modeled_idx = np.asarray(modeled_idx)
        if modeled_idx.size > 0:
            min_idx = int(np.min(modeled_idx))
            max_idx = int(np.max(modeled_idx))
            aatype = aatype[min_idx : (max_idx + 1)]

    aatype = np.clip(aatype.astype(np.int64), 0, len(nc.restypes_with_x) - 1)
    seq = nc.restype_to_str_sequence(aatype)
    return seq.replace("-", "X")


def _write_fasta(records: Sequence[Tuple[str, str]], path: pathlib.Path) -> None:
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def _run_cdhit_est(
    in_fasta: pathlib.Path,
    out_fasta: pathlib.Path,
    cfg: CDHitSplitConfig,
) -> pathlib.Path:
    cmd = [
        cfg.cdhit_bin,
        "-i",
        str(in_fasta),
        "-o",
        str(out_fasta),
        "-c",
        str(cfg.identity_threshold),
        "-n",
        str(cfg.word_length),
        "-d",
        "0",
        "-M",
        str(cfg.memory_mb),
        "-T",
        str(cfg.threads),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Unable to run CD-HIT binary '{cfg.cdhit_bin}'. Install cd-hit and/or set data_cfg.cdhit_bin."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CD-HIT failed with exit code {e.returncode}.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    return pathlib.Path(str(out_fasta) + ".clstr")


def _parse_cdhit_clstr(clstr_path: pathlib.Path) -> List[List[str]]:
    clusters: List[List[str]] = []
    cur: List[str] = []
    with open(clstr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if cur:
                    clusters.append(cur)
                cur = []
                continue
            start = line.find(">")
            end = line.find("...", start + 1)
            if start == -1 or end == -1:
                continue
            name = line[start + 1 : end]
            cur.append(name)
    if cur:
        clusters.append(cur)
    return clusters


def load_or_build_split_manifest(
    data_dir: pathlib.Path,
    cfg: CDHitSplitConfig,
) -> Dict[str, List[str]]:
    cache_path = data_dir / cfg.cache_filename
    valid_dirs = collect_valid_cluster_dirs(data_dir)
    valid_names = sorted([d.name for d in valid_dirs])

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            meta = cached.get("_meta", {})
            cached_cfg = meta.get("cfg", {})
            cached_names = meta.get("cluster_names", [])
            if (
                float(cached_cfg.get("identity_threshold", -1)) == cfg.identity_threshold
                and int(cached_cfg.get("seed", -1)) == cfg.seed
                and float(cached_cfg.get("val_fraction", -1)) == cfg.val_fraction
                and cached_names == valid_names
                and "train" in cached
                and "val_ensemble" in cached
                and "val_single" in cached
            ):
                return {
                    "train": list(cached["train"]),
                    "val_ensemble": list(cached["val_ensemble"]),
                    "val_single": list(cached["val_single"]),
                }
        except Exception:
            pass

    records: List[Tuple[str, str]] = []
    for cluster_dir in valid_dirs:
        seq = cluster_dir_to_sequence(cluster_dir)
        if len(seq) == 0:
            continue
        records.append((cluster_dir.name, seq))

    if not records:
        raise RuntimeError(f"No valid cluster sequences found under {data_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = pathlib.Path(tmpdir)
        in_fasta = tmpdir_p / "clusters.fasta"
        out_fasta = tmpdir_p / "clusters_cdhit.fasta"
        _write_fasta(records, in_fasta)
        clstr_path = _run_cdhit_est(in_fasta, out_fasta, cfg)
        groups = _parse_cdhit_clstr(clstr_path)

    name_to_group: Dict[str, int] = {}
    for gi, names in enumerate(groups):
        for n in names:
            name_to_group[n] = gi

    group_ids = sorted(set(name_to_group.values()))
    group_key = {gid: f"cdhit_cluster_{gid}" for gid in group_ids}

    val_groups: List[int] = []
    train_groups: List[int] = []
    val_percent = int(round(cfg.val_fraction * 100))
    for gid in group_ids:
        p = _stable_hash_percent(group_key[gid], cfg.seed)
        if p < val_percent:
            val_groups.append(gid)
        else:
            train_groups.append(gid)

    name_to_dir = {d.name: d for d in valid_dirs}

    def _val_counts(groups_in_val: Iterable[int]) -> Tuple[int, int]:
        ens = 0
        single = 0
        for gid in groups_in_val:
            for n, g in name_to_group.items():
                if g != gid:
                    continue
                k = _cluster_num_structures(name_to_dir[n])
                if k >= 5:
                    ens += 1
                else:
                    single += 1
        return ens, single

    val_ens, val_single = _val_counts(val_groups)

    if val_ens == 0 or val_single == 0:
        candidates = []
        for gid in train_groups:
            ens, single = _val_counts([gid])
            candidates.append((gid, ens, single, _stable_hash_percent(group_key[gid], cfg.seed)))
        candidates.sort(key=lambda x: x[3])

        if val_ens == 0:
            for gid, ens, single, _ in candidates:
                if ens > 0:
                    val_groups.append(gid)
                    train_groups.remove(gid)
                    break
        if val_single == 0:
            for gid, ens, single, _ in candidates:
                if gid in val_groups:
                    continue
                if single > 0:
                    val_groups.append(gid)
                    train_groups.remove(gid)
                    break

    train_names: List[str] = []
    val_ensemble_names: List[str] = []
    val_single_names: List[str] = []

    for name in valid_names:
        gid = name_to_group.get(name, None)
        if gid is None:
            continue
        if gid in train_groups:
            train_names.append(name)
        else:
            k = _cluster_num_structures(name_to_dir[name])
            if k >= 5:
                val_ensemble_names.append(name)
            else:
                val_single_names.append(name)

    manifest = {
        "train": train_names,
        "val_ensemble": val_ensemble_names,
        "val_single": val_single_names,
        "_meta": {
            "cfg": {
                "identity_threshold": cfg.identity_threshold,
                "word_length": cfg.word_length,
                "val_fraction": cfg.val_fraction,
                "seed": cfg.seed,
                "cdhit_bin": cfg.cdhit_bin,
            },
            "cluster_names": valid_names,
            "cwd": os.getcwd(),
        },
    }
    cache_path.write_text(json.dumps(manifest, indent=2))
    return {
        "train": train_names,
        "val_ensemble": val_ensemble_names,
        "val_single": val_single_names,
    }

