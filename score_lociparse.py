"""
Standalone lociPARSE scoring for generated RNA conformers.

Runs in its own venv (torch 1.12 + numpy 1.22) because lociPARSE pins old
versions that conflict with the main `rna` env. Outputs a CSV of per-cluster
mean pMoL, which evaluate_metrics.py merges back in via `--lociparse-csv`.

Usage:
    /path/to/lociparse/venv/bin/python score_lociparse.py                   # all methods
    /path/to/lociparse/venv/bin/python score_lociparse.py --methods af3     # subset
    /path/to/lociparse/venv/bin/python score_lociparse.py --split test_ensemble
"""

import argparse
import csv
import glob
import json
import logging
import os
import tempfile

from Bio.PDB import MMCIFParser, PDBIO

from lociPARSE import lociparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Mirror the paths from evaluate_metrics.py so the two scripts stay in lockstep.
DATA_DIR = "/projects/u6bk/wanli/ensemble_data_filtered"
INFERENCE_ROOT = "/projects/u6bk/wanli/inference_ours"
SPLIT_JSON = os.path.join(DATA_DIR, "split_cdhit80.json")

_cif_parser = MMCIFParser(QUIET=True)


def _cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    structure = _cif_parser.get_structure("s", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def _extract_pmol(result, tmpdir: str) -> float:
    """Pull pMoL from a lociPARSE result; fall back to the .save() text file."""
    pmol = getattr(result, "pMoL", None)
    if pmol is not None:
        show = getattr(pmol, "show", None)
        if callable(show):
            val = show()
            if val is not None:
                return float(val)
        if isinstance(pmol, (int, float)):
            return float(pmol)
    out = os.path.join(tmpdir, "lp_score.txt")
    result.save(out)
    with open(out) as fh:
        return float(fh.readline().split()[-1])


def score_cluster(scorer, cluster_dir: str) -> float | None:
    """Return mean pMoL over every generated conformer in a cluster dir."""
    files = sorted(
        glob.glob(os.path.join(cluster_dir, "*_conf*.cif"))
        + glob.glob(os.path.join(cluster_dir, "*_conf*.pdb"))
    )
    if not files:
        return None

    scores = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            try:
                if f.endswith(".cif"):
                    pdb_path = os.path.join(tmpdir, os.path.basename(f)[:-4] + ".pdb")
                    _cif_to_pdb(f, pdb_path)
                    result = scorer.score(pdb_path)
                else:
                    result = scorer.score(f)
                scores.append(_extract_pmol(result, tmpdir))
            except Exception as e:
                log.warning(f"lociPARSE failed on {f}: {e}")

    if not scores:
        return None
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="Score generated conformers with lociPARSE")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Method dirs under inference_root (default: all)")
    parser.add_argument("--split", default="all",
                        choices=["all", "test_single", "test_ensemble"],
                        help="Restrict to clusters in this split")
    parser.add_argument("--inference_root", default=INFERENCE_ROOT)
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: {inference_root}/lociparse_{split}.csv)")
    args = parser.parse_args()

    # Restrict to the requested split if any
    allowed_clusters: set[str] | None = None
    if args.split != "all":
        with open(SPLIT_JSON) as f:
            split_data = json.load(f)
        allowed_clusters = set(split_data.get(args.split, []))
        log.info(f"Restricting to {len(allowed_clusters)} clusters in split '{args.split}'")

    if args.methods:
        method_dirs = [os.path.join(args.inference_root, m) for m in args.methods]
    else:
        method_dirs = sorted(
            d for d in glob.glob(os.path.join(args.inference_root, "*"))
            if os.path.isdir(d)
        )
    log.info(f"Methods: {[os.path.basename(d) for d in method_dirs]}")

    log.info("Loading lociPARSE model (one-time cost) ...")
    scorer = lociparse()

    rows: list[dict] = []
    for method_dir in method_dirs:
        if not os.path.isdir(method_dir):
            log.warning(f"Skipping {method_dir}: not a directory")
            continue
        method_name = os.path.basename(method_dir)
        cluster_dirs = sorted(
            d for d in glob.glob(os.path.join(method_dir, "cluster_*"))
            if os.path.isdir(d)
        )
        log.info(f"[{method_name}] {len(cluster_dirs)} clusters")
        for i, cluster_dir in enumerate(cluster_dirs, 1):
            cname = os.path.basename(cluster_dir)
            if allowed_clusters is not None and cname not in allowed_clusters:
                continue
            conf = score_cluster(scorer, cluster_dir)
            if conf is None:
                log.warning(f"[{method_name}/{cname}] no valid structures scored")
                continue
            rows.append({"method": method_name, "cluster": cname, "conf_lociparse": conf})
            log.info(f"[{method_name}] {i}/{len(cluster_dirs)} {cname}: pMoL={conf:.4f}")

    output_path = args.output or os.path.join(
        args.inference_root, f"lociparse_{args.split}.csv"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cluster", "conf_lociparse"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
