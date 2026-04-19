"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import torch
import time
import os
import glob
import shutil
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from rna_backbone_design.analysis import metrics
from rna_backbone_design.analysis import utils as au
from rna_backbone_design.models.flow_model import FlowModel
from rna_backbone_design.models.flexibility_net import FlexibilityNet
from rna_backbone_design.models import utils as mu
from rna_backbone_design.data.interpolant import Interpolant
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import all_atom as rna_all_atom
from rna_backbone_design.data import so3_utils
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.analysis import utils as au
from pytorch_lightning.loggers.wandb import WandbLogger


class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._validation = self._exp_cfg.validation
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        self.model = FlowModel(cfg.model)

        self._bridge_cfg = getattr(cfg, 'stochastic_bridge', None)
        self._level = int(self._bridge_cfg.level) if self._bridge_cfg is not None else 1

        self.interpolant = Interpolant(cfg.interpolant, bridge_cfg=self._bridge_cfg)

        self.flexibility_net = None
        self._bfactor_cfg = None
        if self._level >= 3:
            flex_cfg = self._bridge_cfg.flexibility
            self.flexibility_net = FlexibilityNet(
                c_single_in=int(cfg.model.node_features.c_single_in),
                c_pair_in=int(cfg.model.edge_features.c_pair_in),
                hidden_dim=int(flex_cfg.hidden_dim),
                n_heads=int(flex_cfg.n_heads),
            )
            self._bfactor_cfg = self._bridge_cfg.bfactor

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        # Inference-time orchestration attrs. Populated externally by
        # `inference_se3_flows.py` before `predict_step` is called. Initialized
        # to `None` here so that an early/accidental `predict_step` invocation
        # fails with a clear AttributeError-equivalent rather than a silent
        # ``has no attribute`` crash.
        self._infer_cfg = None
        self._output_dir = None

        self.validation_epoch_metrics_by_loader = {"ensemble": [], "single": []}
        self.validation_epoch_samples = []
        self.save_hyperparameters()

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric("epoch")
            self.logger.experiment.define_metric("train/*", step_metric="epoch")
            self.logger.experiment.define_metric("valid/*", step_metric="epoch")

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self._log_scalar(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # epoch_time = (time.time() - self._epoch_start_time) / 60.0
        # self.log(
        #     "train/epoch_time_minutes",
        #     epoch_time,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        # )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch):
        """
        Given a batch of corrupted Frame objects, predicts the vector field, and calculate the loss between the predicted one and the ground truth.
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects

        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones.
            Loss form is identical across Level 1 and Level 2 — the only difference is how
            `noisy_batch["rotmats_t"]` / `noisy_batch["trans_t"]` are produced upstream in
            `Interpolant.corrupt_batch`.

        Returns:
            batch_losses: dict of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"]
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        num_batch, num_res = loss_mask.shape

        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index

        if training_cfg.num_non_frame_atoms == 0:
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']
        elif training_cfg.num_non_frame_atoms == 3:
            bb_filtered_atom_idx = [2, 3, 6] + [
                0,
                7,
                9,
            ]  # [C3', C4', O4'] + [C1', O3', P]
        elif training_cfg.num_non_frame_atoms == 7:
            bb_filtered_atom_idx = [2, 3, 6] + [
                0,
                4,
                7,
                9,
                10,
                11,
                12,
            ]  # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
        else:
            # NOTE: default is the original frame
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']

        n_merged_atoms = len(bb_filtered_atom_idx)

        # Ground truth labels
        gt_trans_1 = noisy_batch["trans_1"]
        gt_rotmats_1 = noisy_batch["rotmats_1"]
        gt_torsions_1 = noisy_batch["torsion_angles_sin_cos"][
            :, :, torsions_start_index:torsions_end_index, :
        ].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch["rotmats_t"]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms = rna_all_atom.to_atom37_rna(
            gt_trans_1,
            gt_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=gt_torsions_1,
        )
        gt_bb_atoms = gt_bb_atoms[:, :, bb_filtered_atom_idx]

        # Timestep used for normalization.
        t = noisy_batch["t"]
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]
        pred_rotmats_1 = model_output["pred_rotmats"]
        pred_torsions_1 = model_output["pred_torsions"].reshape(
            num_batch, num_res, num_torsions * 2
        )
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = rna_all_atom.to_atom37_rna(
            pred_trans_1,
            pred_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=pred_torsions_1,
        )
        pred_bb_atoms = pred_bb_atoms[:, :, bb_filtered_atom_idx]

        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        bb_error = torch.nan_to_num((gt_bb_atoms - pred_bb_atoms) ** 2, nan=0.0)
        loss_denom = torch.sum(loss_mask, dim=-1) * n_merged_atoms
        bb_atom_loss = (
            torch.sum(
                bb_error * loss_mask[..., None, None],
                dim=(-1, -2, -3),
            )
            / loss_denom
        )

        # SFM correction: add bridge_noise / (2t) to velocity targets (Level >= 2).
        # Applied AFTER auxiliary losses (bb_atom, dist_mat) which use clean targets.
        if "trans_bridge_noise" in noisy_batch:
            sfm_scale = 1.0 / (2.0 * noisy_batch["t"].clamp(min=1e-4))  # [B, 1]
            gt_trans_1 = gt_trans_1 + noisy_batch["trans_bridge_noise"] * sfm_scale[..., None]
            gt_rot_vf = gt_rot_vf + noisy_batch["rot_bridge_noise"] * sfm_scale[..., None]

        # Translation VF loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        )
        trans_error = torch.nan_to_num(trans_error, nan=0.0)
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_error = torch.nan_to_num(rots_vf_error, nan=0.0)
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape(
            [num_batch, num_res * n_merged_atoms, 3]
        )
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_merged_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_merged_atoms])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res).clamp(min=1)

        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * 8
        tors_loss = (
            training_cfg.tors_loss_scale
            * torch.sum(
                torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2
                * loss_mask[..., None],
                dim=(-1, -2),
            )
            / loss_denom
        )

        assert bb_atom_loss.shape[0] == dist_mat_loss.shape[0] == tors_loss.shape[0], (
            f"Loss tensors shape mismatch: {bb_atom_loss.shape} vs {dist_mat_loss.shape} vs {tors_loss.shape}"
        )

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss + tors_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        if torch.isnan(auxiliary_loss).any():
            print("NaN loss in aux_loss")
            auxiliary_loss = auxiliary_loss * 0.0

        if torch.isnan(se3_vf_loss).any():
            print("NaN loss in se3_vf_loss")
            se3_vf_loss = se3_vf_loss * 0.0

        batch_losses = {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
        }
        return batch_losses

    def _compute_bfactor_metrics(self, nu, res_mask, b_factors):
        """Validation metrics comparing predicted absolute B to observed B."""
        if self._bfactor_cfg is None:
            return {}

        b_global_mean = float(self._bfactor_cfg.global_mean_b)
        min_valid_b = float(self._bfactor_cfg.min_valid_b)
        valid = (res_mask > 0.5) & (b_factors > min_valid_b)
        if int(valid.sum().item()) == 0:
            nan = float("nan")
            return {
                "bfactor_mae": nan,
                "bfactor_rmse": nan,
                "bfactor_log_mae": nan,
                "bfactor_log_rmse": nan,
                "bfactor_corr": nan,
            }

        true_b = b_factors[valid].to(nu.dtype)
        pred_b = (nu[valid] * b_global_mean).to(nu.dtype)

        err = pred_b - true_b
        log_err = (
            torch.log(nu[valid].clamp(min=1e-8))
            - torch.log((true_b / b_global_mean).clamp(min=1e-8))
        )

        corr = torch.tensor(float("nan"), device=nu.device, dtype=nu.dtype)
        if true_b.numel() >= 2:
            true_centered = true_b - true_b.mean()
            pred_centered = pred_b - pred_b.mean()
            denom = true_centered.norm() * pred_centered.norm()
            if float(denom.item()) > 1e-8:
                corr = (true_centered * pred_centered).sum() / denom

        return {
            "bfactor_mae": float(err.abs().mean().detach().cpu()),
            "bfactor_rmse": float(torch.sqrt((err ** 2).mean()).detach().cpu()),
            "bfactor_log_mae": float(log_err.abs().mean().detach().cpu()),
            "bfactor_log_rmse": float(torch.sqrt((log_err ** 2).mean()).detach().cpu()),
            "bfactor_corr": float(corr.detach().cpu()),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-cluster validation.

        Both val loaders are now cluster-level (``RNAClusterDataset``) and
        decoy '_cNNN_model' records are filtered out at dataset-load time, so
        each batch item is one real cluster with:

            - ``c4_coords``: anchor real experimental structure [L, 3]
            - ``gt_c4_ensemble``: all real experimental conformers [K, L, 3]
            - ``res_mask``: anchor residue mask [L]

        Metric families are split-specific to match ``evaluate_metrics.py``:

            - ``single`` loader (test_single, K==1):
                ``rmsd`` = best-of-G RMSD vs the anchor
                ``tm_score`` = best-of-G TM-score vs the anchor
            - ``ensemble`` loader (test_ensemble, K>=2):
                ``amr_recall`` / ``amr_precision`` / ``cov_*`` / ``pairwise_rmsd``
                vs the real conformer ensemble

        wandb keys end up as ``valid/single_{rmsd,tm_score}`` and
        ``valid/ensemble_{amr_recall,...}``. ``valid/ensemble_amr_recall`` is
        monitored by ModelCheckpoint in train_se3_flows.py.
        """
        self.interpolant.set_device(batch["res_mask"].device)
        loader_name = "ensemble" if int(dataloader_idx) == 0 else "single"

        num_batch = batch["is_na_residue_mask"].shape[0]
        num_generated = int(getattr(self._validation, "num_generated", 10))
        c4_idx = nucleotide_constants.atom_order["C4'"]

        batch_metrics = []

        for i in range(num_batch):
            sample = {k: v[i] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
            na_mask = sample["is_na_residue_mask"].bool()
            num_res = int(na_mask.sum().item())
            # Strip padding: na_mask selects the actual nucleotide positions
            na_single = sample["single_embedding"][na_mask]          # [num_res, d]
            na_pair = sample["pair_embedding"][na_mask][:, na_mask]  # [num_res, num_res, d]
            # eval_mask restricts metrics to resolved residues only
            eval_mask = sample["res_mask"][na_mask].bool()           # [num_res]

            context = {
                "single_embedding": na_single.unsqueeze(0).repeat(num_generated, 1, 1),
                "pair_embedding": na_pair.unsqueeze(0).repeat(num_generated, 1, 1, 1),
            }

            # Level 3: query FlexibilityNet once on the un-replicated input,
            # then broadcast to num_generated. Mask = resolved (matches
            # training; unresolved positions get nu=1 via head's padding rule).
            nu = None
            nu_one = None
            if self.flexibility_net is not None:
                with torch.no_grad():
                    nu_one = self.flexibility_net(
                        na_single.unsqueeze(0),
                        na_pair.unsqueeze(0),
                        eval_mask.unsqueeze(0).to(na_single.dtype),
                    )
                nu = nu_one.expand(num_generated, -1).contiguous()

            atom37_traj, _, _ = self.interpolant.sample(
                num_generated, num_res, self.model, context=context, nu=nu,
            )
            pred_c4 = atom37_traj[-1][:, :, c4_idx]

            sample_metrics = {}
            if nu_one is not None and "b_factors" in sample:
                sample_metrics.update(
                    self._compute_bfactor_metrics(
                        nu_one.squeeze(0),
                        eval_mask.to(nu_one.device),
                        sample["b_factors"][na_mask].to(nu_one.device),
                    )
                )

            if loader_name == "single":
                gt_c4 = sample["c4_coords"][na_mask].unsqueeze(0).to(pred_c4.device)  # [1, L, 3]
                single_result = metrics.compute_single_metrics(
                    pred_c4, gt_c4, eval_mask.unsqueeze(0).to(pred_c4.device)
                )
                sample_metrics["rmsd"] = float(single_result["rmsd"])
                sample_metrics["tm_score"] = float(single_result["tm_score"])
            else:  # "ensemble"
                gt_c4_ens = sample["gt_c4_ensemble"].to(pred_c4.device)  # [K, L, 3]
                ensemble_result = metrics.compute_ensemble_metrics(
                    pred_c4, gt_c4_ens, eval_mask.to(pred_c4.device)
                )
                sample_metrics.update(ensemble_result)

            batch_metrics.append(sample_metrics)

        self.validation_epoch_metrics_by_loader[loader_name].append(
            pd.DataFrame(batch_metrics)
        )

    def on_validation_epoch_end(self):
        for loader_name, dfs in self.validation_epoch_metrics_by_loader.items():
            if len(dfs) == 0:
                continue
            val_epoch_metrics = pd.concat(dfs)
            for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f"valid/{loader_name}_{metric_name}",
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.validation_epoch_metrics_by_loader[loader_name].clear()

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if on_epoch and not on_step:
            sync_dist = True
            rank_zero_only = False
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def _compute_flex_losses(self, nu, res_mask, b_factors, b_global_mean, min_valid_b):
        """Level 3 B-factor losses on absolute scale.

        L_B  = mean of (log nu_i - log(b_i / B_global))^2 on valid-B residues.
        L_nu = mean of (log nu_i)^2 over resolved residues. Pulls nu -> 1.
        Valid-B = resolved AND b_i > min_valid_b (drops near-zero artifacts that
        would dominate L_B in log space).
        """
        log_nu = torch.log(nu.clamp(min=1e-8))

        res_mask_f = res_mask.to(nu.dtype)
        n_res = res_mask_f.sum().clamp(min=1.0)
        L_nu = (log_nu ** 2 * res_mask_f).sum() / n_res

        valid = (res_mask > 0.5) & (b_factors > min_valid_b)
        valid_f = valid.to(b_factors.dtype)
        n_valid_per_sample = valid_f.sum(dim=-1)
        n_safe = n_valid_per_sample.clamp(min=1.0)

        log_b_hat = torch.log((b_factors / b_global_mean).clamp(min=1e-8))
        diff2 = (log_nu - log_b_hat) ** 2
        per_sample = (diff2 * valid_f).sum(dim=-1) / n_safe
        good_f = (n_valid_per_sample >= 1).to(per_sample.dtype)
        L_B = (per_sample * good_f).sum() / good_f.sum().clamp(min=1.0)

        return L_B, L_nu, int(valid_f.sum().item())

    def training_step(self, batch):
        """One iteration of SE(3) flow matching.

        Level 3 gradient routing: nu enters the bridge detached so L_bridge
        only updates FlowModel; L_B + L_nu only update FlexibilityNet.
        """
        self.interpolant.set_device(batch["res_mask"].device)

        nu_raw = None
        if self._level >= 3:
            nu_raw = self.flexibility_net(
                batch["single_embedding"],
                batch["pair_embedding"],
                batch["res_mask"].to(batch["single_embedding"].dtype),
            )
            nu_det = nu_raw.detach()
            noisy_batch = self.interpolant.corrupt_batch(batch, nu=nu_det)
            noisy_batch["nu"] = nu_det
        else:
            noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["trans_sc"] = model_sc["pred_trans"]

        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses["bb_atom_loss"].shape[0]
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=num_batch,
            )

        train_loss = (
            total_losses[self._exp_cfg.training.loss] + total_losses["auxiliary_loss"]
        )

        if self._level >= 3 and bool(self._bfactor_cfg.enabled):
            L_B, L_nu, n_valid_res = self._compute_flex_losses(
                nu_raw,
                batch["res_mask"],
                batch["b_factors"],
                float(self._bfactor_cfg.global_mean_b),
                float(self._bfactor_cfg.min_valid_b),
            )
            lam_b = float(self._bfactor_cfg.lambda_b)
            lam_nu = float(self._bfactor_cfg.lambda_nu)
            train_loss = train_loss + lam_b * L_B + lam_nu * L_nu

            for key, val in (("L_B", L_B), ("L_nu", L_nu)):
                self._log_scalar(
                    f"train/{key}", val,
                    on_step=False, on_epoch=True, prog_bar=False,
                    batch_size=num_batch,
                )
            self._log_scalar(
                "train/n_valid_bfactor_residues", float(n_valid_res),
                on_step=False, on_epoch=True, prog_bar=False,
                batch_size=num_batch,
            )
            with torch.no_grad():
                mask_f = batch["res_mask"].to(nu_raw.dtype)
                nu_sum = (nu_raw * mask_f).sum()
                nu_count = mask_f.sum().clamp(min=1.0)
                nu_mean = nu_sum / nu_count
                nu_std = ((((nu_raw - nu_mean) ** 2) * mask_f).sum() / nu_count).sqrt()
            self._log_scalar(
                "train/nu_mean", nu_mean,
                on_step=False, on_epoch=True, prog_bar=False,
                batch_size=num_batch,
            )
            self._log_scalar(
                "train/nu_std", nu_std,
                on_step=False, on_epoch=True, prog_bar=False,
                batch_size=num_batch,
            )

        self.log(
            "pbar/loss",
            train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            batch_size=num_batch,
            rank_zero_only=True,
        )
        self._log_scalar(
            "train/loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=num_batch,
        )

        return train_loss

    def configure_optimizers(self):
        if self._level >= 3:
            params = (
                list(self.model.parameters())
                + list(self.flexibility_net.parameters())
            )
        else:
            params = list(self.model.parameters())
        return torch.optim.AdamW(params=params, **self._exp_cfg.optimizer)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Reuse self.interpolant (same config as validation) to ensure
        # identical sampling behaviour.  Only set_device per batch.
        self.interpolant.set_device(batch["res_mask"].device)

        num_batch = batch["is_na_residue_mask"].shape[0]
        c4_idx = nucleotide_constants.atom_order["C4'"]

        ensemble_cfg = getattr(self._infer_cfg, "ensemble", None)
        ensemble_enabled = bool(getattr(ensemble_cfg, "enabled", False)) if ensemble_cfg else False
        num_generated = int(getattr(ensemble_cfg, "num_generated", 1)) if ensemble_enabled else 1
        num_generated = max(1, num_generated)

        outputs = []

        for i in range(num_batch):
            sample = {k: v[i] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
            na_mask = sample["is_na_residue_mask"].bool()
            num_res = int(na_mask.sum().item())
            # Strip padding with na_mask
            eval_mask = sample["res_mask"][na_mask].bool()           # [num_res]
            cluster_id = str(sample.get("cluster_id", sample.get("cluster_name", "unknown")))
            pdb_name = sample.get("pdb_name", None)

            # Output directory
            sample_dir = os.path.join(self._output_dir, cluster_id)
            if not ensemble_enabled:
                sample_dir = os.path.join(sample_dir, f"length_{num_res}")
            os.makedirs(sample_dir, exist_ok=True)

            # Generate samples
            na_single = sample["single_embedding"][na_mask]
            na_pair = sample["pair_embedding"][na_mask][:, na_mask]
            context = {
                "single_embedding": na_single.unsqueeze(0).repeat(num_generated, 1, 1),
                "pair_embedding": na_pair.unsqueeze(0).repeat(num_generated, 1, 1, 1),
            }
            # Sampler selection: by default driven by self._level inside
            # Interpolant.sample (level 1 → ODE, level >= 2 → SDE); set
            # `inference.interpolant.use_sde: false/true` to force-override
            # (e.g. run an L2 checkpoint through ODE for ablation).
            interp_cfg = getattr(self._infer_cfg, "interpolant", None)
            use_sde_cfg = getattr(interp_cfg, "use_sde", None) if interp_cfg else None
            use_sde = None if use_sde_cfg is None else bool(use_sde_cfg)

            # Inference has no "unresolved" concept: mask = all-ones, head
            # predicts nu for every position. Compute once, broadcast.
            nu = None
            if self.flexibility_net is not None:
                with torch.no_grad():
                    nu_one = self.flexibility_net(
                        na_single.unsqueeze(0),
                        na_pair.unsqueeze(0),
                        torch.ones(1, num_res, device=na_single.device, dtype=na_single.dtype),
                    )
                nu = nu_one.expand(num_generated, -1).contiguous()
            atom37_traj, _, _ = self.interpolant.sample(
                num_generated, num_res, self.model, context=context,
                nu=nu, use_sde=use_sde,
            )
            pred_c4 = atom37_traj[-1][:, :, c4_idx]  # [num_generated, num_res, 3]

            # Write structure files (CIF or PDB)
            output_format = getattr(self, "_output_format", "pdb")
            na_aatype = du.to_numpy(sample["aatype"][na_mask]) if "aatype" in sample else None

            if output_format == "cif":
                for j in range(num_generated):
                    au.write_complex_to_cif(
                        du.to_numpy(atom37_traj[-1][j:j+1])[0],
                        os.path.join(sample_dir, f"{cluster_id}_conf{j+1}"),
                        aatype=na_aatype,
                    )
            else:
                is_na = np.ones(num_res, dtype=np.int64)
                for j in range(num_generated):
                    au.write_complex_to_pdbs(
                        du.to_numpy(atom37_traj[-1][j:j+1])[0],
                        os.path.join(sample_dir, f"{cluster_id}_conf{j+1}"),
                        is_na_residue_mask=is_na,
                    )

            outputs.append({
                "cluster_id": cluster_id,
                "length": num_res,
                "num_generated": num_generated,
                "sample_dir": sample_dir,
            })

        return outputs
