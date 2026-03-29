import random
import torch, copy
from rna_backbone_design.data import so3_utils
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import all_atom as rna_all_atom

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment


def _centered_gaussian(num_batch, num_res, device):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    # returns uniformly-sampled rotations (from U(SO(3)))
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),  # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        1 - diffuse_mask[..., None, None]
    )


class Interpolant:
    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """
        Sample random timestep for flow matching
        """
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask):
        """
        Returns optimal transport-based interpolated "flow" for translations
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)

        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[
            ..., None
        ] * trans_1  # flow: t*x1 + (1-t)*x0
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _batch_ot(self, trans_0, trans_1, res_mask):
        """
        Use optimal transport to create the unified "flow" for
        translations by computing a cost matrix
        """
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        """
        Returns interpolated "flow" for rotations without optimal transport
        """
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(  # sample from IGSO(3)
            torch.tensor([1.5]), num_batch * num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)

        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        # compute "flow" for rotations using log/exp maps
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        """
        Params:
            batch (dict) : a dictionary where each key is a tensor name and each value is the corresponding tensor of Rigid objects (representing the RNA frames).

        Remarks:
            This method takes in a clean ground truth batch (x_1) and adds returns the interpolated (noised) batch (x_t).

        Returns:
            batch of corrupted/noisy/interpolated tensors
        """

        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch["trans_1"]  # # [B, N, 3] (in Angstrom)
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3]
        res_mask = batch["res_mask"]  # [B, N]
        num_batch, _ = res_mask.shape

        # get random timestep in [0, 1] for flow matching
        t = self.sample_t(num_batch)[:, None]  # [B, 1]
        noisy_batch["t"] = t

        # apply corruptions to translations
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch["trans_t"] = trans_t

        # apply corruptions to rotations
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch["rotmats_t"] = rotmats_t

        return noisy_batch

    def corrupt_ensemble(self, batch):
        """Corrupt anchor + neighbor conformers with shared x_0 and shared t.

        Returns noisy_batch with:
        - Standard keys (trans_t, rotmats_t, trans_1, rotmats_1) for a randomly chosen conformer j
        - ensemble_trans_t: [B, K, N, 3] all conformers' noisy translations
        - ensemble_rotmats_t: [B, K, N, 3, 3] all conformers' noisy rotations
        - ensemble_mask: [B, K] valid conformer mask
        """
        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch["trans_1"]  # [B, N, 3] anchor
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3] anchor
        res_mask = batch["res_mask"]  # [B, N]
        num_batch, num_res = res_mask.shape

        # Shared t across all conformers
        t = self.sample_t(num_batch)[:, None]  # [B, 1]
        noisy_batch["t"] = t

        # Shared trans_0, OT-aligned to anchor
        trans_0 = _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)

        # Shared rotation noise
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]), num_batch * num_res
        ).to(self._device).reshape(num_batch, num_res, 3, 3)

        # Collect neighbor data (passthrough lists from collate)
        nbr_trans_list = batch.get("nbr_trans", [None] * num_batch)
        nbr_rotmats_list = batch.get("nbr_rotmats", [None] * num_batch)

        all_ens_trans_t = []
        all_ens_rotmats_t = []
        all_ens_trans_1 = []
        all_ens_rotmats_1 = []

        for i in range(num_batch):
            # Build ensemble: anchor + neighbors
            anchor_t1 = trans_1[i].unsqueeze(0)  # [1, N, 3]
            anchor_r1 = rotmats_1[i].unsqueeze(0)  # [1, N, 3, 3]

            nbr_t = nbr_trans_list[i]
            nbr_r = nbr_rotmats_list[i]
            if nbr_t is not None and nbr_t.numel() > 0:
                nbr_t = nbr_t.to(self._device)
                nbr_r = nbr_r.to(self._device)
                ens_trans_1 = torch.cat([anchor_t1, nbr_t], dim=0)  # [K, N, 3]
                ens_rotmats_1 = torch.cat([anchor_r1, nbr_r], dim=0)  # [K, N, 3, 3]
            else:
                ens_trans_1 = anchor_t1
                ens_rotmats_1 = anchor_r1

            t_i = t[i]  # [1]
            trans_0_i = trans_0[i]  # [N, 3]
            noise_i = noisy_rotmats[i]  # [N, 3, 3]

            # Corrupt translations: x_t = (1-t)*x_0 + t*x_1 (shared x_0)
            ens_trans_t = (1 - t_i) * trans_0_i.unsqueeze(0) + t_i * ens_trans_1

            # Corrupt rotations: rotmats_0 = rotmats_1 @ noise (shared noise)
            ens_rotmats_0 = torch.einsum("knij,njl->knil", ens_rotmats_1, noise_i)
            ens_rotmats_t = so3_utils.geodesic_t(
                t_i.unsqueeze(0), ens_rotmats_1, ens_rotmats_0
            )

            all_ens_trans_t.append(ens_trans_t)
            all_ens_rotmats_t.append(ens_rotmats_t)
            all_ens_trans_1.append(ens_trans_1)
            all_ens_rotmats_1.append(ens_rotmats_1)

        # Pad ensemble dim to max K across batch
        max_K = max(x.shape[0] for x in all_ens_trans_t)
        ensemble_trans_t = torch.zeros(num_batch, max_K, num_res, 3, device=self._device)
        ensemble_rotmats_t = torch.zeros(num_batch, max_K, num_res, 3, 3, device=self._device)
        ensemble_trans_1 = torch.zeros(num_batch, max_K, num_res, 3, device=self._device)
        ensemble_rotmats_1 = torch.zeros(num_batch, max_K, num_res, 3, 3, device=self._device)
        ensemble_mask = torch.zeros(num_batch, max_K, device=self._device)

        for i in range(num_batch):
            K_i = all_ens_trans_t[i].shape[0]
            ensemble_trans_t[i, :K_i] = all_ens_trans_t[i]
            ensemble_rotmats_t[i, :K_i] = all_ens_rotmats_t[i]
            ensemble_trans_1[i, :K_i] = all_ens_trans_1[i]
            ensemble_rotmats_1[i, :K_i] = all_ens_rotmats_1[i]
            ensemble_mask[i, :K_i] = 1.0

        noisy_batch["ensemble_trans_t"] = ensemble_trans_t
        noisy_batch["ensemble_rotmats_t"] = ensemble_rotmats_t
        noisy_batch["ensemble_mask"] = ensemble_mask

        # Scheme B: randomly select one conformer j per sample as decoder target
        j_indices = torch.tensor(
            [random.randint(0, int(ensemble_mask[i].sum().item()) - 1) for i in range(num_batch)],
            device=self._device,
        )

        # Gather selected conformer for decoder
        j_idx = j_indices[:, None, None, None].expand(-1, 1, num_res, 3)
        noisy_batch["trans_t"] = ensemble_trans_t.gather(1, j_idx).squeeze(1)
        noisy_batch["trans_1"] = ensemble_trans_1.gather(1, j_idx).squeeze(1)

        j_idx_rot = j_indices[:, None, None, None, None].expand(-1, 1, num_res, 3, 3)
        noisy_batch["rotmats_t"] = ensemble_rotmats_t.gather(1, j_idx_rot).squeeze(1)
        noisy_batch["rotmats_1"] = ensemble_rotmats_1.gather(1, j_idx_rot).squeeze(1)

        # Apply diffuse mask
        noisy_batch["trans_t"] = _trans_diffuse_mask(
            noisy_batch["trans_t"], batch["trans_1"], res_mask
        ) * res_mask[..., None]
        noisy_batch["rotmats_t"] = _rots_diffuse_mask(
            noisy_batch["rotmats_t"], batch["rotmats_1"], res_mask
        )

        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """
        Perform Euler ODESolve step for translations
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Perform Euler ODESolve step for rotations based on prescribed sampling schedule
        """
        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f"Unknown sample schedule {self._rots_cfg.sample_schedule}"
            )

        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def sample(self, num_batch, num_res, model, context=None):
        """
        Params:
            num_batch : number of independent samples
            num_res : number of nucleotides to model
            model : the parameterised vector field (ie, the "denoiser")
            context: dict containing conditioning information (single_embedding, pair_embedding)

        Returns:
            Generated backbone samples in the ATOM37 format
        """

        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = (
            _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        )
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        batch = {
            "res_mask": res_mask,
        }

        if context is not None:
            batch["single_embedding"] = context["single_embedding"]
            batch["pair_embedding"] = context["pair_embedding"]
            if "latent_z" in context:
                batch["latent_z"] = context["latent_z"]

        # get diffusion timesteps in order between [0, 1]
        ts = torch.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:
            # run one step of reverse diffusion (denoising)
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch["trans_t"] = trans_t_1
            batch["rotmats_t"] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch["t"] = t
            with torch.no_grad():
                model_out = model(batch)

            # store model outputs
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch["trans_sc"] = pred_trans_1

            # take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))

            t_1 = t_2

        # NOTE: we only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch["trans_t"] = trans_t_1
        batch["rotmats_t"] = rotmats_t_1
        batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1

        with torch.no_grad():
            model_out = model(
                batch
            )  # final predicted frames (translations + rotations)

        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        pred_torsions_1 = model_out["pred_torsions"]
        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        is_na_residue_mask = torch.ones(num_batch, num_res, device=self._device).bool()

        assert res_mask.shape == is_na_residue_mask.shape, (
            "Shape mismatch between NA masks"
        )

        # retrieve all-atom backbone in ATOM37 format
        atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            prot_traj, is_na_residue_mask, pred_torsions_1
        )
        clean_atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            prot_traj, is_na_residue_mask, pred_torsions_1
        )

        return atom37_traj_rna, clean_atom37_traj_rna, clean_traj
