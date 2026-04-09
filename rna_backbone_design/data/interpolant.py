import os
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
    def __init__(self, cfg, bridge_cfg=None):
        """
        Args:
            cfg: interpolant config (rots, sampling, min_t, self_condition).
            bridge_cfg: optional stochastic-bridge config. When `bridge_cfg.level >= 2`,
                `corrupt_batch` switches to the tangent-space-lifted SE(3) stochastic bridge
                with fixed widths `bridge_cfg.bridge.sigma_x` (Å) and
                `bridge_cfg.bridge.sigma_omega` (rad). Level 1 ignores `bridge_cfg` entirely.
        """
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._sample_cfg = cfg.sampling
        self._bridge_cfg = bridge_cfg
        self._level = int(bridge_cfg.level) if bridge_cfg is not None else 1
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            # Cache directory is read from the IGSO3_CACHE_DIR env var so that
            # Modal (or any other persistent-volume backend) can point it at a
            # mounted volume and avoid re-computing the ~5-minute IGSO(3) CDF
            # expansion on every container start. Default `.cache` keeps local
            # development behaviour unchanged.
            cache_dir = os.environ.get("IGSO3_CACHE_DIR", ".cache")
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=cache_dir)
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
        # Only sum over resolved (res_mask=1) positions to avoid padding contamination
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        norms = torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1)
        masked_cost = (norms * batch_mask).sum(dim=-1)
        denom = batch_mask.sum(dim=-1).clamp(min=1)  # avoid div-by-zero for empty masks
        cost_matrix = masked_cost / denom
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

    # ---------------------------------------------------------------------
    # Level 2: SE(3) stochastic bridge (tangent-space lifted, fixed sigma)
    # ---------------------------------------------------------------------

    def _sample_rotmats_0(self, rotmats_1, res_mask):
        """Sample R_0 ~ IGSO(3)(sigma=1.5) * R_1 (identical prior to Level 1).

        Returns:
            rotmats_0: [B, N, 3, 3] prior rotations in the same frame as `rotmats_1`.
        """
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]), num_batch * num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        return torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

    def _corrupt_trans_bridge(self, trans_1, t, res_mask, sigma_x, nu=None):
        """Euclidean Brownian bridge in translation space, with post-multiply
        re-centering of the heteroscedastic noise across resolved residues.

        Level 2 (nu is None or constant):
            x_t = (1 - t) x_0 + t x_1 + sigma_x * sqrt(t(1-t)) * eps_centered
        Level 3 (nu is a [B, N] per-residue flexibility multiplier):
            eta_i           = sqrt(nu_i) * eps_i
            eta_centered_i  = eta_i - mean_{k in R} eta_k    (R = resolved residues)
            x_{i,t}         = (1 - t) x_{i,0} + t x_{i,1}
                              + sigma_x * sqrt(t(1-t)) * eta_centered_i

        Why post-multiply re-centering. The deterministic Level-1 prior is a
        centered Gaussian (zero residue-mean per sample, after batch optimal
        transport). Heteroscedastic IID noise breaks that prior because
        ``mean_i sqrt(nu_i) eps_i`` is in general nonzero. Subtracting the
        residue-mean of ``eta`` *after* the per-residue scaling restores
        ``sum_{i in R} eta_centered_i = 0`` for any nu, so the corrupted x_t
        keeps the same residue-mean as the linear interpolation
        ``(1-t) mean_R x_0 + t mean_R x_1``.

        When ``nu is None`` (or nu ≡ 1) the centering is still applied; this
        only nudges Level-2 corruption by O(1/m_R) below the noise floor and
        keeps the code path uniform between L2 and L3.
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)

        mean = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1  # [B, N, 3]
        # Per-batch scalar BB std: sigma_x * sqrt(t(1-t)). Shape [B, 1].
        std = sigma_x * torch.sqrt(t * (1.0 - t) + 1e-8)              # [B, 1]

        if nu is None:
            eta_factor = torch.ones_like(res_mask)                    # [B, N]
        else:
            # eta_i = sqrt(nu_i) * eps_i
            eta_factor = torch.sqrt(nu.clamp(min=0.0) + 1e-12)        # [B, N]

        raw_eps = torch.randn_like(mean)                              # [B, N, 3]
        eta = raw_eps * eta_factor[..., None]                         # [B, N, 3]

        # Post-multiply re-center across RESOLVED residues per sample.
        # res_mask is [B, N]; m_R = number of resolved residues per sample.
        res_mask_f = res_mask.to(eta.dtype)                           # [B, N]
        m_R = res_mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)     # [B, 1]
        eta_sum = (eta * res_mask_f[..., None]).sum(dim=-2)           # [B, 3]
        eta_mean = eta_sum / m_R                                      # [B, 3]
        # Subtract eta_mean only from resolved positions; padding stays at 0
        # because the trailing res_mask multiplication zeros it anyway.
        eta_centered = eta - eta_mean[:, None, :] * res_mask_f[..., None]   # [B, N, 3]

        # std is [B, 1]; we want [B, 1, 1] so it broadcasts cleanly across
        # [B, N, 3]. Adding two trailing Nones would give [B, 1, 1, 1] which
        # broadcasts catastrophically against [B, N, 3] into [B, B, N, 3].
        noise = std[..., None] * eta_centered                          # [B, N, 3]
        trans_t = mean + noise

        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_rotmats_bridge(self, rotmats_1, t, res_mask, sigma_omega, nu=None):
        """Tangent-space-lifted Brownian bridge on SO(3).

        Level 2:
            Omega_t = t * omega + sigma_omega * sqrt(t(1-t)) * eps
        Level 3 (nu is a [B, N] per-residue flexibility multiplier):
            Omega_{i,t} = t * omega_i
                          + sigma_omega * sqrt(nu_i) * sqrt(t(1-t)) * eps_i

        When `nu is None` (or nu ≡ 1), this reduces exactly to Level 2.

        1. Sample R_0 ~ IGSO(3)(sigma=1.5) * R_1 (same prior as Level 1).
        2. omega = Log(R_0^T @ R_1) in so(3) ~= R^3.
        3. Omega_t as above.
        4. R_t = R_0 @ Exp(hat(Omega_t)).

        At t=0, Omega_t=0 => R_t = R_0. At t=1, Omega_t = omega => R_t = R_1.
        sigma_omega=0 recovers Level 1's geodesic interpolation exactly.
        """
        rotmats_0 = self._sample_rotmats_0(rotmats_1, res_mask)

        # omega = Log(R_0^T R_1)
        omega = so3_utils.calc_rot_vf(rotmats_0, rotmats_1)        # [B, N, 3]
        std = sigma_omega * torch.sqrt(t * (1.0 - t) + 1e-8)       # [B, 1]
        if nu is None:
            std_bn = std.expand_as(res_mask)                       # [B, N]
        else:
            std_bn = std * torch.sqrt(nu.clamp(min=0.0) + 1e-12)   # [B, N]
        noise = torch.randn_like(omega) * std_bn[..., None]        # [B, N, 3]
        Omega_t = t[..., None] * omega + noise                     # [B, N, 3]

        # Push forward to SO(3): R_t = R_0 @ Exp(hat(Omega_t))
        delta_R = so3_utils.rotvec_to_rotmat(Omega_t)              # [B, N, 3, 3]
        rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_0, delta_R)

        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch, nu=None):
        """
        Params:
            batch (dict) : a dictionary where each key is a tensor name and each value is the corresponding tensor of Rigid objects (representing the RNA frames).
            nu : optional ``[B, N]`` per-residue flexibility multiplier for
                Level 3. When provided, the bridge widths become
                sigma_x_i = sigma_x * sqrt(nu_i) and likewise for rotations.
                **Must already be detached** if the caller does not want
                ``L_bridge`` to backprop through the flexibility head — this
                function does not re-detach.

        Remarks:
            Level 1 (self._level == 1): deterministic SE(3) flow-matching corruption
                (linear interpolation for translations, geodesic for rotations).
            Level 2 (self._level == 2): tangent-space-lifted SE(3) stochastic bridge
                with fixed (sigma_x, sigma_omega) read from self._bridge_cfg.bridge.
                The only difference from Level 1 is a single additive BB noise term
                per modality inside corruption — loss, model, and gradients are unchanged.
            Level 3 (self._level >= 3): same as Level 2 but with per-residue
                sigma scaling via ``nu``. If ``nu is None`` at Level 3, falls
                back to Level 2 behaviour (nu ≡ 1).

        Returns:
            batch of corrupted/noisy/interpolated tensors
        """

        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch["trans_1"]  # # [B, N, 3] (in Angstrom)
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3]
        res_mask = batch["res_mask"]  # [B, N]
        num_batch = res_mask.shape[0]

        # get random timestep in [0, 1] for flow matching
        t = self.sample_t(num_batch)[:, None]  # [B, 1]
        noisy_batch["t"] = t

        if self._level >= 2:
            sigma_x = float(self._bridge_cfg.bridge.sigma_x)
            sigma_omega = float(self._bridge_cfg.bridge.sigma_omega)
            trans_t = self._corrupt_trans_bridge(
                trans_1, t, res_mask, sigma_x, nu=nu
            )
            rotmats_t = self._corrupt_rotmats_bridge(
                rotmats_1, t, res_mask, sigma_omega, nu=nu
            )
        else:
            trans_t = self._corrupt_trans(trans_1, t, res_mask)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch["trans_t"] = trans_t
        noisy_batch["rotmats_t"] = rotmats_t

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

    def _trans_euler_maruyama_step(self, d_t, t, trans_1, trans_t, a_trans):
        """Euler-Maruyama step for translations: drift + diffusion.

        Implements: x_{k+1} = x_k + Δt·v_{x,θ} + √Δt · a_{x,ψ} · ξ_k
        (Section 5, with U_{x,θ}=I)
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        noise = torch.randn_like(trans_t)
        return trans_t + trans_vf * d_t + a_trans[..., None] * noise * d_t.sqrt()

    def _rots_euler_maruyama_step(self, d_t, t, rotmats_1, rotmats_t, a_rots):
        """Euler-Maruyama step on SO(3): geodesic drift + tangent-space noise.

        Implements: R_{k+1} = R_k exp(Δt·ω_θ + √Δt · σ_ω · ζ_k)
        """
        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self._rots_cfg.sample_schedule}")
        rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        noise = torch.randn_like(rot_vf)
        total_rotvec = scaling * d_t * rot_vf + a_rots[..., None] * noise * d_t.sqrt()
        delta_rot = so3_utils.rotvec_to_rotmat(total_rotvec)
        return torch.einsum("...ij,...jk->...ik", rotmats_t, delta_rot)

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

        # retrieve all-atom backbone in ATOM37 format. The "noisy" trajectory
        # comes from the SDE state in `prot_traj`; the "clean" trajectory comes
        # from the model's per-step clean-endpoint predictions stored in
        # `clean_traj` (each entry is the FlowModel's prediction of (x_1, R_1)
        # at that step). Building both from `prot_traj` would have made the
        # "clean" output identical to the noisy output and lose the predicted
        # endpoint information.
        atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            prot_traj, is_na_residue_mask, pred_torsions_1
        )
        clean_atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            clean_traj, is_na_residue_mask, pred_torsions_1
        )

        return atom37_traj_rna, clean_atom37_traj_rna, clean_traj

    def sample_bridge(self, num_batch, num_res, model, context=None, nu=None):
        """Level 2 / Level 3 inference: Euler-Maruyama on SE(3).

        At each step the drift model predicts (x_hat_1, R_hat_1), which induces
        the same drift field as Level 1. Fixed sigma_x, sigma_omega from the
        bridge config drive the diffusion term:

            x_{k+1,i} = x_{k,i} + dt * u_theta^x
                        + sqrt(dt) * sigma_x * sqrt(nu_i) * xi
            R_{k+1,i} = R_{k,i} @ Exp(hat( dt * u_theta^omega
                        + sqrt(dt) * sigma_omega * sqrt(nu_i) * zeta ))

        When ``nu is None`` the per-residue multiplier defaults to 1 and
        inference is bit-identical to Level 2.

        Requires self._bridge_cfg to carry bridge.sigma_x and bridge.sigma_omega
        (i.e. Interpolant was constructed with bridge_cfg and level >= 2).

        Args:
            num_batch: number of independent samples.
            num_res: number of residues.
            model: drift field network (same FlowModel used for Level 1).
            context: conditioning dict (single_embedding, pair_embedding).
            nu: optional ``[num_batch, num_res]`` per-residue flexibility
                multiplier from the Level 3 FlexibilityNet. Should be detached
                before passing (inference is no-grad anyway, but this keeps
                the contract uniform with ``corrupt_batch``).

        Returns:
            (atom37_traj, clean_atom37_traj, clean_traj)
        """
        assert self._bridge_cfg is not None and self._level >= 2, (
            "sample_bridge requires Interpolant to be constructed with a "
            "stochastic_bridge config at level >= 2."
        )
        sigma_x = float(self._bridge_cfg.bridge.sigma_x)
        sigma_omega = float(self._bridge_cfg.bridge.sigma_omega)

        res_mask = torch.ones(num_batch, num_res, device=self._device)
        trans_0 = (
            _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        )
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        # Broadcast-ready per-residue sigma tensors for the existing Euler-Maruyama
        # helpers (which expect `a_trans` / `a_rots` shaped [B, N]).
        # Level 2: constant across residues. Level 3: scale by sqrt(nu_i).
        if nu is None:
            nu_factor = torch.ones(num_batch, num_res, device=self._device)
        else:
            nu_factor = torch.sqrt(nu.to(self._device).clamp(min=0.0) + 1e-12)
        sigma_x_tensor = sigma_x * nu_factor
        sigma_omega_tensor = sigma_omega * nu_factor

        batch = {"res_mask": res_mask}
        if context is not None:
            batch["single_embedding"] = context["single_embedding"]
            batch["pair_embedding"] = context["pair_embedding"]

        ts = torch.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []

        for t_2 in ts[1:]:
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch["trans_t"] = trans_t_1
            batch["rotmats_t"] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch["t"] = t

            with torch.no_grad():
                model_out = model(batch)
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]

            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch["trans_sc"] = pred_trans_1

            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_maruyama_step(
                d_t, t_1, pred_trans_1, trans_t_1, sigma_x_tensor
            )
            rotmats_t_2 = self._rots_euler_maruyama_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1, sigma_omega_tensor
            )
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # Final deterministic step: use the last model prediction as the clean endpoint.
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch["trans_t"] = trans_t_1
        batch["rotmats_t"] = rotmats_t_1
        batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1

        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        pred_torsions_1 = model_out["pred_torsions"]
        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        is_na_residue_mask = torch.ones(num_batch, num_res, device=self._device).bool()
        # See `sample()` for the prot_traj-vs-clean_traj distinction.
        atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            prot_traj, is_na_residue_mask, pred_torsions_1
        )
        clean_atom37_traj_rna = rna_all_atom.transrot_to_atom37_rna(
            clean_traj, is_na_residue_mask, pred_torsions_1
        )

        return atom37_traj_rna, clean_atom37_traj_rna, clean_traj

