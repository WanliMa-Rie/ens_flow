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
        Returns interpolated "flow" for rotations without optimal transport.
        R_0 is sampled uniformly on SO(3) (Haar measure) to match the
        inference-time prior; previously used IGSO(3)(sigma=1.5) centered on
        R_1, which biased the training prior toward the data.
        """
        num_batch, num_res = res_mask.shape
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

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
        """Sample R_0 ~ Uniform(SO(3)) (Haar), matching the inference prior."""
        num_batch, num_res = res_mask.shape
        return _uniform_so3(num_batch, num_res, self._device)

    def _corrupt_trans_bridge(self, trans_1, t, res_mask, sigma_x, min_sigma_x, nu=None):
        """Euclidean Brownian bridge in translation space.

        Envelope (FoldFlow-style floor):
            std_i(t)^2 = sigma_x^2 * nu_i * t(1-t) + min_sigma_x^2
        Level 2 uses nu_i = 1. min_sigma_x = 0 recovers the strict bridge.
        Per-residue independent Gaussian noise; no CoM re-centering.
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)

        mean = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1  # [B, N, 3]

        if nu is None:
            nu_bn = torch.ones_like(res_mask)                         # [B, N]
        else:
            nu_bn = nu.clamp(min=0.0)                                 # [B, N]
        var = (sigma_x ** 2) * nu_bn * (t * (1.0 - t)) + (min_sigma_x ** 2)
        std_bn = torch.sqrt(var + 1e-12)                              # [B, N]

        trans_t = mean + std_bn[..., None] * torch.randn_like(mean)
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_rotmats_bridge(self, rotmats_1, t, res_mask, sigma_omega, min_sigma_omega, nu=None):
        """Tangent-space-lifted Brownian bridge on SO(3).

        Envelope (FoldFlow-style floor):
            std_i(t)^2 = sigma_omega^2 * nu_i * t(1-t) + min_sigma_omega^2
        Level 2 uses nu_i = 1. min_sigma_omega = 0 recovers the strict bridge.

        1. Sample R_0 ~ Uniform(SO(3)) Haar (same prior as Level 1, matches inference).
        2. omega = Log(R_0^T @ R_1) in so(3) ~= R^3.
        3. Omega_t = t * omega + std_i(t) * eps.
        4. R_t = R_0 @ Exp(hat(Omega_t)).
        """
        rotmats_0 = self._sample_rotmats_0(rotmats_1, res_mask)

        # omega = Log(R_0^T R_1)
        omega = so3_utils.calc_rot_vf(rotmats_0, rotmats_1)        # [B, N, 3]
        if nu is None:
            nu_bn = torch.ones_like(res_mask)                      # [B, N]
        else:
            nu_bn = nu.clamp(min=0.0)                              # [B, N]
        var = (sigma_omega ** 2) * nu_bn * (t * (1.0 - t)) + (min_sigma_omega ** 2)
        std_bn = torch.sqrt(var + 1e-12)                           # [B, N]
        Omega_t = t[..., None] * omega + std_bn[..., None] * torch.randn_like(omega)

        # Push forward to SO(3): R_t = R_0 @ Exp(hat(Omega_t))
        delta_R = so3_utils.rotvec_to_rotmat(Omega_t)              # [B, N, 3, 3]
        rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_0, delta_R)

        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch, nu=None, sigma_scale=1.0):
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
            scale = float(sigma_scale)
            sigma_x = float(self._bridge_cfg.bridge.sigma_x) * scale
            sigma_omega = float(self._bridge_cfg.bridge.sigma_omega) * scale
            min_sigma_x = float(getattr(self._bridge_cfg.bridge, "min_sigma_x", 0.0)) * scale
            min_sigma_omega = float(getattr(self._bridge_cfg.bridge, "min_sigma_omega", 0.0)) * scale
            trans_t = self._corrupt_trans_bridge(
                trans_1, t, res_mask, sigma_x, min_sigma_x, nu=nu
            )
            rotmats_t = self._corrupt_rotmats_bridge(
                rotmats_1, t, res_mask, sigma_omega, min_sigma_omega, nu=nu
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

    def _bridge_sde_step(
        self,
        d_t,
        t,
        pred_trans_1,
        pred_rotmats_1,
        trans_t,
        rotmats_t,
        nu=None,
    ):
        """Bridge-consistent Euler-Maruyama step on SE(3)^L.

        Drift identical to the deterministic ODE under the linear rotation
        schedule; diffusion follows the training bridge envelope
        sigma * sqrt(t(1-t)) per unit time, optionally scaled per residue
        by sqrt(nu_i). Per-residue independent noise (no CoM constraint).
        Rotation noise lives in the Lie algebra and is added inside the exp.

        Requires self._level >= 2.
        """
        sigma_x = float(self._bridge_cfg.bridge.sigma_x)
        sigma_omega = float(self._bridge_cfg.bridge.sigma_omega)
        min_sigma_x = float(getattr(self._bridge_cfg.bridge, "min_sigma_x", 0.0))
        min_sigma_omega = float(getattr(self._bridge_cfg.bridge, "min_sigma_omega", 0.0))

        # Envelope: std_i(t)^2 = sigma^2 * nu_i * t(1-t) + min_sigma^2
        t_bridge = t * (1.0 - t)
        if nu is None:
            trans_noise_scale = torch.sqrt((sigma_x ** 2) * t_bridge + (min_sigma_x ** 2) + 1e-12)
            rot_noise_scale = torch.sqrt((sigma_omega ** 2) * t_bridge + (min_sigma_omega ** 2) + 1e-12)
        else:
            nu_bn = nu.clamp(min=0.0)                                   # [B, N]
            trans_noise_scale = torch.sqrt(
                (sigma_x ** 2) * nu_bn * t_bridge + (min_sigma_x ** 2) + 1e-12
            )[..., None]                                                # [B, N, 1]
            rot_noise_scale = torch.sqrt(
                (sigma_omega ** 2) * nu_bn * t_bridge + (min_sigma_omega ** 2) + 1e-12
            )[..., None]                                                # [B, N, 1]

        sqrt_dt = d_t.sqrt()

        # --- Translations: ODE drift + per-residue bridge noise ---------
        trans_vf = (pred_trans_1 - trans_t) / (1 - t)
        xi_x = torch.randn_like(trans_t)                                # [B, N, 3]
        trans_new = trans_t + trans_vf * d_t + trans_noise_scale * xi_x * sqrt_dt
        # CoM recenter to prevent drift from per-residue independent noise
        # (matches FoldFlow's r3_fm.reverse center=True behavior).
        trans_new = trans_new - trans_new.mean(dim=-2, keepdim=True)

        # --- Rotations: linear-schedule drift + tangent-space bridge noise
        rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        xi_R = torch.randn_like(rot_vf)                                 # [B, N, 3]
        omega_step = (d_t / (1 - t)) * rot_vf + rot_noise_scale * xi_R * sqrt_dt
        delta_R = so3_utils.rotvec_to_rotmat(omega_step)
        rotmats_new = torch.einsum("...ij,...jk->...ik", rotmats_t, delta_R)

        return trans_new, rotmats_new

    def sample(self, num_batch, num_res, model, context=None, nu=None, use_sde=None):
        """
        Params:
            num_batch : number of independent samples
            num_res : number of nucleotides to model
            model : the parameterised vector field (ie, the "denoiser")
            context: dict containing conditioning information (single_embedding, pair_embedding)
            nu : optional ``[num_batch, num_res]`` per-residue flexibility
                multiplier from the Level-3 FlexibilityNet on absolute scale
                (nu>1 = more dynamic than dataset average, nu<1 = more rigid).
                ``None`` (Level 2 or no head) falls back to ``nu_i ≡ 1``
                (homoscedastic bridge SDE).
            use_sde: optional sampler override. ``None`` → auto from level
                (level 1 → ODE, level >= 2 → SDE). ``True``/``False`` force
                the choice; ``True`` at level 1 is invalid and raises.

        Returns:
            Generated backbone samples in the ATOM37 format
        """

        if use_sde is None:
            use_sde = self._level >= 2
        if use_sde and self._level < 2:
            raise ValueError(
                "SDE sampling requires level >= 2 (bridge config must be present)."
            )
        if nu is not None:
            nu = nu.to(self._device)

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

        if nu is not None:
            batch["nu"] = nu

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

            if not use_sde:
                # Level 1: deterministic ODE.
                trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
                rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            else:
                # Level >= 2: bridge-consistent Euler-Maruyama.
                trans_t_2, rotmats_t_2 = self._bridge_sde_step(
                    d_t,
                    t_1,
                    pred_trans_1,
                    pred_rotmats_1,
                    trans_t_1,
                    rotmats_t_1,
                    nu,
                )
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

