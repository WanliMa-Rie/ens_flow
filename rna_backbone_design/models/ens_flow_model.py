"""
Ensemble-Supervised Variational Latent Flow Matching model.

Wraps FlowModel with posterior/prior encoders and AdaLN latent conditioning.
See docs/ens.md for the full design.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rna_backbone_design.models.flow_model import FlowModel
from rna_backbone_design.models.ipa_pytorch import (
    InvariantPointAttention,
    StructureModuleTransition,
    EdgeTransition,
    Linear,
)
from rna_backbone_design.models import utils as model_utils
from rna_backbone_design.data import utils as du


class StructureEncoder(nn.Module):
    """Encodes ensemble structures into per-residue tokens using IPA.

    Purely geometric encoder — no sequence information.
    Processes each conformer independently; cross-conformer interaction
    is handled downstream by PosteriorEncoder.
    """

    def __init__(self, model_conf, num_blocks=2):
        super().__init__()
        ipa_conf = model_conf.ipa
        c_s = ipa_conf.c_s
        c_z = ipa_conf.c_z
        self.num_blocks = num_blocks

        # --- Geometric node embedder (position encoding only) ---
        c_pos_emb = 128
        self.c_pos_emb = c_pos_emb
        self.node_proj = nn.Linear(c_pos_emb, c_s)

        # --- Geometric edge embedder (distogram + relpos + cross-node) ---
        feat_dim = 64
        num_bins = 22
        self.feat_dim = feat_dim
        self.num_bins = num_bins
        self.linear_s_p = nn.Linear(c_s, feat_dim)
        self.linear_relpos = nn.Linear(feat_dim, feat_dim)
        total_edge_feats = feat_dim * 2 + feat_dim + num_bins
        self.edge_proj = nn.Sequential(
            nn.Linear(total_edge_feats, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z),
        )

        # --- Unit conversion (IPA expects nanometers) ---
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda t: t * du.ANG_TO_NM_SCALE
        )

        # --- IPA trunk (no backbone update) ---
        self.trunk = nn.ModuleDict()
        for b in range(num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(c_s)
            tfmr_layer = nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(c_s, c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=c_s)
            if b < num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=c_z,
                    edge_embed_out=c_z,
                )

    def _embed_nodes(self, mask):
        """Position-encoding-only node embedding."""
        BK, N = mask.shape
        pos = torch.arange(N, dtype=torch.float32, device=mask.device)[None]
        pos_emb = model_utils.get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb.expand(BK, -1, -1)
        return self.node_proj(pos_emb) * mask[..., None]

    def _embed_edges(self, node_embed, trans, mask):
        """Geometric edge embedding: distogram + relpos + cross-node."""
        BK, N, _ = node_embed.shape
        device = node_embed.device

        # Cross-node features [BK, N, N, 2*feat_dim]
        p_i = self.linear_s_p(node_embed)
        cross = torch.cat([
            p_i[:, :, None, :].expand(-1, -1, N, -1),
            p_i[:, None, :, :].expand(-1, N, -1, -1),
        ], dim=-1)

        # Relative position encoding [BK, N, N, feat_dim]
        pos = torch.arange(N, device=device).unsqueeze(0).expand(BK, -1)
        rel_pos = pos[:, :, None] - pos[:, None, :]
        relpos_emb = self.linear_relpos(
            model_utils.get_index_embedding(rel_pos, self.feat_dim, max_len=2056)
        )

        # Distogram [BK, N, N, num_bins]
        dist_feats = model_utils.calc_distogram(
            trans, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins
        )

        all_feats = torch.cat([cross, relpos_emb, dist_feats], dim=-1)
        edge_mask = mask[:, None] * mask[:, :, None]
        return self.edge_proj(all_feats) * edge_mask[..., None]

    def forward(self, ensemble_trans_t, ensemble_rotmats_t, res_mask, ensemble_mask):
        """
        Args:
            ensemble_trans_t: [B, K, N, 3]
            ensemble_rotmats_t: [B, K, N, 3, 3]
            res_mask: [B, N]
            ensemble_mask: [B, K]
        Returns:
            struct_tokens: [B, K, N, c_s]
        """
        B, K, N, _ = ensemble_trans_t.shape

        # Flatten B*K — process each conformer independently
        trans = ensemble_trans_t.reshape(B * K, N, 3)
        rotmats = ensemble_rotmats_t.reshape(B * K, N, 3, 3)
        mask = res_mask[:, None].expand(-1, K, -1).reshape(B * K, N)

        # Embeddings
        node_embed = self._embed_nodes(mask)
        edge_embed = self._embed_edges(node_embed, trans, mask)

        # Build rigids and convert to nm
        rigids = du.create_rigid(rotmats, trans)
        rigids = self.rigids_ang_to_nm(rigids)

        # IPA trunk (fixed rigids, no backbone update)
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, rigids, mask)
            ipa_embed *= mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - mask).bool()
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed *= mask[..., None]

            if b < self.num_blocks - 1:
                edge_mask = mask[:, None] * mask[:, :, None]
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # Reshape back to [B, K, N, c_s]
        return node_embed.reshape(B, K, N, -1)



class PosteriorEncoder(nn.Module):
    """q_φ(z | X_t, c, t) — dynamics-aware posterior encoder.

    Three-level hierarchy:
      1. Cross-attention over conformers  → per-residue consensus
      2. Conformer variance + gated fusion → per-residue flexibility signal
      3. Self-attention over residues      → correlated dynamics
      4. Attention pooling (learned query) → global latent summary
    """

    def __init__(self, c_s, z_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.c_s = c_s

        # --- 1. Cross-attention over conformers (per-residue) ---
        self.W_Q = nn.Linear(c_s, c_s)
        self.W_K = nn.Linear(c_s, c_s)
        self.W_V = nn.Linear(c_s, c_s)
        self.cross_ln = nn.LayerNorm(c_s)

        # --- 2. Conformer variance (flexibility) + gated fusion ---
        self.var_proj = nn.Linear(c_s, c_s)
        self.dynamics_gate = nn.Sequential(
            nn.Linear(c_s * 2, c_s),
            nn.Sigmoid(),
        )
        self.dynamics_ln = nn.LayerNorm(c_s)

        # --- 3. Self-attention over residues (correlated motions) ---
        tfmr_layer = nn.TransformerEncoderLayer(
            d_model=c_s, nhead=num_heads, dim_feedforward=c_s * 4,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.residue_tfmr = nn.TransformerEncoder(
            tfmr_layer, num_layers, enable_nested_tensor=False,
        )

        # --- 4. Attention pooling (learned query) ---
        self.pool_query = nn.Parameter(torch.randn(1, 1, c_s) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            c_s, num_heads, batch_first=True,
        )

        # --- 5. Output ---
        self.out_mlp = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.GELU(),
            nn.Linear(c_s, 2 * z_dim),
        )

    def forward(self, seq_tokens, struct_tokens, res_mask, ensemble_mask):
        """
        Args:
            seq_tokens: [B, N, c_s] sequence + time embedding
            struct_tokens: [B, K, N, c_s] K conformers encoded
            res_mask: [B, N]
            ensemble_mask: [B, K] which conformers are valid
        Returns:
            mu_q: [B, z_dim], log_sigma_q: [B, z_dim]
        """
        B, _, _, D = struct_tokens.shape

        # 1. Cross-attention: each residue attends over K conformers
        q = self.W_Q(seq_tokens)         # [B, N, D]
        k = self.W_K(struct_tokens)      # [B, K, N, D]
        v = self.W_V(struct_tokens)      # [B, K, N, D]

        q_r = q.unsqueeze(2)             # [B, N, 1, D]
        k_r = k.permute(0, 2, 1, 3)     # [B, N, K, D]
        v_r = v.permute(0, 2, 1, 3)     # [B, N, K, D]

        attn = torch.matmul(q_r, k_r.transpose(-1, -2)) / (D ** 0.5)
        attn_mask = ensemble_mask[:, None, None, :]   # [B, 1, 1, K]
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(attn_mask == 0, 0.0)
        cross_out = torch.matmul(attn, v_r).squeeze(2)  # [B, N, D]
        cross_out = self.cross_ln(seq_tokens + cross_out)

        # 2. Conformer variance → per-residue flexibility signal
        ens_mask_exp = ensemble_mask[:, :, None, None]               # [B, K, 1, 1]
        n_valid = ensemble_mask.sum(1, keepdim=True)[:, :, None, None]  # [B, 1, 1, 1]
        conf_mean = (struct_tokens * ens_mask_exp).sum(1) / n_valid.squeeze(1)  # [B, N, D]
        conf_var = (
            (struct_tokens - conf_mean.unsqueeze(1)) ** 2 * ens_mask_exp
        ).sum(1) / n_valid.squeeze(1)   # [B, N, D]
        var_feat = self.var_proj(conf_var)

        # Gated fusion of cross-attention output and variance feature
        gate = self.dynamics_gate(torch.cat([cross_out, var_feat], dim=-1))
        dynamics = self.dynamics_ln(cross_out + gate * var_feat)

        # 3. Self-attention over residues (captures correlated motions)
        pad_mask = (1 - res_mask).bool()
        dynamics = self.residue_tfmr(dynamics, src_key_padding_mask=pad_mask)
        dynamics = dynamics * res_mask[..., None]

        # 4. Attention pooling with learned query
        pool_q = self.pool_query.expand(B, -1, -1)       # [B, 1, D]
        pooled, _ = self.pool_attn(
            pool_q, dynamics, dynamics, key_padding_mask=pad_mask,
        )
        pooled = pooled.squeeze(1)  # [B, D]

        # 5. Output
        params = self.out_mlp(pooled)
        mu_q, log_sigma_q = params.chunk(2, dim=-1)
        return mu_q, log_sigma_q


class PriorEncoder(nn.Module):
    """p_θ(z | c) — sequence-only prior, structurally aligned with posterior.

    Mirrors the posterior's reasoning pattern (self-attention + attention pooling)
    but uses only sequence information (no structure).  Learns to predict which
    regions of the sequence are likely flexible from sequence context alone.
    """

    def __init__(self, c_s, z_dim, num_heads=8, num_layers=2):
        super().__init__()

        # Self-attention over residues (learn flexibility from sequence)
        tfmr_layer = nn.TransformerEncoderLayer(
            d_model=c_s, nhead=num_heads, dim_feedforward=c_s * 4,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.residue_tfmr = nn.TransformerEncoder(
            tfmr_layer, num_layers, enable_nested_tensor=False,
        )

        # Attention pooling (mirrors posterior)
        self.pool_query = nn.Parameter(torch.randn(1, 1, c_s) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            c_s, num_heads, batch_first=True,
        )

        # Output
        self.out_mlp = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.GELU(),
            nn.Linear(c_s, 2 * z_dim),
        )

    def forward(self, seq_tokens, mask):
        """
        Args:
            seq_tokens: [B, N, c_s] time-independent sequence tokens
            mask: [B, N]
        Returns:
            mu_p: [B, z_dim], log_sigma_p: [B, z_dim]
        """
        B = seq_tokens.shape[0]
        pad_mask = (1 - mask).bool()

        # Self-attention: learn which residues are likely flexible
        h = self.residue_tfmr(seq_tokens, src_key_padding_mask=pad_mask)
        h = h * mask[..., None]

        # Attention pooling
        pool_q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(pool_q, h, h, key_padding_mask=pad_mask)
        pooled = pooled.squeeze(1)  # [B, c_s]

        params = self.out_mlp(pooled)
        mu_p, log_sigma_p = params.chunk(2, dim=-1)
        return mu_p, log_sigma_p


class EnsFlowModel(nn.Module):
    """Flow model with latent variable conditioning for ensemble generation."""

    def __init__(self, model_conf, ens_flow_conf):
        super().__init__()
        self.flow_model = FlowModel(model_conf)
        self._ipa_conf = model_conf.ipa

        c_s = model_conf.node_embed_size
        z_dim = ens_flow_conf.z_dim
        num_blocks = model_conf.ipa.num_blocks

        se_cfg = ens_flow_conf.structure_encoder
        self.structure_encoder = StructureEncoder(
            model_conf,
            num_blocks=getattr(se_cfg, "num_blocks", 2),
        )
        self.posterior_encoder = PosteriorEncoder(c_s, z_dim)
        self.prior_encoder = PriorEncoder(c_s, z_dim)

        # AdaLN: z -> (gamma, beta) per trunk block, zero-init for identity start
        self.adaln_mlps = nn.ModuleList()
        for _ in range(num_blocks):
            mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(z_dim, 2 * c_s),
            )
            nn.init.zeros_(mlp[1].weight)
            nn.init.zeros_(mlp[1].bias)
            self.adaln_mlps.append(mlp)

    def encode_posterior(self, input_feats, ensemble_trans_t, ensemble_rotmats_t, res_mask, ensemble_mask):
        """
        Args:
            input_feats: batch dict with single_embedding, t, etc.
            ensemble_trans_t: [B, K, N, 3]
            ensemble_rotmats_t: [B, K, N, 3, 3]
            res_mask: [B, N]
            ensemble_mask: [B, K]
        Returns:
            mu_q, log_sigma_q: each [B, z_dim]
        """
        # Sequence tokens with time
        seq_tokens = self.flow_model.node_embedder(
            input_feats["t"], res_mask, input_feats["single_embedding"]
        )  # [B, N, c_s]

        # Structure encode each conformer
        struct_tokens = self.structure_encoder(
            ensemble_trans_t,
            ensemble_rotmats_t,
            res_mask,
            ensemble_mask,
        )

        return self.posterior_encoder(seq_tokens, struct_tokens, res_mask, ensemble_mask)

    def encode_prior(self, input_feats, res_mask):
        """
        Args:
            input_feats: batch dict with single_embedding
            res_mask: [B, N]
        Returns:
            mu_p, log_sigma_p: each [B, z_dim]
        """
        # Time-independent sequence tokens
        seq_tokens = self.flow_model.node_embedder.single_embedder(
            input_feats["single_embedding"]
        )  # [B, N, c_s]
        return self.prior_encoder(seq_tokens, res_mask)

    @staticmethod
    def sample_z(mu, log_sigma):
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, input_feats):
        """Same interface as FlowModel.forward(). Reads latent_z from input_feats."""
        z = input_feats.get("latent_z")  # [B, z_dim] or None

        node_mask = input_feats["res_mask"]
        trans_t = input_feats["trans_t"]
        rotmats_t = input_feats["rotmats_t"]
        trans_sc = input_feats.get("trans_sc", torch.zeros_like(trans_t))

        # Reuse FlowModel's embedders
        init_node_embed = self.flow_model.node_embedder(
            input_feats["t"], node_mask, input_feats["single_embedding"]
        )
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        edge_embed = self.flow_model.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask, input_feats["pair_embedding"]
        )

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.flow_model.rigids_ang_to_nm(curr_rigids)

        trunk = self.flow_model.trunk
        num_blocks = self._ipa_conf.num_blocks

        for b in range(num_blocks):
            ipa_embed = trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)

            # AdaLN modulation
            if z is not None:
                adaln_params = self.adaln_mlps[b](z)  # [B, 2*c_s]
                gamma, beta = adaln_params.chunk(2, dim=-1)  # [B, c_s]
                node_embed = (1 + gamma[:, None, :]) * node_embed + beta[:, None, :]

            seq_tfmr_out = trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).bool()
            )
            node_embed = node_embed + trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = trunk[f"bb_update_{b}"](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None]
            )

            if b < num_blocks - 1:
                edge_embed = trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        _, pred_torsions = self.flow_model.angle_pred_net(node_embed, init_node_embed)

        curr_rigids = self.flow_model.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        return {
            "pred_torsions": pred_torsions,
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
        }
