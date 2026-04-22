"""
Neural network for embedding node features.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/node_embedder.py
"""

import torch
from torch import nn
from rna_backbone_design.models import utils


class NuFiLM(nn.Module):
    """log(nu) -> (gamma, beta) FiLM. Zero-init weight + bias = identity at start.

    Applied to the projected single embedding so the trunk sees the per-residue
    noise level. log(nu)=0 (i.e. dataset-average flexibility) maps to no modulation.
    """

    def __init__(self, c_out: int):
        super().__init__()
        self.proj = nn.Linear(1, 2 * c_out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, log_nu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(log_nu.unsqueeze(-1)).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class NodeEmbedder(nn.Module):
    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.c_single_in = self._cfg.c_single_in

        self.single_embedder = nn.Sequential(
            nn.LayerNorm(self.c_single_in),
            nn.Linear(self.c_single_in, self.c_s, bias=False),
        )

        if getattr(self._cfg, "use_nu_film", True):
            self.nu_film = NuFiLM(self.c_s)

        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self.c_s, self.c_s
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = utils.get_time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, single_embedding, nu=None):
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = utils.get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        single_emb = self.single_embedder(single_embedding)
        single_emb = single_emb * mask.unsqueeze(-1)
        if nu is not None:
            log_nu = torch.log(nu.clamp(min=1e-8))
            single_emb = self.nu_film(log_nu, single_emb) * mask.unsqueeze(-1)

        input_feats = [pos_emb, self.embed_t(timesteps, mask), single_emb]
        return self.linear(torch.cat(input_feats, dim=-1))
