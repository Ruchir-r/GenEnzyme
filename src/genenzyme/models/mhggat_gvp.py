"""
GVP-backed variant of MHGGAT.

Replaces the parent's hand-rolled featurizer + MultiHeadGeometricGAT loop
with an SE(3)-equivariant GVP featurizer + GVPConv message passing. Two
goals:

  1. Drop the bf16 numerical instability in `rot_to_quat` (parent's path).
  2. Use a properly equivariant geometric encoder so the structural channel
     can learn from rotated/translated proteins consistently.

What's reused vs replaced
-------------------------
Reused (same parameter names, so warm-starting from mhggat_full's
`best_geometric_model.pt` via load_state_dict(strict=False) loads them):
  - shared_embedding, decoder, lm_head, decoder_config
  - esm_proj
  - modality_emb, seq_pos_emb, gnn_gate, gnn_final_norm

Replaced (these keys WON'T exist in the parent ckpt; new ones are random
init, so the GNN stream starts effectively zeroed via the parent's
zero-init `gnn_gate`):
  - featurizer (ProteinGeometricFeaturizer  -> ProteinGVPFeaturizer)
  - gat_layers (MultiHeadGeometricGAT       -> GVPConv)
  - gnn_norms  (GraphNorm                   -> GVPLayerNorm, lives inside GVPConv)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import os as _os
_NO_STRUCT = _os.environ.get("RXN_NO_STRUCT", "0") == "1"
if _NO_STRUCT:
    print("[mhggat_gvp] RXN_NO_STRUCT=1 -- GNN pathway zeroed (sequence-only ablation)", flush=True)
import torch.nn.functional as F

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
for p in (_SRC_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_geometric.nn import global_mean_pool, global_max_pool

from mhggat import GraphReactionPredictor as _BaseMHGGAT
from config import SPECIAL_TOKENS, TRAIN_CONFIG, MODEL_CONFIG
from model import top_k_batch_sampling, beam_search

from gvp_layers import GVP, GVPConv, GVPLayerNorm, _vector_norm


# =============================================================================
# Featurizer: coords -> (node_s, node_v, edge_s, edge_v)
# =============================================================================
class _RBF(nn.Module):
    """Gaussian basis expansion: distance -> [..., num_bins]."""
    def __init__(self, v_min=0.0, v_max=20.0, num_bins=16):
        super().__init__()
        centers = torch.linspace(v_min, v_max, num_bins)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0])

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).pow(2))


class ProteinGVPFeaturizer(nn.Module):
    """
    Build (s, v) features for the GVPConv backbone from coords/torsions.

    Node scalars  (per residue): pLDDT (1) + sin/cos torsions (6) + RBF
                                 of d_NCA, d_CCA, d_OCA, d_RCA (4 * num_rbf)
    Node vectors  (per residue): unit vectors CA->N, CA->C, CA->O, CA->R
                                 in the GLOBAL frame (3 vectors x 3 dims). Equivariance
                                 is enforced inside the GVP layers (channel-only linear
                                 maps + magnitude gates), not here.
    Edge scalars  (per edge):   RBF of |CA_src - CA_dst|.
    Edge vectors  (per edge):   single unit vector from CA_src to CA_dst.

    Outputs are projected to (d_s, d_v) via a single GVP so downstream
    message-passing layers can stay at one fixed channel width.
    """
    def __init__(self, d_s: int, d_v: int, num_rbf: int = 16):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf = _RBF(num_bins=num_rbf)

        # raw counts before projection
        # nodes: 1 (plddt) + 6 (sin/cos torsions) + 4*num_rbf
        self.raw_node_s_dim = 1 + 6 + 4 * num_rbf
        self.raw_node_v_dim = 4   # CA->{N,C,O,R}
        # edges: num_rbf scalars + 1 vector channel
        self.raw_edge_s_dim = num_rbf
        self.raw_edge_v_dim = 1

        # Single GVP to project raw features into the d_s/d_v width used
        # downstream. Acts like a learned input embedding.
        self.node_proj = GVP((self.raw_node_s_dim, self.raw_node_v_dim),
                             (d_s, d_v), scalar_act=nn.ReLU)
        self.edge_proj = GVP((self.raw_edge_s_dim, self.raw_edge_v_dim),
                             (d_s, d_v), scalar_act=nn.ReLU)

    # ------- helpers
    @staticmethod
    def _safe_norm_dir(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        n = vec.norm(dim=-1, keepdim=True).clamp(min=eps)
        return vec / n

    def _node_scalars(self, plddt, torsions, dN, dC, dO, dR) -> torch.Tensor:
        plddt_s = plddt / 100.0    # [N, 1] in [0, 1]
        torsion_s = torch.cat([torch.sin(torsions), torch.cos(torsions)], dim=-1)   # [N, 6]
        dist_rbf = torch.cat([self.rbf(dN), self.rbf(dC),
                              self.rbf(dO), self.rbf(dR)], dim=-1)   # [N, 4*num_rbf]
        return torch.cat([plddt_s, torsion_s, dist_rbf], dim=-1)     # [N, raw_node_s_dim]

    def forward(self, coords: dict, plddt: torch.Tensor,
                torsions: torch.Tensor, edge_index: torch.Tensor):
        """
        coords: dict with keys N, CA, C, O, R; each [N_total, 3]
        plddt:  [N_total, 1]   (B-factor /100 inside this fn)
        torsions: [N_total, 3]   (phi, psi, omega; radians)
        edge_index: [2, E]

        Returns (node_s, node_v, edge_s, edge_v) at width (d_s, d_v).
        """
        N = coords["N"].float(); CA = coords["CA"].float()
        C = coords["C"].float(); O  = coords["O"].float(); R = coords["R"].float()
        torsions = torsions.float()

        # --- Node features
        N_dir = self._safe_norm_dir(N - CA)   # [N_total, 3]
        C_dir = self._safe_norm_dir(C - CA)
        O_dir = self._safe_norm_dir(O - CA)
        R_dir = self._safe_norm_dir(R - CA)
        node_v_raw = torch.stack([N_dir, C_dir, O_dir, R_dir], dim=1)   # [N_total, 4, 3]

        node_s_raw = self._node_scalars(
            plddt, torsions,
            (N - CA).norm(dim=-1), (C - CA).norm(dim=-1),
            (O - CA).norm(dim=-1), (R - CA).norm(dim=-1),
        )   # [N_total, raw_node_s_dim]

        node_s, node_v = self.node_proj(node_s_raw, node_v_raw)

        # --- Edge features
        if edge_index.numel() == 0:
            E = 0
            edge_s = node_s.new_zeros(0, node_s.shape[-1])
            edge_v = node_s.new_zeros(0, node_v.shape[1], 3)
            return node_s, node_v, edge_s, edge_v

        src, dst = edge_index[0], edge_index[1]
        d_ca = (CA[dst] - CA[src]).norm(dim=-1)         # [E]
        edge_s_raw = self.rbf(d_ca)                      # [E, num_rbf]
        edge_v_raw = self._safe_norm_dir(CA[dst] - CA[src]).unsqueeze(1)   # [E, 1, 3]
        edge_s, edge_v = self.edge_proj(edge_s_raw, edge_v_raw)

        return node_s, node_v, edge_s, edge_v


# =============================================================================
# The model: MHGGAT with GVP backbone
# =============================================================================
class GVPMHGGAT(_BaseMHGGAT):
    """
    Inherits the parent for the dual-sequence encoder and decoder. Replaces
    the structural pathway with GVP. The parent's `gnn_gate` (already
    zero-init) is reused — at step 0 the GNN stream contributes zero,
    so the model behaves like sequence-only and val-loss matches the
    parent's "ESM-only" trajectory. As gnn_gate fades in, the new GVP
    stream takes over.
    """
    def __init__(self,
                 num_mpnn_layers: int = 2,
                 d_s: int = None,
                 d_v: int = 16,
                 num_rbf: int = 16):
        # Parent's __init__ creates the structural modules we're about to
        # replace; that's wasteful but harmless (just dropped objects).
        # The alternative is to copy parent __init__ wholesale; not worth it.
        super().__init__(num_mpnn_layers=num_mpnn_layers)

        if d_s is None:
            d_s = self.d_model     # match parent's per-residue width

        # Replace structural modules
        del self.featurizer
        del self.gat_layers
        del self.gnn_norms

        self.featurizer = ProteinGVPFeaturizer(d_s=d_s, d_v=d_v, num_rbf=num_rbf)
        self.gat_layers = nn.ModuleList([
            GVPConv(node_dims=(d_s, d_v), edge_dims=(d_s, d_v),
                    n_msg_layers=2, dropout=TRAIN_CONFIG["dropout"])
            for _ in range(num_mpnn_layers)
        ])
        self.d_s = d_s
        self.d_v = d_v

        # Final scalar projection: cat(s, |v|_per_channel) -> d_model.
        # We keep d_s == d_model so the cat is 2 * d_model -> d_model.
        # NOTE: not zero-initialized. The parent's `gnn_gate` (already
        # zero-init) suppresses the per-residue GNN stream at start; the
        # global pools (g_mean, g_max) DO contribute from step 0, but
        # since they're random projections at init, they look like noise
        # to the decoder. The decoder's cross-attention will learn to
        # ignore them until `gnn_gate` rises and the per-residue path
        # kicks in. Zero-initing here would kill upstream gradient flow
        # to the GVP layers entirely.
        self.gnn_to_dmodel = nn.Linear(d_s + d_v, self.d_model)

    # ------------------------------------------------------------------
    def _build_encoder_hidden(self, graph_x, plddt, coords, torsions,
                              edge_index, batch_idx, protein_mask):
        """Mirrors steps 1-5 of parent forward, with GVP in step 2."""
        device = graph_x.device
        batch_size = protein_mask.shape[0]
        max_protein_len = protein_mask.shape[1]

        # 1. ESM
        x_esm = self.esm_proj(torch.cat([graph_x, plddt / 100.0], dim=-1))

        # 2. GVP message passing
        node_s, node_v, edge_s, edge_v = self.featurizer(coords, plddt, torsions, edge_index)

        if edge_index.numel() > 0:
            for conv in self.gat_layers:
                node_s, node_v = conv(node_s, node_v, edge_index, edge_s, edge_v)

        # Combine s and |v| into a single scalar tensor for downstream mixing.
        x_gnn = self.gnn_to_dmodel(torch.cat([node_s, _vector_norm(node_v)], dim=-1))   # [N, d_model]

        # Structure ablation (plan 3A). RXN_NO_STRUCT=1 zeroes the GNN pathway, which is the
        # ONLY source of structural information in this encoder: g_mean, g_max and gnn_hidden
        # are all derived from x_gnn below, so zeroing it here removes structure from every
        # channel the decoder can attend to. The ESM pathway (x_esm) is untouched, and the
        # positional/modality embeddings added to gnn_hidden are input-independent constants.
        # The GNN PARAMETERS still exist but receive no gradient, so this isolates structural
        # INFORMATION at fixed decoder capacity rather than removing the module.
        if _NO_STRUCT:
            x_gnn = torch.zeros_like(x_gnn)

        # 3. Global tokens (parent style)
        g_mean = global_mean_pool(x_gnn, batch_idx).unsqueeze(1)
        g_max  = global_max_pool(x_gnn, batch_idx).unsqueeze(1)

        # 4. Align to dense per-residue tensors
        esm_hidden = torch.zeros(batch_size, max_protein_len, self.d_model,
                                 device=device, dtype=x_esm.dtype)
        gnn_hidden = torch.zeros(batch_size, max_protein_len, self.d_model,
                                 device=device, dtype=x_gnn.dtype)
        counts = torch.bincount(batch_idx)
        local_idx = torch.cat([torch.arange(c, device=device) for c in counts])

        # Note on dtypes: under torch.autocast(bf16) on CUDA, nn.LayerNorm is
        # on the "promote to fp32" list, so `gnn_final_norm(x_gnn)` returns
        # fp32 even when x_gnn is bf16. We cast on assignment so destination
        # buffers (which we allocated to match x_gnn's bf16 dtype above)
        # accept the value cleanly.
        esm_hidden[batch_idx, local_idx] = x_esm.to(esm_hidden.dtype)
        x_gnn_normed = self.gnn_final_norm(x_gnn)
        gated = x_gnn_normed * self.gnn_gate.to(x_gnn_normed.dtype)
        gnn_hidden[batch_idx, local_idx] = gated.to(gnn_hidden.dtype)

        # 5. Modality + position tags
        positions = torch.arange(max_protein_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.seq_pos_emb(positions)
        esm_hidden = esm_hidden + pos_emb + self.modality_emb(torch.tensor(0, device=device))
        gnn_hidden = gnn_hidden + pos_emb + self.modality_emb(torch.tensor(1, device=device))

        encoder_hidden = torch.cat([g_mean, g_max, esm_hidden, gnn_hidden], dim=1)
        global_mask = torch.ones((batch_size, 2), dtype=protein_mask.dtype, device=device)
        extended_protein_mask = torch.cat([global_mask, protein_mask, protein_mask], dim=1)
        return encoder_hidden, extended_protein_mask

    # ------------------------------------------------------------------
    def forward(self, graph_x, plddt, coords, torsions, edge_index, batch_idx,
                protein_mask, reaction_ids, reaction_mask):
        encoder_hidden, extended_protein_mask = self._build_encoder_hidden(
            graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
        )
        decoder_outputs = self.decoder(
            input_ids=reaction_ids,
            attention_mask=reaction_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=extended_protein_mask,
            output_attentions=True,
        )
        return self.lm_head(decoder_outputs.last_hidden_state)

    # ------------------------------------------------------------------
    def predict(self, graph_x, plddt, coords, torsions, edge_index, batch_idx,
                protein_mask, beam_width: int = 1, max_length: int = 250,
                length_normalization: float = 0.0, use_sampling: bool = False,
                temperature: float = 1.0, top_k: int = 5, num_samples: int = 1,
                decoder_input_ids=None):
        self.eval()
        device = graph_x.device
        with torch.no_grad():
            encoder_hidden_states, extended_protein_mask = self._build_encoder_hidden(
                graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
            )

        if use_sampling:
            preds, _ = top_k_batch_sampling(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                top_k=top_k, max_length=max_length, temperature=temperature,
                input_ids=encoder_hidden_states,
                attention_mask=extended_protein_mask,
                device=device, num_samples=num_samples,
                initial_tokens=decoder_input_ids,
            )
        else:
            preds, _, _, _, _ = beam_search(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                beam_width=beam_width, max_length=max_length,
                length_normalization=length_normalization,
                input_ids=encoder_hidden_states,
                attention_mask=extended_protein_mask,
                device=device, initial_tokens=decoder_input_ids,
            )
        return preds


# =============================================================================
# Smoke test: build a 2-protein batch from debug data, forward+backward.
# =============================================================================
def _self_test():
    import os, sys
    debug_csv = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "debug_train.csv"))
    vocab     = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "vocab.json"))
    embeds    = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "local_embeds"))
    structs   = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "local_structs"))
    if not all(os.path.exists(p) for p in (debug_csv, vocab, embeds, structs)):
        print("[mhggat_gvp] debug data missing; skipping smoke test.")
        return

    import config as _config
    _config.DATA_CONFIG['pdb_structures'] = structs + os.sep

    from data_featurizer import GraphReactionDataset
    from torch.utils.data import DataLoader

    ds = GraphReactionDataset(debug_csv, vocab, embeds)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=ds.collate_fn, num_workers=0)
    batch = next(iter(loader))

    model = GVPMHGGAT(num_mpnn_layers=2, d_v=16)
    model.train()
    logits = model(
        graph_x=batch["graph_x"], plddt=batch["plddt"],
        coords=batch["coords"], torsions=batch["torsions"],
        edge_index=batch["edge_index"], batch_idx=batch["batch_idx"],
        protein_mask=batch["protein_mask"],
        reaction_ids=batch["reaction_ids"],
        reaction_mask=batch["reaction_mask"],
    )
    assert logits.dim() == 3, logits.shape
    print(f"[mhggat_gvp] forward OK | logits shape={tuple(logits.shape)}")

    # Backward — check gnn_gate gradient flows.
    loss = logits.mean()
    loss.backward()
    assert model.gnn_gate.grad is not None, "gnn_gate got no gradient"
    # GVP layer parameters should also have gradients
    any_gvp_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if "gat_layers" in n or "featurizer" in n
    )
    assert any_gvp_grad, "no gradient flowed into GVP layers"
    print("[mhggat_gvp] backward OK | GVP grads present, gnn_gate grad present")


if __name__ == "__main__":
    _self_test()

