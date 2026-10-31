"""
P11 — Foldseek-RAG model.

GVPIdAwareMHGGAT (P07) extended with retrieved-reaction context tokens.
Retrieved reactions are tokenized with the same vocab as the decoder,
embedded via the shared token embedding, projected through a zero-init
retrieval_proj, tagged with a new modality (id=2), and concatenated to
the encoder hidden state.

Encoder hidden shape:
    [B, 2 + 2L + K*R, d_model]
       g_mean, g_max, esm, gnn, retrieved_tokens

Warm-start safety
-----------------
`retrieval_proj` is zero-init → retrieved tokens contribute zero
vectors at step 0. The decoder's cross-attention still *sees* these
positions (the encoder_attention_mask has 1s for real retrieved tokens),
which means there's a small softmax-normalization perturbation vs
strict P07 forward. In practice this is bounded and the fine-tune loss
catches up within a few hundred steps. If you want hard-identity
warm-start, set `--retrieval-warmup-steps N>0` in train.py and the
mask stays 0 for the first N steps.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR  = os.path.abspath(os.path.join(_THIS_DIR, "..", "_lib"))
_GVP_DIR  = os.path.abspath(os.path.join(_THIS_DIR, "..", "03_gvp_gnn"))
_P07_DIR  = os.path.abspath(os.path.join(_THIS_DIR, "..", "07_id_curriculum"))
for p in (_LIB_DIR, _GVP_DIR, _P07_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from mhggat_id_aware import GVPIdAwareMHGGAT


class GVPRAGMHGGAT(GVPIdAwareMHGGAT):
    """
    P11 model. Adds retrieved-reaction context to P07's encoder.

    New parameters:
        retrieval_proj          Linear(d_model, d_model), zero-init
        retrieval_modality_emb  Embedding(1, d_model), zero-init
        retrieval_pos_emb       Embedding(K * R, d_model), small init
    """

    def __init__(self, num_mpnn_layers: int = 2, d_v: int = 16,
                 num_rbf: int = 16,
                 top_k: int = 5,
                 max_retrieved_tokens: int = 64):
        super().__init__(num_mpnn_layers=num_mpnn_layers, d_v=d_v, num_rbf=num_rbf)
        self.top_k = top_k
        self.max_retrieved_tokens = max_retrieved_tokens
        self.retrieved_total_len = top_k * max_retrieved_tokens

        d = self.d_model

        # Zero-init projection — retrieved tokens contribute the zero vector
        # at warm-start; layer learns to inject signal during fine-tuning.
        self.retrieval_proj = nn.Linear(d, d)
        nn.init.zeros_(self.retrieval_proj.weight)
        nn.init.zeros_(self.retrieval_proj.bias)

        # Modality embedding for retrieved tokens (id 2). Sibling to the
        # parent's `modality_emb` (which has slots 0=ESM, 1=GNN). Kept
        # separate (not extended in-place) so warm-start state_dict load
        # of P07 weights doesn't reshape the parent's modality_emb.
        self.retrieval_modality_emb = nn.Embedding(1, d)
        nn.init.zeros_(self.retrieval_modality_emb.weight)

        # Position embedding for the K*R retrieved-token slots. Small init
        # so the position channel isn't overwhelming the (zero) projected
        # token at warm-start.
        self.retrieval_pos_emb = nn.Embedding(self.retrieved_total_len, d)
        nn.init.normal_(self.retrieval_pos_emb.weight, std=0.02)

    # ------------------------------------------------------------------
    def _embed_retrieved(self,
                          retrieved_token_ids: torch.Tensor,
                          retrieved_mask: torch.Tensor):
        """
        retrieved_token_ids : [B, K, R]   long
        retrieved_mask      : [B, K, R]   float  (1 = real, 0 = pad)

        Returns:
            retrieved_hidden : [B, K*R, d]   float
            retrieved_flat_mask : [B, K*R]   float
        """
        B, K, R = retrieved_token_ids.shape
        device = retrieved_token_ids.device

        # 1. Token embedding via the shared decoder embedding.
        embeds = self.shared_embedding(retrieved_token_ids)        # [B, K, R, d]

        # 2. Zero-init projection — at start of fine-tune, returns zero.
        embeds = self.retrieval_proj(embeds)

        # 3. Position emb over the flat K*R index.
        pos_idx = torch.arange(K * R, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.retrieval_pos_emb(pos_idx).view(B, K, R, -1)
        embeds = embeds + pos_emb

        # 4. Modality (id=2) — single embedding, broadcast.
        mod = self.retrieval_modality_emb(torch.zeros(1, dtype=torch.long, device=device))
        embeds = embeds + mod.view(1, 1, 1, -1)

        # 5. Zero out pad slots (no signal from positions that have no
        #    retrieved token).
        embeds = embeds * retrieved_mask.unsqueeze(-1).to(embeds.dtype)

        # Flatten K and R into a single sequence dim.
        retrieved_hidden    = embeds.view(B, K * R, -1)            # [B, K*R, d]
        retrieved_flat_mask = retrieved_mask.view(B, K * R)        # [B, K*R]
        return retrieved_hidden, retrieved_flat_mask

    # ------------------------------------------------------------------
    def _build_encoder_hidden_rag(self,
                                   graph_x, plddt, coords, torsions,
                                   edge_index, batch_idx, protein_mask,
                                   retrieved_token_ids, retrieved_mask,
                                   retrieval_active: bool = True):
        """
        Wraps parent's _build_encoder_hidden, then concatenates retrieved
        context. retrieval_active=False zeroes the retrieved attention
        mask — effectively turns off retrieval for that forward pass
        (used for warmup).
        """
        encoder_hidden, extended_protein_mask = self._build_encoder_hidden(
            graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
        )

        retrieved_hidden, retrieved_flat_mask = self._embed_retrieved(
            retrieved_token_ids, retrieved_mask,
        )

        if not retrieval_active:
            retrieved_flat_mask = torch.zeros_like(retrieved_flat_mask)

        encoder_hidden_rag = torch.cat(
            [encoder_hidden, retrieved_hidden.to(encoder_hidden.dtype)], dim=1,
        )
        extended_mask_rag = torch.cat(
            [extended_protein_mask, retrieved_flat_mask.to(extended_protein_mask.dtype)],
            dim=1,
        )
        return encoder_hidden_rag, extended_mask_rag

    # ------------------------------------------------------------------
    def forward(self, graph_x, plddt, coords, torsions, edge_index, batch_idx,
                protein_mask, reaction_ids, reaction_mask,
                retrieved_token_ids, retrieved_mask,
                retrieval_active: bool = True):
        encoder_hidden, extended_mask = self._build_encoder_hidden_rag(
            graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
            retrieved_token_ids, retrieved_mask, retrieval_active=retrieval_active,
        )
        decoder_outputs = self.decoder(
            input_ids=reaction_ids,
            attention_mask=reaction_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=extended_mask,
            output_attentions=True,
        )
        main_logits = self.lm_head(decoder_outputs.last_hidden_state)
        # id_head reads encoder_hidden[:, 0, :] — first token is g_mean,
        # unchanged from parent. Safe to call directly.
        id_logit = self._id_logit_from_encoder(encoder_hidden)
        return main_logits, id_logit

    # ------------------------------------------------------------------
    def predict(self, graph_x, plddt, coords, torsions, edge_index, batch_idx,
                protein_mask, retrieved_token_ids, retrieved_mask,
                beam_width: int = 1, max_length: int = 250,
                length_normalization: float = 0.0, use_sampling: bool = False,
                temperature: float = 1.0, top_k: int = 5, num_samples: int = 1,
                decoder_input_ids=None, retrieval_active: bool = True):
        from model import top_k_batch_sampling, beam_search
        self.eval()
        device = graph_x.device
        with torch.no_grad():
            encoder_hidden, extended_mask = self._build_encoder_hidden_rag(
                graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
                retrieved_token_ids, retrieved_mask,
                retrieval_active=retrieval_active,
            )
            id_logit = self._id_logit_from_encoder(encoder_hidden)

        if use_sampling:
            preds, _ = top_k_batch_sampling(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                top_k=top_k, max_length=max_length, temperature=temperature,
                input_ids=encoder_hidden,
                attention_mask=extended_mask,
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
                input_ids=encoder_hidden,
                attention_mask=extended_mask,
                device=device, initial_tokens=decoder_input_ids,
            )
        return preds, id_logit
