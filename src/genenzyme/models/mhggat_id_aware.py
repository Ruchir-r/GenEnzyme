"""
GVP backbone + identity-reaction auxiliary head.

Empirical motivation (from per-side analysis of the 5-way infer CSV):
~4.4% of test samples are identity reactions where reactants == products
(transporters, chaperones, simple chloride/iodide pass-throughs). All
current models score ~0.7 mean Tanimoto on these vs ~0.93 elsewhere —
because the decoder hallucinates a plausible enzymatic transformation
where there is none.

Mitigation: train a binary classifier on top of the structural encoder's
g_mean token. Output P(reactants == products). Train jointly with the
main reaction-prediction loss (BCE on identity-vs-not + CE on tokens).
At inference, if P > 0.5, override the predicted reaction by copying
the predicted reactants verbatim into the products slot. Cheap potential
~+1 pt mean Tanimoto across the test set.

API
---
forward()  returns a tuple (main_logits, id_logit). Train script unpacks.
predict()  returns a tuple (token_ids, id_logit). Inference script can
           apply the optional override.
decode_with_identity_override()  utility that does the override on
           predicted SMILES strings.

Reuses GVPMHGGAT's encoder via inheritance — only the head and the
forward/predict bookkeeping change. Backwards compatible: if you ignore
the id_logit, behavior is identical to GVPMHGGAT.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "_lib"))
_GVP_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "03_gvp_gnn"))
for p in (_LIB_DIR, _GVP_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from mhggat_gvp import GVPMHGGAT


class GVPIdAwareMHGGAT(GVPMHGGAT):
    """
    GVPMHGGAT extended with a binary classifier head on top of the
    encoder's g_mean token. Predicts whether the enzyme catalyzes an
    identity reaction (R==P, transporter/chaperone-like).
    """

    def __init__(self, num_mpnn_layers: int = 2, d_v: int = 16, num_rbf: int = 16):
        super().__init__(num_mpnn_layers=num_mpnn_layers, d_v=d_v, num_rbf=num_rbf)

        # +id_head: 1-class classifier reading off the encoder's first
        # global token (g_mean). Init small so warm-start from plain GVP
        # gives near-zero auxiliary signal at start. Not strictly zero-init
        # (we *want* gradient to flow into this head from epoch 0), but
        # bounded so the joint loss isn't dominated by random noise initially.
        self.id_head = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.id_head.weight, std=0.01)
        nn.init.zeros_(self.id_head.bias)

    # ------------------------------------------------------------------
    def _id_logit_from_encoder(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        encoder_hidden: [B, 2L+2, d_model]   (parent's GVP layout)
        Returns id_logit: [B, 1]   raw logit, no sigmoid.

        We use the first encoder token (`g_mean`) as the global protein
        summary for identity classification. g_max would also work but
        is more sensitive to outlier residues; mean is steadier for a
        whole-protein property like "is this a transporter".
        """
        g_mean_token = encoder_hidden[:, 0, :]                   # [B, d_model]
        return self.id_head(g_mean_token.to(self.id_head.weight.dtype))

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
        main_logits = self.lm_head(decoder_outputs.last_hidden_state)
        id_logit    = self._id_logit_from_encoder(encoder_hidden)
        return main_logits, id_logit

    # ------------------------------------------------------------------
    def predict(self, graph_x, plddt, coords, torsions, edge_index, batch_idx,
                protein_mask, beam_width: int = 1, max_length: int = 250,
                length_normalization: float = 0.0, use_sampling: bool = False,
                temperature: float = 1.0, top_k: int = 5, num_samples: int = 1,
                decoder_input_ids=None, logits_processor=None):
        """
        Returns a (predicted_token_ids, id_logit) tuple. Inference scripts
        that don't care about identity can just take preds[0].
        """
        from model import top_k_batch_sampling, beam_search
        self.eval()
        device = graph_x.device
        with torch.no_grad():
            encoder_hidden_states, extended_protein_mask = self._build_encoder_hidden(
                graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask,
            )
            id_logit = self._id_logit_from_encoder(encoder_hidden_states)

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
                logits_processor=logits_processor,
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
                logits_processor=logits_processor,
            )
        return preds, id_logit


# =============================================================================
# Inference utility — apply the identity override on decoded SMILES strings
# =============================================================================
def decode_with_identity_override(pred_smiles: list[str],
                                  id_logits: torch.Tensor,
                                  threshold: float = 0.0) -> list[str]:
    """
    For each sample: if id_logit > threshold (i.e. P(identity) > sigmoid(thresh)),
    replace the prediction with `<reactants>>><reactants>`, where reactants
    are taken from the model's own predicted output (everything before the
    first ">>"). If the predicted SMILES has no ">>" or is empty, leave as-is.

    Args:
        pred_smiles : list of decoded prediction strings, length B
        id_logits   : [B, 1] tensor of raw logits
        threshold   : raw-logit threshold (default 0 -> P > 0.5)

    Returns:
        list of strings with the override applied where appropriate.
    """
    out = []
    flags = (id_logits.squeeze(-1) > threshold).tolist()
    for smi, is_id in zip(pred_smiles, flags):
        if is_id and isinstance(smi, str) and ">>" in smi:
            reactants = smi.split(">>", 1)[0]
            out.append(f"{reactants}>>{reactants}")
        else:
            out.append(smi)
    return out


# =============================================================================
# Smoke test
# =============================================================================
def _self_test() -> None:
    """
    Forward+backward on a 2-protein debug batch.
    Verifies:
      - forward returns a (main_logits, id_logit) tuple of correct shape
      - id_logit gets gradient when we add a BCE loss
      - decode_with_identity_override does the right replacement
    """
    debug_csv = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "debug_train.csv"))
    vocab     = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "vocab.json"))
    embeds    = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "local_embeds"))
    structs   = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "local_structs"))
    if not all(os.path.exists(p) for p in (debug_csv, vocab, embeds, structs)):
        print("[mhggat_id_aware] debug data missing; skipping smoke test.")
        return

    import config as _config
    _config.DATA_CONFIG['pdb_structures'] = structs + os.sep

    from torch.utils.data import DataLoader
    from data_curriculum import CurriculumGraphReactionDataset

    ds = CurriculumGraphReactionDataset(debug_csv, vocab, embeds)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=ds.collate_fn, num_workers=0)
    batch = next(iter(loader))

    model = GVPIdAwareMHGGAT(num_mpnn_layers=2, d_v=16)
    model.train()
    main_logits, id_logit = model(
        graph_x=batch["graph_x"], plddt=batch["plddt"],
        coords=batch["coords"], torsions=batch["torsions"],
        edge_index=batch["edge_index"], batch_idx=batch["batch_idx"],
        protein_mask=batch["protein_mask"],
        reaction_ids=batch["reaction_ids"],
        reaction_mask=batch["reaction_mask"],
    )

    assert main_logits.dim() == 3, main_logits.shape
    B = batch["is_identity"].shape[0]
    assert id_logit.shape == (B, 1), id_logit.shape
    print(f"[mhggat_id_aware] forward OK | "
          f"main {tuple(main_logits.shape)} | id {tuple(id_logit.shape)}")

    # Backward — combine main and identity losses, check both flow.
    loss_main = main_logits.mean()
    loss_id = nn.functional.binary_cross_entropy_with_logits(
        id_logit, batch["is_identity"]
    )
    (loss_main + 0.1 * loss_id).backward()
    assert model.id_head.weight.grad is not None, "id_head got no gradient"
    assert (model.id_head.weight.grad.abs().sum() > 0), "id_head gradient is exactly zero"
    print(f"[mhggat_id_aware] backward OK | id_head.grad_norm = "
          f"{model.id_head.weight.grad.norm().item():.3e}")

    # decode_with_identity_override behavior
    smis = ["CCO>>CC=O", "FOO>>BAR", "no_arrow", ""]
    logits = torch.tensor([[2.0], [-2.0], [3.0], [3.0]])  # 1st & 3rd & 4th -> identity
    out = decode_with_identity_override(smis, logits, threshold=0.0)
    assert out == ["CCO>>CCO", "FOO>>BAR", "no_arrow", ""], out
    print("[mhggat_id_aware] decode_with_identity_override OK")


if __name__ == "__main__":
    _self_test()
