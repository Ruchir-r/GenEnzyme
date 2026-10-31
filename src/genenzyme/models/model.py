import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartDecoder, BartEncoder
from transformers.cache_utils import EncoderDecoderCache
from torch_geometric.nn import GATv2Conv, GraphNorm

# Assuming these are available in your project structure
from config import TRAIN_CONFIG, SPECIAL_TOKENS, MODEL_CONFIG

# ⚠️ Import your sampling function here if you use it!
# from your_custom_module import top_k_batch_sampling 

def top_k_batch_sampling(
    model,
    bos_token_id,
    eos_token_id,
    pad_token_id,
    top_k,
    max_length,
    temperature,
    input_ids,
    attention_mask,
    device,
    num_samples=1,
    initial_tokens=None,
):

    model.eval()
    torch.set_grad_enabled(False)

    batch_size = input_ids.size(0)

    # Expand inputs for multiple samples per input
    encoder_hidden_states = input_ids.repeat_interleave(num_samples, dim=0)
    encoder_mask = attention_mask.repeat_interleave(num_samples, dim=0)

    # Initialize generation
    if initial_tokens is not None:
        # If initial tokens provided, start with them (prepending BOS token)
        initial_seq = torch.cat(
            [
                torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
                initial_tokens,
            ],
            dim=1,
        )
        decoder_input_ids = _repeat_tensor_for_beams(initial_seq, num_samples, dim=0)
    else:
        decoder_input_ids = torch.full(
            (batch_size * num_samples, 1), bos_token_id, dtype=torch.long, device=device
        )
    decoder_attention_mask = torch.ones_like(decoder_input_ids, device=device)

    # Track finished sequences
    is_finished = torch.zeros(batch_size * num_samples, dtype=torch.bool, device=device)
    scores = torch.zeros(batch_size * num_samples, device=device)

    # Generate tokens
    for cur_len in range(1, max_length + 1):
        with torch.no_grad():
            # Get predictions
            outputs = model._decoder_forward(
                reaction_ids=decoder_input_ids,
                reaction_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_mask,
                use_cache=True,
                return_attention=True,
            )

            # Get logits for next token
            logits = model.lm_head(outputs.last_hidden_state[:, -1])
            logits = logits / temperature

            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(
                logits, min(top_k, logits.size(-1))
            )
            probs = torch.softmax(top_k_logits, dim=-1)

            # Sample next tokens
            next_token_indices = torch.multinomial(probs, num_samples=1)
            next_tokens = torch.gather(top_k_indices, -1, next_token_indices)

            # Update scores with log probabilities
            next_scores = torch.gather(probs, -1, next_token_indices)
            scores = scores + (~is_finished).float() * torch.log(next_scores).squeeze(
                -1
            )

            # Append tokens
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            decoder_attention_mask = torch.cat(
                [decoder_attention_mask, torch.ones_like(next_tokens)], dim=-1
            )

            # Update finished state
            is_finished = is_finished | (next_tokens.squeeze(-1) == eos_token_id)

            # Stop if all sequences are finished
            if is_finished.all():
                break

    # Reshape outputs for multiple samples
    decoder_input_ids = decoder_input_ids.view(batch_size, num_samples, -1)
    scores = scores.view(batch_size, num_samples)

    return decoder_input_ids, scores

# -----------------------
# GENERATION UTILITIES
# -----------------------
def _repeat_tensor_for_beams(tensor, beam_width, dim=0):
    """
    Repeats an input tensor along a specific dimension to match the beam width.
    E.g., for dim=0: (batch, seq_len) -> (batch * beam_width, seq_len)
    """
    if tensor is None:
        return None
    return tensor.repeat_interleave(beam_width, dim=dim)

def beam_search(
    model,
    bos_token_id,
    eos_token_id,
    pad_token_id,
    beam_width,
    max_length,
    length_normalization,
    input_ids,
    attention_mask,
    device,
    initial_tokens=None,
    full_input=False,
):
    assert max_length >= 1
    assert beam_width >= 1
    assert length_normalization >= 0.0

    model.eval()
    torch.set_grad_enabled(False)

    batch_size = input_ids.size(0)
    dim0_indexer = torch.arange(batch_size, device=device)

    flat_encoder_hidden_states = _repeat_tensor_for_beams(input_ids, beam_width, dim=0)
    flat_encoder_mask = _repeat_tensor_for_beams(attention_mask, beam_width, dim=0)

    past_key_values = None

    if initial_tokens is not None:
        initial_seq = torch.cat(
            [torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device), initial_tokens], dim=1
        )
        decoder_input_ids = _repeat_tensor_for_beams(initial_seq, beam_width, dim=0)
    else:
        decoder_input_ids = torch.full((batch_size * beam_width, 1), bos_token_id, dtype=torch.long, device=device)
    
    decoder_attention_mask = torch.ones_like(decoder_input_ids, device=device)

    cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
    mask = torch.arange(beam_width, device=device).repeat(batch_size) != 0
    cum_log_probs[mask] = float("-inf")

    is_finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)
    lengths = torch.ones(batch_size * beam_width, dtype=torch.long, device=device)

    seq_len = 0 if initial_tokens is None else initial_tokens.size(1)
    target_len = max_length + 1 if not full_input else seq_len + 2
    
    for cur_len in range(seq_len + 1, target_len):
        model.eval()
        with torch.no_grad():
            if past_key_values is not None:
                current_input_ids = decoder_input_ids[:, -1:]
            else:
                current_input_ids = decoder_input_ids

            decoder_outputs = model._decoder_forward(
                reaction_ids=current_input_ids,
                reaction_mask=decoder_attention_mask,
                encoder_hidden_states=flat_encoder_hidden_states,
                encoder_attention_mask=flat_encoder_mask,
                use_cache=True,
                return_attention=True,
                past_key_values=past_key_values, 
            )

            hidden_states = decoder_outputs.last_hidden_state
            logits = model.lm_head(hidden_states)

        past_key_values = decoder_outputs.past_key_values
        logits = logits[:, -1]  
        log_probs = logits.log_softmax(-1)

        log_probs[is_finished, :] = float("-inf")
        log_probs[is_finished, pad_token_id] = 0.0

        vocab_size = logits.size(-1)
        next_scores = cum_log_probs.unsqueeze(-1) + log_probs
        finished_scores = next_scores[is_finished, pad_token_id]
        next_scores[is_finished, pad_token_id] = float("inf")

        _, topk_indices = next_scores.view(batch_size, beam_width * vocab_size).topk(beam_width, -1)
        next_scores[is_finished, pad_token_id] = finished_scores
        topk_scores = next_scores.view(batch_size, beam_width * vocab_size)[dim0_indexer.unsqueeze(-1), topk_indices]
        cum_log_probs.copy_(topk_scores.view(-1))

        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        offset = dim0_indexer * beam_width
        gather_beam_idx = (offset.unsqueeze_(1) + beam_indices).view(-1)

        decoder_input_ids = decoder_input_ids[gather_beam_idx]
        is_finished = is_finished[gather_beam_idx]
        lengths = lengths[gather_beam_idx]

        # --- Replace this section in beam_search ---
        if past_key_values is not None:
            # If it's the new EncoderDecoderCache/DynamicCache object
            if hasattr(past_key_values, "reorder_cache"):
                past_key_values.reorder_cache(gather_beam_idx)
            # Fallback for legacy tuple-based cache
            else:
                new_past_key_values = []
                for layer_cache in past_key_values:
                    new_layer_cache = []
                    for kv_tensor in layer_cache:
                        new_layer_cache.append(kv_tensor[gather_beam_idx])
                    new_past_key_values.append(tuple(new_layer_cache))
                past_key_values = tuple(new_past_key_values)

        next_tokens = token_indices.view(-1)
        decoder_input_ids = torch.cat((decoder_input_ids, next_tokens.unsqueeze(1)), 1)
        decoder_attention_mask = torch.cat((decoder_attention_mask, torch.ones_like(next_tokens.unsqueeze(1))), 1)

        is_finished |= next_tokens == eos_token_id

        if is_finished.all():
            break

        if cur_len < max_length:
            lengths += (~is_finished).long()

    seq_len = decoder_input_ids.size(1) - 1  
    decoder_attention_mask = (torch.arange(seq_len, device=device) < lengths.unsqueeze(1)).long()

    decoder_input_ids = decoder_input_ids[:, 1:]
    decoder_input_ids = decoder_input_ids.view(batch_size, beam_width, seq_len)
    decoder_attention_mask = decoder_attention_mask.view(batch_size, beam_width, seq_len)

    topk_norm_scores = cum_log_probs / lengths.pow(length_normalization)
    topk_norm_scores = topk_norm_scores.view(batch_size, beam_width)
    best_norm_scores, j = topk_norm_scores.topk(beam_width, dim=1)

    best_decoder_input_ids = decoder_input_ids[dim0_indexer.unsqueeze(1), j]
    best_decoder_attention_mask = decoder_attention_mask[dim0_indexer.unsqueeze(1), j]
    
    return (
        best_decoder_input_ids,
        best_decoder_attention_mask,
        best_norm_scores,
        decoder_outputs.cross_attentions,
        decoder_outputs.attentions,
    )


# -----------------------
# SEQUENCE MODEL
# -----------------------
class ReactionPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        vocab_size = TRAIN_CONFIG["vocab_size"]
        d_model = TRAIN_CONFIG["d_model"]
        nhead = TRAIN_CONFIG["nhead"]
        num_encoder_layers = TRAIN_CONFIG["num_encoder_layers"]
        num_decoder_layers = TRAIN_CONFIG["num_decoder_layers"]
        encoder_ffn_dim = TRAIN_CONFIG["encoder_ffn_dim"]
        decoder_ffn_dim = TRAIN_CONFIG["decoder_ffn_dim"]
        dropout = TRAIN_CONFIG["dropout"]
        encoder_dim = MODEL_CONFIG["input_dim"]
        
        pad_idx = SPECIAL_TOKENS["<PAD>"]

        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.encoder_projector = nn.Linear(encoder_dim, d_model)

        if num_encoder_layers > 0:
            encoder_config = BartConfig(
                vocab_size=vocab_size, d_model=d_model,
                encoder_layers=num_encoder_layers,
                encoder_attention_heads=nhead,
                encoder_ffn_dim=encoder_ffn_dim,
                dropout=dropout,
            )
            self.encoder = BartEncoder(encoder_config)
        else:
            self.encoder = None

        self.decoder_config = BartConfig(
            vocab_size=vocab_size, d_model=d_model,
            decoder_layers=num_decoder_layers,
            decoder_attention_heads=nhead,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            bos_token_id=SPECIAL_TOKENS["<START>"],
            eos_token_id=SPECIAL_TOKENS["<END>"],
            pad_token_id=pad_idx,
        )

        self.decoder = BartDecoder(self.decoder_config)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

    def forward(self, protein_embedding, protein_mask, reaction_ids, reaction_mask, labels=None):
        x = self.encoder_projector(protein_embedding)
        if self.encoder is not None:
            x = self.encoder(inputs_embeds=x, attention_mask=protein_mask, return_dict=True).last_hidden_state
        
        decoder_outputs = self._decoder_forward(
            reaction_ids=reaction_ids,
            reaction_mask=reaction_mask,
            encoder_hidden_states=x,
            encoder_attention_mask=protein_mask
        )
        return self.lm_head(decoder_outputs.last_hidden_state)

    def _decoder_forward(
        self,
        reaction_ids,
        reaction_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        return_attention=False,
        use_cache=False,
        past_key_values=None,
    ):
        if past_key_values is not None and not isinstance(past_key_values, EncoderDecoderCache):
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            
        context = torch.no_grad() if not self.training else torch.enable_grad()
        
        with context:
            decoder_embeds = self.shared_embedding(reaction_ids)
            decoder_outputs = self.decoder(
                attention_mask=reaction_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                inputs_embeds=decoder_embeds,
                output_attentions=return_attention,
                return_dict=True,
            )
        return decoder_outputs

    def predict(
        self,
        protein_ids,
        protein_mask,
        beam_width=1,
        max_length=250,
        length_normalization=0.0,
        is_precomputed=True, 
        use_sampling=False,
        temperature=1.0,
        top_k=5,
        num_samples=1,
        decoder_input_ids=None,
    ):
        self.eval()
        device = protein_ids.device

        with torch.no_grad():
            if is_precomputed:
                esm_encoder_outputs = protein_ids
            else:
                raise NotImplementedError("Only precomputed ESM embeddings are supported.")
                
            projected_embeds = self.encoder_projector(esm_encoder_outputs)

            if self.encoder is not None:
                encoder_outputs_bart = self.encoder(
                    inputs_embeds=projected_embeds,
                    attention_mask=protein_mask,
                    return_dict=True,
                )
                encoder_hidden_states = encoder_outputs_bart.last_hidden_state
            else:
                encoder_hidden_states = projected_embeds

        if use_sampling:
            # Assuming top_k_batch_sampling is imported
            predicted_ids, scores = top_k_batch_sampling(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                top_k=top_k,
                max_length=max_length,
                temperature=temperature,
                input_ids=encoder_hidden_states,
                attention_mask=protein_mask,
                device=device,
                num_samples=num_samples,
                initial_tokens=decoder_input_ids,
            )
        else:
            predicted_ids, attention_mask, scores, _, _ = beam_search(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                beam_width=beam_width,
                max_length=max_length,
                length_normalization=length_normalization,
                input_ids=encoder_hidden_states,
                attention_mask=protein_mask,
                device=device,
                initial_tokens=decoder_input_ids,
            )

        return predicted_ids

    def decode_reactions(self, preds, inv_vocab):
        decoded = []
        for i in range(preds.shape[0]):
            tokens = []
            for tid in preds[i]:
                t = tid.item()
                if t == SPECIAL_TOKENS["<END>"]: 
                    break
                if t not in [SPECIAL_TOKENS["<START>"], SPECIAL_TOKENS["<PAD>"]]:
                    token_str = inv_vocab.get(t, "")
                    if token_str not in ["<START>", "<PAD>", "<END>"]:
                        tokens.append(token_str)
            smi = "".join(tokens)
            smi = smi.replace("<START>", "").replace("<PAD>", "").replace("<END>", "")
            decoded.append(smi)
        return decoded


# -----------------------
# GRAPH MODEL
# -----------------------
class GraphReactionPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        vocab_size = TRAIN_CONFIG["vocab_size"]
        d_model = TRAIN_CONFIG["d_model"]
        nhead = TRAIN_CONFIG["nhead"]
        num_decoder_layers = TRAIN_CONFIG["num_decoder_layers"]
        dropout = TRAIN_CONFIG["dropout"]
        encoder_dim = MODEL_CONFIG["input_dim"] 
        pad_idx = SPECIAL_TOKENS["<PAD>"]

        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.encoder_projector = nn.Sequential(
            nn.Linear(encoder_dim + 1 , d_model), 
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        num_rbf = 16
        edge_attr_dim = num_rbf + 1 

        for _ in range(1):  # Start with 1 layer for debugging; can increase later
            self.gnn_layers.append(GATv2Conv(d_model, d_model // 4, heads=4, edge_dim=edge_attr_dim, dropout=dropout))
            self.gnn_norms.append(GraphNorm(d_model))

        self.decoder_config = BartConfig(
            vocab_size=vocab_size, d_model=d_model,
            decoder_layers=num_decoder_layers,
            decoder_attention_heads=nhead,
            decoder_ffn_dim=TRAIN_CONFIG["decoder_ffn_dim"],
            dropout=dropout,
            bos_token_id=SPECIAL_TOKENS["<START>"],
            eos_token_id=SPECIAL_TOKENS["<END>"],
            pad_token_id=pad_idx,
        )

        self.decoder = BartDecoder(self.decoder_config)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

    def forward(self, graph_x, plddt, edge_index, edge_attr, batch_idx, protein_mask, reaction_ids, reaction_mask):
        # Base projection
        x_raw = torch.cat([graph_x, plddt], dim=-1)
        x_raw = self.encoder_projector(x_raw) 
        
        # GNN Mixing
        x_gnn = x_raw
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            # Pre-norm is generally more stable than post-norm for deep graphs
            x_gnn = x_gnn + conv(norm(x_gnn, batch_idx), edge_index, edge_attr=edge_attr)
            x_gnn = nn.functional.gelu(x_gnn)

        # THE FIX: Global Skip Connection (Pristine ESM + 3D Context)
        x_final = x_raw + x_gnn

        batch_size, max_protein_len = reaction_ids.shape[0], protein_mask.shape[1]
        encoder_hidden = torch.zeros(batch_size, max_protein_len, x_final.shape[-1], device=x_final.device)
        counts = torch.bincount(batch_idx)
        local_idx = torch.cat([torch.arange(c, device=x_final.device) for c in counts])
        
        # Map back to sequence
        encoder_hidden[batch_idx, local_idx] = x_final

        # OPTIONAL BUT HIGHLY RECOMMENDED: 
        # encoder_hidden = encoder_hidden + self.positional_encoding(encoder_hidden)

        decoder_outputs = self._decoder_forward(
            reaction_ids=reaction_ids,
            reaction_mask=reaction_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=protein_mask
        )
        return self.lm_head(decoder_outputs.last_hidden_state)

    def _decoder_forward(
        self,
        reaction_ids,
        reaction_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        return_attention=False,
        use_cache=False,
        past_key_values=None,
    ):
        if past_key_values is not None and not isinstance(past_key_values, EncoderDecoderCache):
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            
        context = torch.no_grad() if not self.training else torch.enable_grad()
        
        with context:
            decoder_embeds = self.shared_embedding(reaction_ids)
            decoder_outputs = self.decoder(
                attention_mask=reaction_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                inputs_embeds=decoder_embeds,
                output_attentions=return_attention,
                return_dict=True,
            )
        return decoder_outputs

    def predict(
        self, 
        graph_x, 
        plddt, 
        edge_index, 
        edge_attr, 
        batch_idx, 
        protein_mask, 
        beam_width=1, 
        max_length=250, 
        length_normalization=0.0,
        use_sampling=False,
        temperature=1.0,
        top_k=5,
        num_samples=1,
        decoder_input_ids=None
    ):
        self.eval()
        device = graph_x.device

        with torch.no_grad():
            x = torch.cat([graph_x, plddt], dim=-1)
            x = self.encoder_projector(x) 
            for conv, norm in zip(self.gnn_layers, self.gnn_norms):
                x = x + conv(x, edge_index, edge_attr=edge_attr)
                x = norm(x, batch_idx)
                x = nn.functional.gelu(x)

            batch_size, max_len = protein_mask.shape[0], protein_mask.shape[1]
            encoder_hidden = torch.zeros(batch_size, max_len, x.shape[-1], device=device)
            counts = torch.bincount(batch_idx)
            local_idx = torch.cat([torch.arange(c, device=device) for c in counts])
            encoder_hidden[batch_idx, local_idx] = x

        if use_sampling:
            # Assuming top_k_batch_sampling is imported
            predicted_ids, scores = top_k_batch_sampling(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                top_k=top_k,
                max_length=max_length,
                temperature=temperature,
                input_ids=encoder_hidden,
                attention_mask=protein_mask,
                device=device,
                num_samples=num_samples,
                initial_tokens=decoder_input_ids,
            )
        else:
            predicted_ids, attention_mask, scores, _, _ = beam_search(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                beam_width=beam_width,
                max_length=max_length,
                length_normalization=length_normalization,
                input_ids=encoder_hidden,
                attention_mask=protein_mask,
                device=device,
                initial_tokens=decoder_input_ids,
            )

        return predicted_ids

    def decode_reactions(self, preds, inv_vocab):
        decoded = []
        for i in range(preds.shape[0]):
            tokens = []
            for tid in preds[i]:
                t = tid.item()
                if t == SPECIAL_TOKENS["<END>"]: 
                    break
                if t not in [SPECIAL_TOKENS["<START>"], SPECIAL_TOKENS["<PAD>"]]:
                    token_str = inv_vocab.get(t, "")
                    if token_str not in ["<START>", "<PAD>", "<END>"]:
                        tokens.append(token_str)
            smi = "".join(tokens)
            smi = smi.replace("<START>", "").replace("<PAD>", "").replace("<END>", "")
            decoded.append(smi)
        return decoded