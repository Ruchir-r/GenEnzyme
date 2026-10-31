import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool, global_max_pool
import wandb
from tqdm import tqdm
from transformers.cache_utils import EncoderDecoderCache

# Standard Transformers imports
from transformers.models.bart.modeling_bart import BartDecoder, BartConfig
from transformers import get_cosine_schedule_with_warmup

# Assuming these are in your local files
from data_featurizer import GraphReactionDataset
from config import DATA_CONFIG, MODEL_CONFIG, SPECIAL_TOKENS, TRAIN_CONFIG
from model import top_k_batch_sampling, beam_search # Uncomment if used for inference later

# ---------------------------------------------------------
# 1. GEOMETRIC FEATURIZER
# ---------------------------------------------------------
class RBFExpansion(nn.Module):
    def __init__(self, v_min=0.0, v_max=20.0, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        centers = torch.linspace(v_min, v_max, num_bins)
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / (centers[1] - centers[0])

    def forward(self, distances):
        return torch.exp(-self.gamma * torch.pow(distances.unsqueeze(-1) - self.centers, 2))

def get_local_frames(n_coords, ca_coords, c_coords):
    v1 = n_coords - ca_coords
    v2 = c_coords - ca_coords
    e1 = F.normalize(v1, dim=-1)
    u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1, keepdim=True))
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)

def rot_to_quat(rot_mat):
    m00, m11, m22 = rot_mat[..., 0, 0], rot_mat[..., 1, 1], rot_mat[..., 2, 2]
    trace = m00 + m11 + m22
    q = torch.zeros(*rot_mat.shape[:-2], 4, device=rot_mat.device, dtype=rot_mat.dtype)
    q[..., 0] = torch.sqrt(torch.clamp(1.0 + trace, min=0.0)) / 2.0 
    q[..., 1] = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) / (4.0 * q[..., 0] + 1e-6)
    q[..., 2] = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) / (4.0 * q[..., 0] + 1e-6)
    q[..., 3] = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) / (4.0 * q[..., 0] + 1e-6)
    return F.normalize(q, dim=-1)

class ProteinGeometricFeaturizer(nn.Module):
    def __init__(self, d_model, num_rbf=16):
        super().__init__()
        self.rbf = RBFExpansion(num_bins=num_rbf)
        self.d_model = d_model
        
        node_raw_dim = (num_rbf * 4) + (3 * 4) + 6 
        edge_raw_dim = (num_rbf * 1) + 3 + 4 
        
        # 🟢 FIX 1: Applied LayerNorm and GELU to align variance with the ESM stream
        self.node_proj = nn.Sequential(
            nn.Linear(node_raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, coords, edge_index, torsion_angles):
        N, CA, C, O, R = coords['N'], coords['CA'], coords['C'], coords['O'], coords['R']
        
        assert not torch.isnan(CA).any(), "NaN detected in CA coordinates!"
        assert N.shape == CA.shape, f"Shape mismatch: N {N.shape} != CA {CA.shape}"
        
        orig_dtype = CA.dtype
        N, CA, C, O, R = N.float(), CA.float(), C.float(), O.float(), R.float()
        torsion_angles = torsion_angles.float()
        
        Q = get_local_frames(N, CA, C)
        
        dist_N = self.rbf(torch.norm(N - CA, dim=-1))
        dist_C = self.rbf(torch.norm(C - CA, dim=-1))
        dist_O = self.rbf(torch.norm(O - CA, dim=-1))
        dist_R = self.rbf(torch.norm(R - CA, dim=-1))
        
        def local_dir(A, CA_ref, Q_ref):
            vec = F.normalize(A - CA_ref, dim=-1).unsqueeze(-1)
            return torch.bmm(Q_ref.transpose(1, 2), vec).squeeze(-1)
            
        dir_N, dir_C = local_dir(N, CA, Q), local_dir(C, CA, Q)
        dir_O, dir_R = local_dir(O, CA, Q), local_dir(R, CA, Q)
        angle_feats = torch.cat([torch.sin(torsion_angles), torch.cos(torsion_angles)], dim=-1)
        
        node_features = torch.cat([
            dist_N, dist_C, dist_O, dist_R,
            dir_N, dir_C, dir_O, dir_R,
            angle_feats
        ], dim=-1).to(orig_dtype)
        
        h_nodes = self.node_proj(node_features)

        if edge_index.numel() == 0:
            # 🟢 FIX 2: Switched to self.d_model since Sequential has no out_features
            h_edges = torch.empty((0, self.d_model), device=CA.device, dtype=orig_dtype)
            return h_nodes, h_edges

        src, dst = edge_index[0], edge_index[1]
        edge_dist = self.rbf(torch.norm(CA[src] - CA[dst], dim=-1))
        
        edge_vec = F.normalize(CA[dst] - CA[src], dim=-1).unsqueeze(-1)
        edge_dir = torch.bmm(Q[src].transpose(1, 2), edge_vec).squeeze(-1)
        Q_rel = torch.bmm(Q[src].transpose(1, 2), Q[dst])
        edge_quat = rot_to_quat(Q_rel)
        
        edge_features = torch.cat([edge_dist, edge_dir, edge_quat], dim=-1).to(orig_dtype)
        h_edges = self.edge_proj(edge_features)
        
        return h_nodes, h_edges

# ---------------------------------------------------------
# 2. MULTI-HEAD GEOMETRIC GAT
# ---------------------------------------------------------
class MultiHeadGeometricGAT(MessagePassing):
    def __init__(self, d_model, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_E = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x

        q = self.W_Q(x).view(-1, self.heads, self.head_dim)
        k = self.W_K(x).view(-1, self.heads, self.head_dim)
        v = self.W_V(x).view(-1, self.heads, self.head_dim)
        e = self.W_E(edge_attr).view(-1, self.heads, self.head_dim)
        
        out = self.propagate(edge_index, q=q, k=k, v=v, e=e)
        out = out.view(-1, self.d_model)
        return self.out_proj(out)

    def message(self, q_i, k_j, v_j, e, index, ptr, size_i):
        alpha = (q_i * (k_j + e)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)
        return alpha.unsqueeze(-1) * (v_j + e)

# ---------------------------------------------------------
# 3. INTERPRETABILITY-FOCUSED PREDICTOR
# ---------------------------------------------------------
class GraphReactionPredictor(nn.Module):
    def __init__(self, num_mpnn_layers=2):
        super().__init__()

        vocab_size = TRAIN_CONFIG["vocab_size"]
        self.d_model = d_model = TRAIN_CONFIG["d_model"]
        pad_idx = SPECIAL_TOKENS["<PAD>"]

        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 🟢 FIX 3: Added Modality Gap infrastructure
        self.modality_emb = nn.Embedding(2, d_model)
        self.seq_pos_emb = nn.Embedding(4096, d_model) # Match decoder max positions
        self.gnn_gate = nn.Parameter(torch.zeros(1))
        self.gnn_final_norm = nn.LayerNorm(d_model)
        
        # STREAM A: Semantic Sequence (ESM)
        self.esm_proj = nn.Sequential(
            nn.Linear(MODEL_CONFIG["input_dim"] + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # STREAM B: 3D Geometric Structure (GNN)
        self.featurizer = ProteinGeometricFeaturizer(d_model=d_model)
        
        self.gat_layers = nn.ModuleList([
            MultiHeadGeometricGAT(d_model=d_model, heads=4, dropout=TRAIN_CONFIG["dropout"])
            for _ in range(num_mpnn_layers)
        ])
        self.gnn_norms = nn.ModuleList([
            GraphNorm(d_model) for _ in range(num_mpnn_layers)
        ])

        # DECODER
        self.decoder_config = BartConfig(
            vocab_size=vocab_size, 
            d_model=d_model,
            encoder_layers=TRAIN_CONFIG["num_decoder_layers"], 
            decoder_layers=TRAIN_CONFIG["num_decoder_layers"],
            num_hidden_layers=TRAIN_CONFIG["num_decoder_layers"], 
            decoder_attention_heads=TRAIN_CONFIG["nhead"],
            decoder_ffn_dim=TRAIN_CONFIG["decoder_ffn_dim"],
            dropout=TRAIN_CONFIG["dropout"],
            bos_token_id=SPECIAL_TOKENS["<START>"],
            eos_token_id=SPECIAL_TOKENS["<END>"],
            pad_token_id=pad_idx,
            max_position_embeddings=4096, 
            attn_implementation="eager" 
        )

        self.decoder = BartDecoder(self.decoder_config, embed_tokens=self.shared_embedding)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

    def forward(self, graph_x, plddt, coords, torsions, edge_index, batch_idx, protein_mask, reaction_ids, reaction_mask):
        device = graph_x.device
        batch_size = reaction_ids.shape[0]
        max_protein_len = protein_mask.shape[1]
        
        # --- 1. PROCESS ESM ---
        plddt = plddt / 100.0   
        x_esm = self.esm_proj(torch.cat([graph_x, plddt], dim=-1))
        
        # --- 2. PROCESS GEOMETRY ---
        h_nodes, h_edges = self.featurizer(coords, edge_index, torsions)
        
        x_gnn = h_nodes
        for gat, norm in zip(self.gat_layers, self.gnn_norms):
            x_norm = norm(x_gnn, batch_idx)
            x_gnn = x_gnn + gat(x_norm, edge_index, h_edges) 
            
        # --- 3. EXTRACT GLOBAL CONTEXT TOKENS ---
        g_mean = global_mean_pool(x_gnn, batch_idx).unsqueeze(1) 
        g_max = global_max_pool(x_gnn, batch_idx).unsqueeze(1)   

        # --- 4. ALIGN TO DENSE SEQUENCE TENSORS ---
        esm_hidden = torch.zeros(batch_size, max_protein_len, self.d_model, device=device, dtype=x_esm.dtype)
        gnn_hidden = torch.zeros(batch_size, max_protein_len, self.d_model, device=device, dtype=x_gnn.dtype)
        
        counts = torch.bincount(batch_idx)
        local_idx = torch.cat([torch.arange(c, device=device) for c in counts])
        
        esm_hidden[batch_idx, local_idx] = x_esm
        
        # 🟢 FIX 4: Apply final norm and zero-init gate to GNN stream
        x_gnn_normed = self.gnn_final_norm(x_gnn)
        gnn_hidden[batch_idx, local_idx] = x_gnn_normed * self.gnn_gate.to(x_gnn_normed.dtype)

        # 🟢 FIX 5: Inject Positional & Modality Tags
        positions = torch.arange(max_protein_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.seq_pos_emb(positions)
        
        esm_hidden = esm_hidden + pos_embeddings + self.modality_emb(torch.tensor(0, device=device))
        gnn_hidden = gnn_hidden + pos_embeddings + self.modality_emb(torch.tensor(1, device=device))

        # --- 5. DUAL-SEQUENCE CONCATENATION ---
        encoder_hidden = torch.cat([g_mean, g_max, esm_hidden, gnn_hidden], dim=1)
        
        global_mask = torch.ones((batch_size, 2), dtype=protein_mask.dtype, device=device)
        extended_protein_mask = torch.cat([global_mask, protein_mask, protein_mask], dim=1)

        # --- 6. DECODER PASS ---
        decoder_outputs = self.decoder(
            input_ids=reaction_ids,           
            attention_mask=reaction_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=extended_protein_mask,
            output_attentions=True 
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
        coords,
        torsions,
        edge_index,
        batch_idx,
        protein_mask,
        beam_width=1,
        max_length=250,
        length_normalization=0.0,
        use_sampling=False,
        temperature=1.0,
        top_k=5,
        num_samples=1,
        decoder_input_ids=None,
    ):
        self.eval()
        device = graph_x.device
        batch_size = protein_mask.shape[0]
        max_protein_len = protein_mask.shape[1]

        with torch.no_grad():
            # --- 1. PROCESS ESM ---
            plddt_norm = plddt / 100.0   
            x_esm = self.esm_proj(torch.cat([graph_x, plddt_norm], dim=-1))
            
            # --- 2. PROCESS GEOMETRY ---
            h_nodes, h_edges = self.featurizer(coords, edge_index, torsions)
            
            x_gnn = h_nodes
            for gat, norm in zip(self.gat_layers, self.gnn_norms):
                x_norm = norm(x_gnn, batch_idx)
                x_gnn = x_gnn + gat(x_norm, edge_index, h_edges) 
                
            # --- 3. EXTRACT GLOBAL CONTEXT TOKENS ---
            g_mean = global_mean_pool(x_gnn, batch_idx).unsqueeze(1) 
            g_max = global_max_pool(x_gnn, batch_idx).unsqueeze(1)   

            # --- 4. ALIGN TO DENSE SEQUENCE TENSORS ---
            esm_hidden = torch.zeros(batch_size, max_protein_len, self.d_model, device=device, dtype=x_esm.dtype)
            gnn_hidden = torch.zeros(batch_size, max_protein_len, self.d_model, device=device, dtype=x_gnn.dtype)
            
            counts = torch.bincount(batch_idx)
            local_idx = torch.cat([torch.arange(c, device=device) for c in counts])
            
            esm_hidden[batch_idx, local_idx] = x_esm
            
            # 🟢 FIX 6: Mirroring Forward Pass Gates and Tags to Predict Loop
            x_gnn_normed = self.gnn_final_norm(x_gnn)
            gnn_hidden[batch_idx, local_idx] = x_gnn_normed * self.gnn_gate
            
            positions = torch.arange(max_protein_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.seq_pos_emb(positions)
            
            esm_hidden = esm_hidden + pos_embeddings + self.modality_emb(torch.tensor(0, device=device))
            gnn_hidden = gnn_hidden + pos_embeddings + self.modality_emb(torch.tensor(1, device=device))

            # --- 5. DUAL-SEQUENCE CONCATENATION ---
            encoder_hidden_states = torch.cat([g_mean, g_max, esm_hidden, gnn_hidden], dim=1)
            
            global_mask = torch.ones((batch_size, 2), dtype=protein_mask.dtype, device=device)
            extended_protein_mask = torch.cat([global_mask, protein_mask, protein_mask], dim=1)

        if use_sampling:
            predicted_ids, scores = top_k_batch_sampling(
                model=self,
                bos_token_id=self.decoder_config.bos_token_id,
                eos_token_id=self.decoder_config.eos_token_id,
                pad_token_id=self.decoder_config.pad_token_id,
                top_k=top_k,
                max_length=max_length,
                temperature=temperature,
                input_ids=encoder_hidden_states,
                attention_mask=extended_protein_mask,
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
                attention_mask=extended_protein_mask,
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

# ---------------------------------------------------------
# 4. UTILS & EVALUATION
# ---------------------------------------------------------
def build_decoder(vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    def decode(seq_tensor):
        tokens = []
        for t in seq_tensor:
            # Handle standard ints vs 0D tensors safely
            val = t.item() if isinstance(t, torch.Tensor) else t
            token = id_to_token.get(val, "<UNK>")
            if token == SPECIAL_TOKENS["<PAD>"]: continue
            tokens.append(token)
            if token == SPECIAL_TOKENS["<END>"]: break
        return " ".join(tokens)
    return decode


def evaluate(model, loader, loss_fn, decode_fn, device="cuda"):
    model.eval()
    total_loss = 0

    rand_batch_idx = random.randint(0, len(loader) - 1)
    sample_pred, sample_true = "", ""

    with torch.no_grad():
        for i, batch in enumerate(loader):
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                logits = model(
                    graph_x=batch["graph_x"].to(device),
                    plddt=batch["plddt"].to(device),
                    coords={k: v.to(device) for k, v in batch["coords"].items()},
                    torsions=batch["torsions"].to(device),
                    edge_index=batch["edge_index"].to(device),
                    batch_idx=batch["batch_idx"].to(device),
                    protein_mask=batch["protein_mask"].to(device),
                    reaction_ids=batch["reaction_ids"].to(device),
                    reaction_mask=batch["reaction_mask"].to(device)
                )
                labels = batch["labels"].to(device)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()

            if i == rand_batch_idx:
                # 1. Get the most likely tokens
                pred_ids_batch = logits.argmax(dim=-1) # Shape: [Batch, Seq_Len]
                
                # 2. Pick a random sample from this batch
                rand_sample_idx = random.randint(0, pred_ids_batch.size(0) - 1)
                pred_ids = pred_ids_batch[rand_sample_idx]
                true_ids = labels[rand_sample_idx]

                # 3. Correct way to find the <END> index in a 1D Tensor
                end_token = SPECIAL_TOKENS["<END>"]
                end_mask = (pred_ids == end_token).nonzero(as_tuple=True)[0]
                
                if len(end_mask) > 0:
                    # Take the first occurrence of <END>
                    end_idx = end_mask[0].item()
                    pred_ids = pred_ids[:end_idx+1]

                # 4. Decode
                sample_pred = decode_fn(pred_ids).replace(" ", "").replace("<START>", "")
                sample_true = decode_fn(true_ids).replace(" ", "").replace("<START>", "")

    return total_loss / len(loader), sample_pred, sample_true

# ---------------------------------------------------------
# 5. MAIN SCRIPT
# ---------------------------------------------------------
BATCH_SIZE = TRAIN_CONFIG["batch_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = TRAIN_CONFIG["num_epochs"]
NUM_WORKERS = 4 if DEVICE == "cuda" else 0

def main():
    print(f"🔧 Starting Geometric GAT Training on {DEVICE}...")

    train_dataset = GraphReactionDataset(DATA_CONFIG["train_path"], DATA_CONFIG["vocab_path"], DATA_CONFIG["embeddings_dir"])
    val_dataset = GraphReactionDataset(DATA_CONFIG["val_path"], DATA_CONFIG["vocab_path"], DATA_CONFIG["embeddings_dir"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=NUM_WORKERS)

    decode_fn = build_decoder(train_dataset.vocab)
    model = GraphReactionPredictor(num_mpnn_layers=2).to(DEVICE)
    
    # Differential LR Setup
    gnn_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # 🟢 Added gnn_gate to the faster learning rate group so it moves quickly
        if any(kw in name for kw in ['featurizer', 'gat_layers', 'gnn_norms', 'gnn_gate']):
            gnn_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-4},
        {'params': gnn_params, 'lr': 5e-3} 
    ], weight_decay=0.01)
    
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.10), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"], label_smoothing=0.1)

    wandb.init(project="rxn_pred", name="geometric_gat_bf16", mode="offline")

    best_val_loss = float("inf")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        epoch_train_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for step, batch in pbar:
            
            if batch["edge_index"].numel() == 0:
                continue

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
                logits = model(
                    graph_x=batch["graph_x"].to(DEVICE),
                    plddt=batch["plddt"].to(DEVICE),
                    coords={k: v.to(DEVICE) for k, v in batch["coords"].items()},
                    torsions=batch["torsions"].to(DEVICE),
                    edge_index=batch["edge_index"].to(DEVICE),
                    batch_idx=batch["batch_idx"].to(DEVICE),
                    protein_mask=batch["protein_mask"].to(DEVICE),
                    reaction_ids=batch["reaction_ids"].to(DEVICE),
                    reaction_mask=batch["reaction_mask"].to(DEVICE)
                )
                labels = batch["labels"].to(DEVICE)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            
            base_lr = optimizer.param_groups[0]['lr']
            gnn_lr = optimizer.param_groups[1]['lr']
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "grad_norm": f"{total_norm.item():.2f}",
                "gnn_gate": f"{model.gnn_gate.item():.3f}" # Watch the gate climb!
            })

            wandb.log({
                "train/step_loss": loss.item(), 
                "train/grad_norm": total_norm.item(),
                "train/gnn_gate": model.gnn_gate.item(), # 🟢 Track the fade-in gate
                "lr/base": base_lr,
                "lr/gnn": gnn_lr,
                "global_step": epoch * len(train_loader) + step
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        val_loss, pred, true = evaluate(model, val_loader, loss_fn, decode_fn, device=DEVICE)
        
        print(f"\n📈 Avg Train Loss: {avg_train_loss:.4f} | 📉 Val Loss: {val_loss:.4f}")
        print(f"🔮 Pred: {pred}\n🎯 True: {true}")
        
        exact_match = "✅" if pred == true else "❌"
        pred_table = wandb.Table(columns=["Epoch", "Prediction", "True SMILES", "Match"])
        pred_table.add_data(epoch, pred, true, exact_match)

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "val/loss": val_loss,
            "val/predictions": pred_table
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("./models/MHGGat", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch
            }, os.path.join("./models/MHGGat", "best_geometric_model.pt"))
            print(f"✨ Saved new best model!")

if __name__ == "__main__":
    main()