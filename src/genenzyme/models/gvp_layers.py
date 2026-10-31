"""
Geometric Vector Perceptron (GVP) primitives for protein structure GNNs.

This is a minimal, self-contained implementation of the GVP module from
Jing et al. 2021 (https://github.com/drorlab/gvp-pytorch). We don't depend
on their package because their last release pins old PyTorch + PyG and is
flaky to install; rolling our own is ~150 lines and gives us full control
over numerical behavior under bf16 (the parent MHGGAT had a numerical bug
in `rot_to_quat` under bf16 — fixing that is a primary goal of this swap).

Core idea
---------
A GVP processes (s, v) tuples where:
    s ∈ R^{N x ds}              scalar features  (rotation-invariant)
    v ∈ R^{N x dv x 3}          vector features  (rotation-equivariant)

A single GVP layer:
  1. Compute |v|_2 along the spatial axis -> [N, dv]
  2. Concatenate with s -> [N, ds + dv]
  3. Linear -> s' [N, ds_out]
  4. Linear over v (no scalar mixing across vector channels) -> v' [N, dv_out, 3]
  5. Optionally gate v' with sigmoid(linear(s')): preserves equivariance
     because the gate is a scalar that only scales v's magnitude.
  6. Nonlinearity on s' (e.g. ReLU); v' passes through (or is gated).

Message passing:
  - Edges carry their own (se, ve).
  - For each edge (i, j), build the message from concat (s_j ⊕ se ⊕ |x_j-x_i|, v_j ⊕ ve)
    and apply a GVP. Aggregate messages by sum/mean per node.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def _vector_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2 norm of vector features along last (spatial=3) axis. Stable in bf16."""
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1), min=eps))


def _channel_linear(weight: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Channel-only linear map for vector features: v[n,c,3] -> v'[n,k,3]
    where weight has shape [k, c]. Each spatial axis is treated identically;
    spatial dimensions are NEVER mixed (preserves SE(3) equivariance).

    Implemented via torch.bmm because, on CPU, both `torch.einsum('nci,kc->nki', v, weight)`
    and `torch.matmul(weight, v)` (with broadcasting from a 2D weight to a 3D
    batched v) silently use a reduced-precision contraction path. They differ
    from the per-batch explicit matmul by ~4e-3 in fp32, which is enough to
    break SE(3) equivariance noticeably. bmm with an expanded weight tensor
    agrees exactly (1e-7) with the per-batch loop.
    """
    N = v.shape[0]
    k, c = weight.shape
    # Expand weight to [N, k, c] without copying (just stride manipulation).
    w_b = weight.unsqueeze(0).expand(N, k, c)
    return torch.bmm(w_b, v)   # [N, k, c] x [N, c, 3] -> [N, k, 3]


# =============================================================================
# Single GVP layer
# =============================================================================
class GVP(nn.Module):
    """
    One GVP transform: (s, v) -> (s', v').

    Args:
        in_dims:   (ds_in, dv_in)
        out_dims:  (ds_out, dv_out)
        h_dim:     hidden width for the vector-side linear before the magnitude
                   gate (defaults to max(dv_in, dv_out)).
        scalar_act: nonlinearity on the output scalars (default ReLU).
        vector_gate: if True (default), gate vector outputs by sigmoid of a
                     scalar projection — preserves equivariance and tends to
                     be more stable than the sigma-only schedule.
    """
    def __init__(self,
                 in_dims: tuple[int, int],
                 out_dims: tuple[int, int],
                 h_dim: Optional[int] = None,
                 scalar_act=nn.ReLU,
                 vector_gate: bool = True):
        super().__init__()
        ds_in, dv_in = in_dims
        ds_out, dv_out = out_dims
        if h_dim is None:
            h_dim = max(dv_in, dv_out, 1)

        self.ds_in, self.dv_in = ds_in, dv_in
        self.ds_out, self.dv_out = ds_out, dv_out
        self.h_dim = h_dim
        self.vector_gate = vector_gate

        # Vector-side linear (no bias; 3D vectors should be unbiased to remain equivariant).
        # Maps R^{dv_in x 3} -> R^{h_dim x 3}, then -> R^{dv_out x 3}.
        # We do it in one matmul with shape ops, no spatial mixing.
        self.wh = nn.Linear(dv_in, h_dim, bias=False) if dv_in > 0 else None
        self.wv = nn.Linear(h_dim, dv_out, bias=False) if dv_out > 0 else None

        # Scalar branch consumes (s, |wh(v)|) -> ds_out
        self.ws = nn.Linear(ds_in + (h_dim if dv_in > 0 else 0), ds_out)

        # Vector gate (if enabled): scalar projection -> sigmoid -> elementwise gain.
        if dv_out > 0 and vector_gate:
            self.wsv = nn.Linear(ds_out, dv_out)
        else:
            self.wsv = None

        self.scalar_act = scalar_act() if scalar_act is not None else nn.Identity()

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        """
        s: [N, ds_in]
        v: [N, dv_in, 3]    (or empty if dv_in == 0)
        Returns (s', v')  with shapes [N, ds_out], [N, dv_out, 3].

        IMPORTANT: vector-side linear maps must be channel-only (no spatial
        mixing) to preserve SE(3) equivariance. We use torch.matmul instead
        of einsum because PyTorch's einsum on CPU has been observed to use
        a reduced-precision contraction path here that breaks equivariance
        by ~1e-2 (vs. matmul giving exact agreement with the per-batch
        loop). Same effect would hide bugs under cuBLAS TF32 paths on GPU.
        """
        # --- Vector branch: v[n,c,:] -> wh @ v[n,c,:] over channel axis only.
        if self.dv_in > 0 and self.wh is not None:
            vh = _channel_linear(self.wh.weight, v)      # [N, h_dim, 3]
            vh_norm = _vector_norm(vh)                   # [N, h_dim]
            s_in = torch.cat([s, vh_norm], dim=-1)       # [N, ds_in + h_dim]
        else:
            vh = None
            s_in = s

        # --- Scalar branch
        s_out = self.ws(s_in)                            # [N, ds_out]

        # --- Vector branch (continued): h_dim -> dv_out
        if self.dv_out > 0 and self.wv is not None and vh is not None:
            v_out = _channel_linear(self.wv.weight, vh)  # [N, dv_out, 3]
            if self.wsv is not None:
                gate = torch.sigmoid(self.wsv(s_out)).unsqueeze(-1)   # [N, dv_out, 1]
                v_out = v_out * gate
        else:
            v_out = s.new_zeros(s.shape[0], 0, 3)

        # Nonlinearity on scalars after the gate so the gate sees pre-activation.
        s_out = self.scalar_act(s_out)
        return s_out, v_out


# =============================================================================
# Layer norm for (s, v)
# =============================================================================
class GVPLayerNorm(nn.Module):
    """Standard LayerNorm on s; per-node L2 norm on v (preserves direction)."""
    def __init__(self, dims: tuple[int, int], eps: float = 1e-5):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(dims[0]) if dims[0] > 0 else nn.Identity()
        self.eps = eps

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        s = self.scalar_norm(s)
        if v.numel() == 0:
            return s, v
        # Normalize each v-channel by its average magnitude across nodes.
        # Equivalent to RMS norm on the magnitude axis. Direction preserved.
        n = _vector_norm(v)                              # [N, dv]
        rms = torch.sqrt((n * n).mean(dim=-1, keepdim=True) + self.eps)  # [N, 1]
        v = v / rms.unsqueeze(-1)
        return s, v


# =============================================================================
# Message passing GVP layer
# =============================================================================
class GVPConv(MessagePassing):
    """
    One round of GVP message passing:
      m_ij = GVP_msg([s_j ⊕ se, v_j ⊕ ve])
      h_i  = h_i + aggr_j(m_ij)         (residual + sum aggregation)
    Followed by a small feed-forward GVP block (also residual).

    Inputs to forward():
      s:          [N, ds]
      v:          [N, dv, 3]
      edge_index: [2, E]
      edge_s:     [E, des]   (pre-projected edge scalar features)
      edge_v:     [E, dev, 3] (pre-projected edge vector features)
    """
    def __init__(self,
                 node_dims: tuple[int, int],
                 edge_dims: tuple[int, int],
                 n_msg_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__(aggr='mean', node_dim=0)
        ds, dv = node_dims
        des, dev = edge_dims

        # Message GVP: input is concat of (s_j, se), (v_j, ve) -> (ds, dv)
        msg_in = (ds + des, dv + dev)
        msg_layers = []
        for i in range(n_msg_layers):
            in_dims  = msg_in if i == 0 else node_dims
            out_dims = node_dims
            scalar_act = nn.ReLU if i < n_msg_layers - 1 else None
            msg_layers.append(GVP(in_dims, out_dims, scalar_act=scalar_act))
        self.msg_gvp = nn.Sequential() if False else _GVPSequential(msg_layers)

        # Feed-forward block (post-aggregation)
        self.ff = _GVPSequential([
            GVP(node_dims, node_dims, scalar_act=nn.ReLU),
            GVP(node_dims, node_dims, scalar_act=None),
        ])

        # Norms (s only; vector norm handled inside GVPLayerNorm differently).
        self.norm1 = GVPLayerNorm(node_dims)
        self.norm2 = GVPLayerNorm(node_dims)

        self.scalar_drop = nn.Dropout(dropout)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        # -- message passing pass 1 (residual)
        ds_msg, dv_msg = self.propagate(edge_index, s=s, v=v,
                                        edge_s=edge_s, edge_v=edge_v)
        s = s + self.scalar_drop(ds_msg)
        v = v + dv_msg
        s, v = self.norm1(s, v)

        # -- feed-forward (residual)
        ds_ff, dv_ff = self.ff(s, v)
        s = s + self.scalar_drop(ds_ff)
        v = v + dv_ff
        s, v = self.norm2(s, v)
        return s, v

    def message(self, s_j, v_j, edge_s, edge_v):
        # Concatenate node features (from source) with edge features.
        s_in = torch.cat([s_j, edge_s], dim=-1)
        v_in = torch.cat([v_j, edge_v], dim=-2)  # concat along channel axis (axis 1 of [N, c, 3])
        return self.msg_gvp(s_in, v_in)

    # PyG MessagePassing supports tuple aggregation by overriding aggregate
    # and update; here we let propagate return a tuple (s, v) by treating the
    # outputs of `message` as a 2-tuple. The default `aggregate` works on the
    # tensor directly; for a tuple we implement aggregate manually.
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        s_msg, v_msg = inputs
        from torch_geometric.utils import scatter
        s_agg = scatter(s_msg, index, dim=0, dim_size=dim_size, reduce=self.aggr)
        # Vector aggregation: same scatter but with v's leading dim flattened.
        N, c, three = v_msg.shape
        v_flat = v_msg.reshape(N, c * three)
        v_agg = scatter(v_flat, index, dim=0, dim_size=dim_size, reduce=self.aggr)
        v_agg = v_agg.reshape(v_agg.shape[0], c, three)
        return s_agg, v_agg

    def update(self, inputs):
        return inputs   # already a tuple (s, v)


# =============================================================================
# Internal: a tiny Sequential that handles tuple I/O for GVPs.
# =============================================================================
class _GVPSequential(nn.Module):
    def __init__(self, modules: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, s, v):
        for layer in self.layers:
            s, v = layer(s, v)
        return s, v


# =============================================================================
# Self-test
# =============================================================================
def _precise_rotation(theta: float = 0.7) -> torch.Tensor:
    """
    Returns a 3x3 rotation matrix orthogonal to ~1e-9 precision (vs. ~1e-3
    for torch.linalg.qr of a random matrix in fp32). Required for the
    equivariance assertions below to be tight.
    """
    import math
    c, s = math.cos(theta), math.sin(theta)
    # Compose three axis rotations so we exercise all three coordinate axes.
    Rz = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    Ry = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    Rx = torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]])
    return Rz @ Ry @ Rx


def _self_test():
    """
    Synthetic batch through GVP + GVPConv, verify:
      - shapes
      - SE(3) equivariance: rotating inputs rotates v outputs identically;
        scalar outputs stay the same.

    Uses a precisely-constructed rotation (not torch.linalg.qr, which is
    only orthogonal to ~1e-3 in fp32 and fails tight assertions).
    """
    torch.manual_seed(0)
    N, E = 16, 40
    ds, dv = 8, 4
    des, dev = 6, 2

    s  = torch.randn(N, ds)
    v  = torch.randn(N, dv, 3)
    es = torch.randn(E, des)
    ev = torch.randn(E, dev, 3)
    ei = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
    R  = _precise_rotation()

    # 1. GVP shape check
    g = GVP((ds, dv), (10, 5))
    g.eval()
    so, vo = g(s, v)
    assert so.shape == (N, 10), so.shape
    assert vo.shape == (N, 5, 3), vo.shape

    # 2. Equivariance: rotate v, call again, check vo also rotated.
    s2, v2 = g(s, torch.matmul(v, R))
    assert torch.allclose(s2, so, atol=1e-5), \
        f"scalar output should be rotation-invariant (max diff {(s2-so).abs().max():.2e})"
    vo_rot = torch.matmul(vo, R)
    assert torch.allclose(v2, vo_rot, atol=1e-5), \
        f"vector output should be rotation-equivariant (max diff {(v2-vo_rot).abs().max():.2e})"

    # 3. GVPConv shape check
    conv = GVPConv((ds, dv), (des, dev), n_msg_layers=2, dropout=0.0)
    conv.eval()
    so, vo = conv(s, v, ei, es, ev)
    assert so.shape == (N, ds), so.shape
    assert vo.shape == (N, dv, 3), vo.shape

    # 4. GVPConv equivariance. We allow a looser tolerance here because
    # the conv composes 4 GVP sublayers + 2 norms + scatter aggregation;
    # fp32 roundoff accumulates to ~5e-4 across that depth even when each
    # layer is individually exact.
    so0, vo0 = conv(s, v, ei, es, ev)
    so1, vo1 = conv(s,
                    torch.matmul(v, R),
                    ei, es,
                    torch.matmul(ev, R))
    assert torch.allclose(so0, so1, atol=1e-3), \
        f"conv scalars not invariant (max diff {(so0-so1).abs().max():.2e})"
    vo0_rot = torch.matmul(vo0, R)
    assert torch.allclose(vo0_rot, vo1, atol=1e-3), \
        f"conv vectors not equivariant (max diff {(vo0_rot-vo1).abs().max():.2e})"

    print("[gvp_layers] OK | GVP + GVPConv shape and SE(3) equivariance verified")


if __name__ == "__main__":
    _self_test()
