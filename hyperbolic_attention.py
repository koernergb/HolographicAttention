import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class HyperbolicRadialAttention(nn.Module):
    """
    Hyperbolic radial attention with boundary residuals.
    
    This attention mechanism operates in hyperbolic space using Poincaré ball projections
    with learnable negative curvature. Features include:
      - Poincaré-ball projection for Q/K/V with learnable curvature (optional)
      - Per-head learned radial scaling parameters
      - Hyperbolic distance-based attention scores using geodesic distances (optional)
      - Boundary embedding residual connections (optional)
      - Temperature-scaled softmax for tropical-like behavior (optional)

    Inspired by holographic bulk–boundary intuitions, but implemented as standard
    hyperbolic attention with geometric constraints.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
        batch_first: If True, input tensors are (batch, seq, embed_dim)
        use_hyperbolic: Whether to use hyperbolic projections and distances
        use_tropical: Whether to apply temperature scaling to softmax
        use_boundary_residual: Whether to add boundary embedding residuals
        curvature: Initial curvature parameter (kept negative during training)

    Assumes batch_first=True: inputs are (batch, seq, embed_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_hyperbolic: bool = True,
        use_tropical: bool = False,
        use_boundary_residual: bool = True,
        curvature: float = -1.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        # Flags
        self.use_hyperbolic = use_hyperbolic
        self.use_tropical = use_tropical
        self.use_boundary_residual = use_boundary_residual

        # Learnable curvature (kept negative)
        self.curvature = nn.Parameter(torch.tensor(float(curvature)))

        # Per-head radial weights (learned, then squashed by sigmoid)
        self.radial_weights = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Boundary embedding
        if self.use_boundary_residual:
            self.boundary_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.boundary_proj = None

        # Tropical / temperature scale
        if self.use_tropical:
            self.tropical_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("tropical_scale", None)

    # ---------- Shape helpers ----------

    def _shape(self, x: Tensor, bsz: int) -> Tensor:
        # (B, L, D) -> (B, num_heads, L, head_dim)
        return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, x: Tensor, bsz: int) -> Tensor:
        # (B, num_heads, L, head_dim) -> (B, L, D)
        return x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)

    # ---------- Hyperbolic geometry ----------

    def _hyperbolic_project(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Project x into a Poincaré ball with learnable curvature c < 0.
        """
        c = -torch.abs(self.curvature)  # ensure negative
        norm = torch.norm(x, dim=-1, keepdim=True)  # (..., 1)
        sqrt_c = torch.sqrt(torch.clamp(-c, min=eps))
        scaled_norm = sqrt_c * norm
        scale = torch.tanh(scaled_norm) / (scaled_norm + eps)
        return x * scale

    def _hyperbolic_distance(self, q: Tensor, k: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Approximate pairwise Poincaré-ball distance between q and k.
        q: (B, H, L, d_h)
        k: (B, H, S, d_h)
        Returns: (B, H, L, S)
        """
        c = -torch.abs(self.curvature)  # negative
        sqrt_c = torch.sqrt(torch.clamp(-c, min=eps))

        # Squared norms
        q_norm2 = (q ** 2).sum(dim=-1, keepdim=True)          # (B, H, L, 1)
        k_norm2 = (k ** 2).sum(dim=-1, keepdim=True)          # (B, H, S, 1)
        k_norm2 = k_norm2.transpose(-2, -1)                   # (B, H, 1, S)

        # Squared distance ||q - k||^2
        qk = torch.matmul(q, k.transpose(-2, -1))             # (B, H, L, S)
        dist2 = q_norm2 + k_norm2 - 2.0 * qk                  # broadcast

        # Denominator (1 - ||q||^2)(1 - ||k||^2)
        denom = (1.0 - q_norm2) * (1.0 - k_norm2)
        denom = torch.clamp(denom, min=eps)

        # Argument to arcosh; must be >= 1
        cosh_arg = 1.0 + 2.0 * dist2 / denom
        cosh_arg = torch.clamp(cosh_arg, min=1.0 + eps)

        # d_H = (1/sqrt(-c)) * arcosh(cosh_arg)
        d = torch.acosh(cosh_arg) / (sqrt_c + eps)
        return d

    # ---------- Forward ----------

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        query, key, value: (B, L, D) with batch_first=True
        attn_mask: optional additive mask with shape broadcastable to (B, H, L, S),
                   where 0 = keep, negative large (e.g. -1e9) = mask.
        """
        if not self.batch_first:
            raise ValueError("This implementation assumes batch_first=True")

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, "query embed_dim mismatch"

        if key is None:
            key = query
        if value is None:
            value = query

        src_len = key.size(1)
        
        # Store original query for boundary residual
        original_query = query

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Shape to (B, H, L, d_h) / (B, H, S, d_h)
        q = self._shape(q, bsz)
        k = self._shape(k, bsz)
        v = self._shape(v, bsz)

        # Optional hyperbolic projection
        if self.use_hyperbolic:
            q = self._hyperbolic_project(q)
            k = self._hyperbolic_project(k)
            v = self._hyperbolic_project(v)

        # Per-head radial scaling for Q
        radial = torch.sigmoid(self.radial_weights).unsqueeze(0)  # (1, H, 1, 1)
        q = q * radial

        # Compute scores
        if self.use_hyperbolic:
            dist = self._hyperbolic_distance(q, k)  # (B, H, L, S)
            scores = -dist / math.sqrt(self.head_dim)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)

        # Apply attention mask if provided (additive mask)
        if attn_mask is not None:
            while attn_mask.dim() < scores.dim():
                attn_mask = attn_mask.unsqueeze(0)
            scores = scores + attn_mask

        # Softmax (with optional tropical scaling)
        if self.use_tropical and self.tropical_scale is not None:
            scaled_scores = scores * self.tropical_scale
        else:
            scaled_scores = scores

        attn_weights = torch.softmax(scaled_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, d_h)

        # Merge heads and project out
        attn_output = self._unshape(attn_output, bsz)  # (B, L, D)
        attn_output = self.out_proj(attn_output)

        # Add boundary residual (computed after projection to match dimensions)
        if self.use_boundary_residual and self.boundary_proj is not None:
            boundary = self.boundary_proj(original_query)
            attn_output = attn_output + boundary

        if need_weights:
            avg_weights = attn_weights.mean(dim=1)  # (B, L, S)
            return attn_output, avg_weights
        else:
            return attn_output, None


class HyperbolicTransformerBlock(nn.Module):
    """
    Transformer block using hyperbolic radial attention.
    
    A standard transformer architecture with HyperbolicRadialAttention replacing
    the standard multi-head attention. Uses pre-layer normalization and residual
    connections.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = HyperbolicRadialAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            use_hyperbolic=True,
            use_tropical=False,
            use_boundary_residual=True,
            curvature=-1.0,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


if __name__ == "__main__":
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    block = HyperbolicTransformerBlock(d_model=D, n_heads=4, d_ff=128)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test the attention layer directly
    attn = HyperbolicRadialAttention(
        embed_dim=64,
        num_heads=4,
        dropout=0.1,
        batch_first=True,
        use_hyperbolic=True,
        use_tropical=False,
        use_boundary_residual=True,
        curvature=-1.0,
    )
    out, weights = attn(x, need_weights=True)
    print(f"Attention output shape: {out.shape}")
    if weights is not None:
        print(f"Attention weights shape: {weights.shape}")
