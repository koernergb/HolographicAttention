import torch
from torch import nn
import torch.nn.functional as F
from tensor_train_utils import to_tensor_train, reconstruct_from_tt
from hyperbolic_attention import HyperbolicRadialAttention


class TensorTrainAttention(nn.Module):
    """Attention mechanism using tensor train decomposition"""
    def __init__(self, d_model, n_heads, tt_ranks):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tt_ranks = tt_ranks
        
        # Create parameters for Q, K, V projections in TT format
        # For simplicity, we'll use full matrices here but decompose during forward pass
        self.w_q = nn.Parameter(torch.randn(d_model, d_model))
        self.w_k = nn.Parameter(torch.randn(d_model, d_model))
        self.w_v = nn.Parameter(torch.randn(d_model, d_model))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Decompose projection matrices
        w_q_tt = to_tensor_train(self.w_q.reshape(self.n_heads, self.head_dim, 
                                                 self.n_heads, self.head_dim), self.tt_ranks)
        w_k_tt = to_tensor_train(self.w_k.reshape(self.n_heads, self.head_dim, 
                                                 self.n_heads, self.head_dim), self.tt_ranks)
        w_v_tt = to_tensor_train(self.w_v.reshape(self.n_heads, self.head_dim, 
                                                 self.n_heads, self.head_dim), self.tt_ranks)
        
        # Perform projections using TT format (simplified here)
        q = x @ self.w_q.reshape(self.d_model, self.d_model)
        k = x @ self.w_k.reshape(self.d_model, self.d_model)
        v = x @ self.w_v.reshape(self.d_model, self.d_model)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return out, attn


class HolographicAttention(nn.Module):
    """
    Backward compatibility wrapper around HyperbolicRadialAttention.
    
    DEPRECATED: Prefer HyperbolicRadialAttention for new code.
    This class is kept for backward compatibility with existing code.
    """
    def __init__(self, d_model, n_heads, use_tropical=False, use_hyperbolic=False):
        super().__init__()
        self.attention = HyperbolicRadialAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
            use_hyperbolic=use_hyperbolic,
            use_tropical=use_tropical,
            use_boundary_residual=True,
            curvature=-1.0,
        )
        
    def forward(self, x):
        """Forward pass - delegates to HyperbolicRadialAttention"""
        return self.attention(x)


def compute_entanglement_entropy(attention_matrix, n_regions=2):
    """Compute entanglement entropy between regions in the attention graph"""
    # Normalize if not already a probability distribution
    if not torch.allclose(attention_matrix.sum(dim=-1), torch.ones_like(attention_matrix.sum(dim=-1))):
        attention_matrix = torch.softmax(attention_matrix, dim=-1)

    seq_len = attention_matrix.shape[-1]
    region_size = seq_len // n_regions
    entropies = []

    for i in range(n_regions):
        region_start = i * region_size
        region_end = (i + 1) * region_size if i < n_regions - 1 else seq_len
        region_attn = attention_matrix[:, region_start:region_end, :]

        # Calculate density matrix (simplified)
        density_matrix = region_attn @ region_attn.transpose(-2, -1)
        # Add larger regularization and ensure correct device/dtype
        eps = 1e-5
        eye = torch.eye(density_matrix.shape[-1], device=density_matrix.device, dtype=density_matrix.dtype)
        try:
            eigenvalues = torch.linalg.eigvalsh(density_matrix + eps * eye)
            # Clamp eigenvalues to avoid log(0) or negative
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
            entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
            entropies.append(entropy.item())
        except Exception as e:
            print(f"Warning: eigvalsh failed for region {i} (error: {e}). Setting entropy to NaN.")
            entropies.append(float('nan'))

    return entropies


if __name__ == "__main__":
    # Test TensorTrainAttention
    print("Testing TensorTrainAttention...")
    x = torch.randn(2, 10, 64)
    tt_attn = TensorTrainAttention(d_model=64, n_heads=4, tt_ranks=[1, 4, 4, 4, 1])
    out, attn_weights = tt_attn(x)
    print(f"TT Attention - Input: {x.shape}, Output: {out.shape}")
    
    # Test HolographicAttention (backward compatibility)
    print("\nTesting HolographicAttention (deprecated)...")
    holo_attn = HolographicAttention(d_model=64, n_heads=4, use_hyperbolic=True)
    out, _ = holo_attn(x)
    print(f"Holographic Attention - Input: {x.shape}, Output: {out.shape}")
    
    # Test entanglement entropy
    print("\nTesting entanglement entropy...")
    dummy_attention = torch.rand(4, 10, 10)  # (heads, seq, seq)
    entropies = compute_entanglement_entropy(dummy_attention, n_regions=2)
    print(f"Entanglement entropies: {entropies}")