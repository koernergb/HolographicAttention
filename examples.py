"""
Usage examples for hyperbolic attention mechanisms.
"""

import torch
from hyperbolic_attention import HyperbolicRadialAttention, HyperbolicTransformerBlock


def test_hyperbolic_attention():
    """Test the HyperbolicRadialAttention layer"""
    print("=" * 50)
    print("Testing HyperbolicRadialAttention")
    print("=" * 50)
    
    # Create attention layer
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
    
    # Test input
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output, weights = attn(x, need_weights=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape if weights is not None else None}")
    print(f"Parameters: {sum(p.numel() for p in attn.parameters())}")
    print()


def test_hyperbolic_transformer_block():
    """Test the HyperbolicTransformerBlock"""
    print("=" * 50)
    print("Testing HyperbolicTransformerBlock")
    print("=" * 50)
    
    # Create transformer block
    block = HyperbolicTransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=128,
        dropout=0.1
    )
    
    # Test input
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = block(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters())}")
    print()


def test_different_configurations():
    """Test different attention configurations"""
    print("=" * 50)
    print("Testing Different Configurations")
    print("=" * 50)
    
    configs = [
        {"use_hyperbolic": True, "use_tropical": False, "use_boundary_residual": True},
        {"use_hyperbolic": False, "use_tropical": True, "use_boundary_residual": True},
        {"use_hyperbolic": True, "use_tropical": True, "use_boundary_residual": False},
        {"use_hyperbolic": False, "use_tropical": False, "use_boundary_residual": False},
    ]
    
    B, L, D = 2, 8, 32
    x = torch.randn(B, L, D)
    
    for i, config in enumerate(configs):
        print(f"Config {i+1}: {config}")
        
        attn = HyperbolicRadialAttention(
            embed_dim=D,
            num_heads=4,
            **config
        )
        
        output, _ = attn(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in attn.parameters())}")
        print()


def test_with_masking():
    """Test attention with causal masking"""
    print("=" * 50)
    print("Testing with Causal Masking")
    print("=" * 50)
    
    attn = HyperbolicRadialAttention(
        embed_dim=32,
        num_heads=2,
        use_hyperbolic=True,
        use_boundary_residual=True,
    )
    
    B, L, D = 1, 8, 32
    x = torch.randn(B, L, D)
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)
    
    print(f"Input shape: {x.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    
    # Forward pass with mask
    output, weights = attn(x, attn_mask=causal_mask, need_weights=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Check that future positions have zero attention
    print("Attention weights (should be lower triangular):")
    print(weights[0].detach().numpy().round(3))
    print()


if __name__ == "__main__":
    test_hyperbolic_attention()
    test_hyperbolic_transformer_block()
    test_different_configurations()
    test_with_masking()
