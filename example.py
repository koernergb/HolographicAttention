import torch
from code import HolographicAttention, TensorTrainAttention, compute_entanglement_entropy

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a sample input sequence
    batch_size = 2
    seq_length = 8
    d_model = 64
    n_heads = 4
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Initialize different attention mechanisms
    print("\n1. Testing Holographic Attention")
    holo_attn = HolographicAttention(
        d_model=d_model, 
        n_heads=n_heads,
        use_tropical=True,  # Enable tropical geometry
        use_hyperbolic=True  # Enable hyperbolic space
    )
    
    # Forward pass
    output, attention = holo_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Compute entanglement entropy
    entropy = compute_entanglement_entropy(attention[0], n_regions=2)
    print(f"\nEntanglement entropy between regions: {entropy}")
    
    # Test tensor train attention
    print("\n2. Testing Tensor Train Attention")
    tt_attn = TensorTrainAttention(
        d_model=d_model,
        n_heads=n_heads,
        tt_ranks=[1, 8, 8, 8, 1]  # Tensor train ranks
    )
    
    # Forward pass
    output, attention = tt_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Visualize attention patterns
    print("\n3. Attention Pattern Analysis")
    print("First head attention pattern:")
    print(attention[0, 0].detach().numpy())  # First batch, first head
    
    # Compare computational efficiency
    print("\n4. Performance Comparison")
    import time
    
    # Time holographic attention
    start = time.time()
    for _ in range(10):
        holo_attn(x)
    holo_time = (time.time() - start) / 10
    
    # Time tensor train attention
    start = time.time()
    for _ in range(10):
        tt_attn(x)
    tt_time = (time.time() - start) / 10
    
    print(f"Holographic attention time: {holo_time:.4f}s")
    print(f"Tensor train attention time: {tt_time:.4f}s")

if __name__ == "__main__":
    main() 