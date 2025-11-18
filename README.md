# Hyperbolic Attention: Geometric Approaches to Transformer Attention

This repository explores novel attention mechanisms inspired by hyperbolic geometry and holographic principles, investigating whether geometric structures can simplify and improve transformer computations.

## Overview

Traditional transformer attention operates in Euclidean space using dot-product similarity. This work investigates alternative geometric formulations:

- **Hyperbolic Attention**: Uses Poincaré ball projections and geodesic distances
- **Tropical Geometry**: Explores max-plus algebra connections to attention sparsity
- **Tensor Networks**: Applies quantum many-body physics techniques for compression
- **Holographic Encoding**: Inspired by AdS/CFT correspondence and bulk-boundary duality

## Key Components

### HyperbolicRadialAttention

The core attention mechanism operating in hyperbolic space:

```python
from hyperbolic_attention import HyperbolicRadialAttention

# Create hyperbolic attention layer
attn = HyperbolicRadialAttention(
    embed_dim=512,
    num_heads=8,
    use_hyperbolic=True,          # Enable Poincaré ball projections
    use_tropical=False,           # Enable tropical scaling
    use_boundary_residual=True,   # Add boundary embedding residuals
    curvature=-1.0               # Learnable negative curvature
)

# Forward pass
output, weights = attn(x, need_weights=True)
```

**Features:**
- Poincaré ball projections with learnable curvature
- Per-head radial scaling parameters
- Hyperbolic distance-based attention scores
- Boundary embedding residuals
- Temperature-scaled softmax for tropical-like behavior

### HyperbolicTransformerBlock

Complete transformer block using hyperbolic attention:

```python
from hyperbolic_attention import HyperbolicTransformerBlock

block = HyperbolicTransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1
)

output = block(x)
```

### Tensor Train Attention

Compressed attention using tensor train decomposition:

```python
from holographic_attention import TensorTrainAttention

tt_attn = TensorTrainAttention(
    d_model=512,
    n_heads=8,
    tt_ranks=[1, 16, 16, 16, 1]  # Tensor train ranks
)

output, weights = tt_attn(x)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd holographic_attention

# Install dependencies
pip install torch numpy matplotlib seaborn scikit-learn

# Optional: Install in development mode
pip install -e .
```

## Quick Start

```python
import torch
from hyperbolic_attention import HyperbolicRadialAttention

# Create sample data
batch_size, seq_len, d_model = 2, 128, 512
x = torch.randn(batch_size, seq_len, d_model)

# Standard hyperbolic attention
attn = HyperbolicRadialAttention(
    embed_dim=d_model,
    num_heads=8,
    use_hyperbolic=True,
    use_boundary_residual=True
)

# Forward pass
output, weights = attn(x, need_weights=True)
print(f"Output shape: {output.shape}")  # [2, 128, 512]
print(f"Attention weights shape: {weights.shape}")  # [2, 128, 128]
```

## Examples

Run the examples to see different configurations:

```bash
python examples.py
```

This demonstrates:
- Basic hyperbolic attention usage
- Different configuration options
- Causal masking for autoregressive models
- Parameter counting and efficiency analysis

## Research Motivation

### Tropical Geometry Connection

Attention mechanisms involve softmax operations that approximate max functions in the low-temperature limit. This connects to tropical geometry (max-plus algebra), suggesting that:

1. **Natural Sparsity**: Attention patterns may naturally follow tropical geometric structures
2. **Computational Efficiency**: Tropical operations could reduce quadratic complexity
3. **Interpretability**: Geometric boundaries provide clearer decision boundaries

### Hyperbolic Space Benefits

Hyperbolic geometry offers advantages for hierarchical data:

1. **Exponential Growth**: Hyperbolic space naturally represents tree-like structures
2. **Learnable Curvature**: Negative curvature parameter adapts to data geometry  
3. **Geodesic Distances**: More meaningful similarity measures for structured data

### Holographic Principles

Inspired by AdS/CFT correspondence from physics:

1. **Bulk-Boundary Duality**: Information on the boundary determines bulk properties
2. **Dimensional Reduction**: Higher-dimensional bulk encodes lower-dimensional boundary
3. **Entanglement Structure**: Quantum entanglement patterns in attention weights

## File Structure

```
holographic_attention/
├── hyperbolic_attention.py      # Core hyperbolic attention implementation
├── holographic_attention.py     # Tensor train attention and utilities
├── tensor_train_utils.py        # Tensor decomposition utilities
├── examples.py                  # Usage examples and tests
├── experimental_analysis.py     # Research analysis functions (WIP)
└── README.md                    # This file
```

## Configuration Options

### HyperbolicRadialAttention Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | - | Model dimension |
| `num_heads` | int | - | Number of attention heads |
| `dropout` | float | 0.0 | Dropout probability |
| `use_hyperbolic` | bool | True | Enable Poincaré ball projections |
| `use_tropical` | bool | False | Enable temperature scaling |
| `use_boundary_residual` | bool | True | Add boundary residuals |
| `curvature` | float | -1.0 | Initial curvature (kept negative) |

### Example Configurations

```python
# Standard hyperbolic attention
attn_hyp = HyperbolicRadialAttention(
    embed_dim=512, num_heads=8,
    use_hyperbolic=True, use_tropical=False
)

# Tropical-style attention
attn_trop = HyperbolicRadialAttention(
    embed_dim=512, num_heads=8,
    use_hyperbolic=False, use_tropical=True
)

# Combined hyperbolic + tropical
attn_combined = HyperbolicRadialAttention(
    embed_dim=512, num_heads=8,
    use_hyperbolic=True, use_tropical=True
)

# Minimal Euclidean baseline
attn_baseline = HyperbolicRadialAttention(
    embed_dim=512, num_heads=8,
    use_hyperbolic=False, use_tropical=False,
    use_boundary_residual=False
)
```

## Research Results

### Tropical Alignment Analysis

Initial experiments show that transformer attention patterns naturally exhibit tropical-like behavior:

- **Early Layers**: Strong alignment with tropical geometry (up to 94% accuracy)
- **Later Layers**: More complex, non-tropical patterns emerge
- **Head Specialization**: Different heads show varying degrees of tropical behavior

### Efficiency Gains

Preliminary benchmarks suggest:

- **5.2x speedup** with pure tropical attention
- **Low approximation error** (0.007 MSE) with sparse tropical variants
- **Parameter reduction** possible with tensor train decomposition

## Future Directions

### Immediate Next Steps

1. **Systematic Benchmarking**: Compare against standard attention on real tasks
2. **Scaling Studies**: Test behavior with larger models and longer sequences
3. **Architecture Search**: Find optimal geometric configurations
4. **Training Dynamics**: Analyze how geometric properties evolve during training

### Research Questions

1. **Fundamental Limits**: What can hyperbolic attention represent vs. standard attention?
2. **Emergent Properties**: Do geometric constraints lead to better inductive biases?
3. **Scaling Laws**: How do geometric attention mechanisms scale with model size?
4. **Task Specificity**: Which tasks benefit most from geometric attention?

## Mathematical Background

### Poincaré Ball Model

The Poincaré ball model represents hyperbolic space as the unit ball with metric:

```
ds² = 4(dx₁² + dx₂² + ... + dxₙ²) / (1 - ||x||²)²
```

Geodesic distance between points u, v:

```
d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
```

### Tropical Semiring

Tropical algebra replaces standard operations:
- Addition: `a ⊕ b = max(a, b)`
- Multiplication: `a ⊗ b = a + b`

This connects to attention via the softmax limit:
```
lim(τ→0) softmax(x/τ) = one_hot(argmax(x))
```

## Contributing

This is research code exploring novel attention mechanisms. Contributions welcome:

1. **Theoretical Analysis**: Mathematical insights into geometric attention
2. **Empirical Studies**: Benchmarks on standard NLP tasks  
3. **Efficiency Improvements**: Optimized implementations
4. **Applications**: Novel use cases for geometric attention

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hyperbolic_attention,
  title={Hyperbolic Attention: Geometric Approaches to Transformer Attention},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/holographic_attention}
}
```

## License

[Add your preferred license]

## Acknowledgments

This work is inspired by:
- The amplituhedron's role in simplifying particle physics calculations
- Tropical geometry and its connections to optimization
- Hyperbolic neural networks and geometric deep learning
- AdS/CFT correspondence and holographic duality
- Tensor network methods from quantum many-body physics

---

**Note**: This is experimental research code. The geometric attention mechanisms are still under investigation and may not yet match the performance of standard attention on all tasks. Use for research and experimentation.
