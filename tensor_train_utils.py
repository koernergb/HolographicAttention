import torch
import numpy as np


def to_tensor_train(tensor, ranks):
    """Convert a tensor to tensor train format with specified ranks"""
    shape = tensor.shape
    d = len(shape)
    
    # Initialize cores
    cores = []
    
    # Start with original tensor
    remaining = tensor
    
    # For all dimensions except the last
    for i in range(d-1):
        # Reshape and perform SVD
        remaining = remaining.reshape(-1, np.prod(shape[i+1:]))
        u, s, v = torch.svd(remaining)
        
        # Truncate to rank
        r = min(ranks[i], s.shape[0])
        u = u[:, :r]
        s = s[:r]
        v = v[:, :r]
        
        # Create core
        core = u.reshape(-1, shape[i], r)
        cores.append(core)
        
        # Update remaining tensor
        remaining = torch.diag(s) @ v.t()
    
    # Add last core
    cores.append(remaining.reshape(ranks[-2], shape[-1], -1))
    
    return cores


def reconstruct_from_tt(cores):
    """Reconstruct original tensor from TT cores"""
    result = cores[0]
    for i in range(1, len(cores)):
        # Contract with next core
        result = torch.einsum('...ij,jkl->...ikl', result, cores[i])
    return result.squeeze()


if __name__ == "__main__":
    # Test tensor train decomposition
    test_tensor = torch.randn(4, 4, 4, 4)
    tt_cores = to_tensor_train(test_tensor, [1, 4, 4, 4, 1])
    reconstructed = reconstruct_from_tt(tt_cores)
    error = torch.norm(test_tensor - reconstructed) / torch.norm(test_tensor)
    print(f"Relative reconstruction error: {error:.6f}")
