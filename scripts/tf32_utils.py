# tf32_utils.py
import torch
import torch.nn as nn
import time
import gc

def enable_tf32():
    """
    Simple, robust implementation to enable TF32 on NVIDIA Ampere or newer GPUs.
    """
    # Just enable TF32 without conditional checks - if not supported, it will be ignored
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    
    # Check if CUDA is available for logging purposes only
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            return True, f"TF32 enabled on {device_name}"
        except Exception as e:
            return True, f"TF32 enabled on CUDA device (error getting device name: {e})"
    else:
        return False, "CUDA not available, TF32 not enabled"

def get_model_size_stats(model):
    """Get memory usage statistics for a model."""
    params = sum(p.numel() for p in model.parameters())
    memory_mb = params * 4 / (1024 * 1024)  # Assuming float32, 4 bytes per parameter
    
    return {
        "parameters": params,
        "model_size_mb": memory_mb
    }