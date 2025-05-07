"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3

print(f"[DEBUG] dist_util.py: CUDA available: {th.cuda.is_available()}")
print(f"[DEBUG] dist_util.py: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")


if th.cuda.is_available():
    print(f"[DEBUG] dist_util.py: Device count: {th.cuda.device_count()}")
    for i in range(th.cuda.device_count()):
        print(f"[DEBUG] dist_util.py: Device {i}: {th.cuda.get_device_name(i)}")



# Check setup_dist function
def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        print("[DEBUG] dist_util.setup_dist(): already initialized")
        return
    
    print(f"[DEBUG] dist_util.setup_dist(): initializing, WORLD_SIZE={os.environ.get('WORLD_SIZE', 'Not set')}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = "127.0.1.1"  # comm.bcast(hostname, root=0)
    os.environ["RANK"] = "0"  # str(comm.rank)
    os.environ["WORLD_SIZE"] = "1"  # str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


# Check the implementation of dev() function
def dev():
    """
    Get the device to use for th.distributed.
    """
    print(f"[DEBUG] dist_util.dev() called: CUDA available={th.cuda.is_available()}")
    
    if th.cuda.is_available():
        device_id = os.environ.get("LOCAL_RANK", 0)
        print(f"[DEBUG] dist_util.dev(): LOCAL_RANK={device_id}")
        try:
            return th.device(f"cuda:{device_id}")
        except:
            return th.device("cuda")
    return th.device("cpu")



def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    
    # Set weights_only=True for security unless explicitly overridden
    # This prevents arbitrary code execution from untrusted models
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = True
        
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
