import psutil
import GPUtil
import argparse
import math
import os
import random
import sys
import gc
import nibabel as nib
from prettytable import PrettyTable
sys.path.append(".")
import time
from datetime import datetime
from tf32_utils import enable_tf32, get_model_size_stats
import create_submission_adapted
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



def print_memory_stats(location=""):
    """Print memory usage stats, I encountered OOM kills"""
    # GPU memory
    gpus = GPUtil.getGPUs()
    gpu_memory_used = f"{gpus[0].memoryUsed:.2f}MB" if gpus else "N/A"
    gpu_memory_total = f"{gpus[0].memoryTotal:.2f}MB" if gpus else "N/A"
    
    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    print(f"\nMemory Stats at {location}:")
    print(f"GPU Memory: {gpu_memory_used}/{gpu_memory_total}")
    print(f"CPU Memory: {cpu_memory:.2f}MB")
    print("-" * 50)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def diffusion_blend_sampling(model, shape, batch, diffusion, clip_denoised=True, model_kwargs=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    batch = batch.to(device)
    x_t = batch
    
    args = create_argparser().parse_args()
    sample_fn = (
        diffusion.p_sample_loop_known
        if not args.use_ddim
        else diffusion.ddim_sample_loop_known
    )
    
    class PatchModelWrapper(th.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x, t, **kwargs):
            # Parallelize to speed up
            return parallel_partition_noise(
                x_t=x,
                t=t,
                model=self.base_model,
                patch_size=3,
                partition_type='cross' if (t[0] % 3 == 0) else 'adjacent',
                device=device,
                model_kwargs=kwargs
            )
    
    wrapped_model = PatchModelWrapper(model)
    
    sample, x_noisy, org = sample_fn(
        wrapped_model,
        shape,
        x_t,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
    )
    
    return sample, x_noisy, org


def parallel_partition_noise(x_t, t, model, patch_size, partition_type, device, model_kwargs):
    """
    An implementation that processes patches in parallel instead of sequentially.
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    total_slices = x_t.shape[0]
    
    # Debug info
    # print(f"[DEBUG] Input tensor shape: {x_t.shape}, total_slices: {total_slices}")
    print(f"[DEBUG] Partition type: {partition_type}, patch_size: {patch_size}")
    
    with th.no_grad():
        sample_output = model(x_t[:1], t[:1], **model_kwargs)
        output_channels = sample_output.shape[1]
    
    final_output = th.zeros((total_slices, output_channels, x_t.shape[2], x_t.shape[3]), 
                          device=device)
    
    # Compute partition groups
    if partition_type == 'cross':
        patch_indices_groups = []
        for offset in range(patch_size):
            for start_idx in range(offset, total_slices, patch_size * 3):
                indices = []
                for i in range(3):  # Always using 3 here
                    idx = (start_idx + i * patch_size) % total_slices
                    if idx < total_slices:
                        indices.append(idx)
                if len(indices) == patch_size:
                    patch_indices_groups.append(indices)
    else:  # adjacent
        patch_indices_groups = []
        m = random.randint(0, patch_size-1)
        if m:
            patch_indices_groups.append(list(range(0, m)))
        for start_idx in range(m, total_slices, patch_size):
            end_idx = min(start_idx + patch_size, total_slices)
            indices = list(range(start_idx, end_idx))
            if indices:
                patch_indices_groups.append(indices)
    
    # print(f"[DEBUG] Generated {len(patch_indices_groups)} patch groups")
    
    # Process in batches of patch groups rather than individually
    # This maintains the patching logic while improving efficiency
    batch_size = 16
    
    for batch_start in range(0, len(patch_indices_groups), batch_size):
        batch_end = min(batch_start + batch_size, len(patch_indices_groups))
        current_batch_groups = patch_indices_groups[batch_start:batch_end]
        
        # Count total patches in this batch
        total_patches_in_batch = sum(len(indices) for indices in current_batch_groups)
        
        # Create batch of patches
        all_patches = []
        all_indices = []
        
        for indices in current_batch_groups:
            # Extract patches for this group
            patches = x_t[indices]
            all_patches.append(patches)
            all_indices.extend(indices)
        
        # Stack all patches into a single batch
        if all_patches:
            # Concatenate all patches into a single batch
            patches_batch = th.cat(all_patches, dim=0)
            
            # Create timesteps
            t_batch = th.ones(patches_batch.shape[0], device=device) * t[0]
            
            # Process the batch of patches
            with th.no_grad():
                outputs_batch = model(patches_batch, t_batch, **model_kwargs)
            
            # Put results back to final output
            start_idx = 0
            for indices in current_batch_groups:
                end_idx = start_idx + len(indices)
                for i, idx in enumerate(indices):
                    final_output[idx] = outputs_batch[start_idx + i]
                start_idx = end_idx
    
    return final_output


def optimize_for_tensor_cores(model):
    """
    An attempt to utilize NVIDIA A100's Tensor Cores for higher throughput.
    """
    # Check if we're on an A100
    device_name = th.cuda.get_device_name() if th.cuda.is_available() else "CPU"
    is_ampere_or_newer = any(gpu in device_name for gpu in ["A100"])
    
    if not is_ampere_or_newer:
        logger.log(f"Tensor Core optimization skipped - current device: {device_name}")
        return model
    
    class TensorCoreOptimizer(th.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
            # Enable optimizations for Tensor Cores, though can be done separately
            th.backends.cuda.matmul.allow_tf32 = True
            th.backends.cudnn.allow_tf32 = True
            
            # Enable memory format optimizations for Tensor Cores
            self.channels_last = True  # Use channels_last memory format which is optimal for convolutions
            
            # Copy all attributes from base_model to avoid recursion in __getattr__, I encountered an infinite recursion error previously
            for name in dir(base_model):
                if not name.startswith('__') and not hasattr(self, name):
                    try:
                        setattr(self, name, getattr(base_model, name))
                    except (AttributeError, RuntimeError):
                        pass  # Skip attributes that can't be copied
            
            logger.log(f"Tensor Core optimizations enabled for {device_name}")
            
        def forward(self, x, timesteps, **kwargs):
            # Convert to channels_last format if it makes sense for the input
            if self.channels_last and x.dim() == 4:
                x = x.to(memory_format=th.channels_last)
            
            # Use automatic mixed precision for Tensor Cores
            with th.amp.autocast('cuda', enabled=True):  # Updated to new syntax
                # Ensure the inputs use the correct memory layout
                output = self.base_model(x, timesteps, **kwargs)
            
            # Return to standard memory format if needed
            if self.channels_last and output.dim() == 4:
                output = output.contiguous()
                
            return output
    
    tensor_core_model = TensorCoreOptimizer(model)
    logger.log("Model optimized for A100 Tensor Cores")
    
    return tensor_core_model

def compile_diffusionblend_unet(model):
    """
    Compile the UNet component of a DiffusionBlend++ model. Compilation promises speedup by:

    TorchDynamo, AOTAutograd, Backend Compiler etc. that optimizes the unet into a cuda graph with operator fusion and mem layout optimizations.
    """
    # Skip if torch.compile is not available (PyTorch < 2.0)
    if not hasattr(th, 'compile'):
        logger.log("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    # Verify PyTorch version
    from packaging import version
    
    if version.parse(th.__version__) < version.parse("2.0.0"):
        logger.log(f"torch.compile requires PyTorch 2.0+, but found {th.__version__}")
        return model
    
    try:
        # Directly compile the model since it is already a DiffusionBlendUNet
        logger.log("Compiling DiffusionBlend UNet...")
        compiled_model = th.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False
        )
        logger.log("UNet compilation successful")
        return compiled_model
    
    except Exception as e:
        logger.log(f"UNet compilation failed: {str(e)}")
        logger.log("Continuing with uncompiled model")
        return model

def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    # Set up persistent compilation cache
    # setup_torch_cache()

    today = datetime.now()
    logger.log("SAMPLING START " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
    )
    data = iter(datal)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    
    # Get model stats
    stats = get_model_size_stats(model)
    logger.log(f"Model parameters: {stats['parameters']:,}")
    logger.log(f"Model size: {stats['model_size_mb']:.2f} MB")
    
    # Enable TF32
    if args.enable_tf32:
        success, message = enable_tf32()
        logger.log(f"TF32 status: {message}")

    model.eval()
    model.use_checkpoint = False 


    if args.use_compile:
        logger.log("Attempting to compile DiffusionBlend UNet...")
        model = compile_diffusionblend_unet(model)

    """
    # Force compilation using cache artifacts
    logger.log("Enforcing compiled model usage...")
    model = force_compile_model(model)
    # No longer using, buggy
    """

    # model = optimize_for_tensor_cores(model)

    dir_path = str(args.data_dir)
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, path)):
            count += 1

    for d in range(0, len(ds)):
        print("sampling file ", str(d + 1), " of ", str(count), "...")
        batch, path, slicedict = next(data)
        p_s = (
            path[0].split("/")[-1].split(".")[0].split("-")[2]
            + "-"
            + path[0].split("/")[-1].split(".")[0].split("-")[3]
        )

        if not os.path.isfile(
            str(args.log_dir) + "/BraTS-GLI-" + str(p_s) + "-t1n-inference.nii.gz"
        ):
            print("file does not exist yet, starting sampling...")
            generated_3D = np.asarray(batch[:, 0, :, :, :].squeeze())

            # Process all slices together to maintain consistency
            out_batch = []
            s_sub = []

            for s in slicedict:
                out_batch.append(batch[..., s])
                s_sub.append(s)

            out_batch = th.stack(out_batch)
            out_batch = out_batch.squeeze(1)
            out_batch = out_batch.squeeze(4)

            c = th.randn_like(out_batch[:, :1, ...])
            out_batch = th.cat((out_batch, c), dim=1)  # Now should be [B, 3, H, W]

            model_kwargs = {}
            
            # Using DiffusionBlend++ sampling
            sample, x_noisy, org = diffusion_blend_sampling(
                model,
                (out_batch.shape[0], 3, args.image_size, args.image_size),
                out_batch,
                diffusion,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            # Store results
            # Below are for resolving OOM. Without these OOM may occur even with quite a small batch size.
            sa = sample.clone().detach().cpu()  # Immediately move to CPU after getting sample
            sa = sa.squeeze(1)
            sam = np.asarray(sa)  # No need for .detach().cpu() since it's already on CPU
            del sa  # Free memory
            th.cuda.empty_cache()
            
            if args.use_ddim:
                sam = sam[:,0,:,:]

            # Update 3D volume
            for idx, s in enumerate(s_sub):
                generated_3D[..., s] = sam[idx, :, :]
            del sam  # Free memory after updating
            gc.collect()

            # Before saving
            nib.save(
                nib.Nifti1Image(generated_3D, None),
                (str(args.log_dir) + "/BraTS-GLI-" + str(p_s) + "-t1n-inference.nii.gz"),
            )
            del generated_3D  # Free memory after saving
            gc.collect()
            th.cuda.empty_cache()
        else:
            print("file exists already, skipping it")

    create_submission_adapted.adapt(
        input_data=args.data_dir,
        samples_dir=args.log_dir,
        adapted_samples_dir=args.adapted_samples,
    )


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        adapted_samples="",
        subbatch=16,
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        weight_only_fp16=False,
        enable_tf32=True,
        use_compile=True, # no persistent compiled artifact?
        model_path="",
        patch_size=3,
    )
    
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()