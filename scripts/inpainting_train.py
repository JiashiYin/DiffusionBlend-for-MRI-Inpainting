"""

Generate a large batch of image samples from a model and save them as a large

numpy array. This can be used to produce samples for FID evaluation.

"""


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



import create_submission_adapted

#import eval_sam

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
    """Print memory usage statistics"""
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
    
    # Instead of computing patches sequentially, process them in parallel
    class ParallelPatchModelWrapper(th.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x, t, **kwargs):
            # Using batch processing for partition computation
            return parallel_partition_noise(
                x_t=x,
                t=t,
                model=self.base_model,
                patch_size=3,
                partition_type='cross' if (t[0] % 3 == 0) else 'adjacent',
                device=device,
                model_kwargs=kwargs
            )
    
    wrapped_model = ParallelPatchModelWrapper(model)
    
    sample, x_noisy, org = sample_fn(
        wrapped_model,
        shape,
        x_t,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
    )
    
    return sample, x_noisy, org


def parallel_partition_noise(x_t, t, model, patch_size, partition_type, device, model_kwargs):
    if model_kwargs is None:
        model_kwargs = {}
    
    total_slices = x_t.shape[0]
    
    # First, get output shape from a single prediction to initialize correctly
    with th.no_grad():
        sample_output = model(x_t[:1], t[:1], **model_kwargs)
        output_channels = sample_output.shape[1]
    
    final_output = th.zeros((total_slices, output_channels, x_t.shape[2], x_t.shape[3]), 
                          device=device)
    
    # Compute partition groups more efficiently
    if partition_type == 'cross':
        # Create all patch indices in a vectorized way
        all_patch_indices = []
        for offset in range(patch_size):
            for start_idx in range(offset, total_slices, patch_size * 3):
                indices = []
                for i in range(patch_size):
                    idx = (start_idx + i * patch_size) % total_slices
                    if idx < total_slices:
                        indices.append(idx)
                if len(indices) == patch_size:
                    all_patch_indices.append(indices)
    else:  # adjacent
        all_patch_indices = []
        m = random.randint(0, patch_size-1)
        if m:
            all_patch_indices.append(list(range(0, m)))
        for start_idx in range(m, total_slices, patch_size):
            end_idx = min(start_idx + patch_size, total_slices)
            indices = list(range(start_idx, end_idx))
            if indices:
                all_patch_indices.append(indices)
    
    # Process patches in parallel rather than sequentially
    # First, we determine optimal batch size based on GPU memory
    num_patches = len(all_patch_indices)
    
    # Estimate optimal batch size (could be adjusted based on available memory)
    mem_info = th.cuda.mem_get_info() if th.cuda.is_available() else (0, 0)
    free_memory = mem_info[0]
    single_patch_memory = x_t[0:patch_size].element_size() * x_t[0:patch_size].nelement()
    optimal_batch_size = max(1, min(num_patches, int(free_memory * 0.7 / single_patch_memory)))
    
    for i in range(0, num_patches, optimal_batch_size):
        # Process patches in batches
        batch_indices = all_patch_indices[i:i+optimal_batch_size]
        batch_size = len(batch_indices)
        
        # Flatten the batch indices for indexing
        flat_indices = [idx for patch_indices in batch_indices for idx in patch_indices]
        flat_batch = x_t[flat_indices]
        
        # Reshape to create proper batches for each patch
        patch_batches = flat_batch.view(batch_size, patch_size, *flat_batch.shape[1:])
        
        # Process all patches in parallel
        with th.no_grad():
            # Flatten and process
            reshaped_patches = patch_batches.reshape(-1, *patch_batches.shape[2:])
            t_expanded = th.ones(reshaped_patches.shape[0], device=device) * t[0]
            
            # Forward pass for all patches at once
            all_outputs = model(reshaped_patches, t_expanded, **model_kwargs)
            
            # Reshape back to batch of patches
            all_outputs = all_outputs.view(batch_size, patch_size, *all_outputs.shape[1:])
            
            # Store each patch's prediction in the final output
            for pidx, patch_indices in enumerate(batch_indices):
                for j, idx in enumerate(patch_indices):
                    final_output[idx] = all_outputs[pidx, j]
    
    return final_output

def compute_partition_noise(x_t, t, model, patch_size, partition_type, device, model_kwargs):
    if model_kwargs is None:
        model_kwargs = {}
    
    total_slices = x_t.shape[0]
    
    # First, get output shape from a single prediction to initialize correctly
    with th.no_grad():
        sample_output = model(x_t[:1], t[:1], **model_kwargs)
        output_channels = sample_output.shape[1]  # Should be C*2 (mean and variance)
    
    # Initialize output tensor with correct shape
    final_output = th.zeros((total_slices, output_channels, x_t.shape[2], x_t.shape[3]), 
                          device=device)
    
    # Compute partition groups
    if partition_type == 'cross':
        patch_indices_groups = []
        for offset in range(patch_size):
            for start_idx in range(offset, total_slices, patch_size * 3):
                indices = []
                for i in range(3):
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
    
    # Process each patch and store results
    for indices in patch_indices_groups:
        patch = x_t[indices]
        patch_t = th.ones(len(indices), device=device) * t[0]
        
        with th.no_grad():
            patch_output = model(patch, patch_t, **model_kwargs)
            # Store each slice's prediction in the final output
            for i, idx in enumerate(indices):
                final_output[idx] = patch_output[i]
    
    return final_output






def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
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
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    model.use_checkpoint = False 
    dir_path = str(args.data_dir)
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, path)):
            count += 1


    #print_memory_stats("Start of main()")

    for d in range(0, 1):
        print("sampling file ", str(d + 1), " of ", str(count), "...")
        batch, path, slicedict = next(data)
        #print_memory_stats(f"After loading batch {d}")
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
            sa = sample.clone().detach().cpu()  # Immediately move to CPU after getting sample
            sa = sa.squeeze(1)
            sam = np.asarray(sa)  # No need for .detach().cpu() since it's already on CPU
            #print(f"for p sample, sam shape is {sam.shape}")
            del sa  # Free memory
            th.cuda.empty_cache()
            
            #print(sam.shape)
            
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

        # gt_dir="",

        adapted_samples="",

        subbatch=16,

        clip_denoised=True,

        batch_size=1,

        use_ddim=True,  # Set to True as we're using DDIM-style steps

        model_path="",

        patch_size=3,  # Added for DiffusionBlend++

    )
    
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)

    return parser





if __name__ == "__main__":

    main()
