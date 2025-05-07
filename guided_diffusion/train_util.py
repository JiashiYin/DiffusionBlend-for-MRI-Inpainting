import copy

import functools

import os

import random

import math

import numpy as np

import nibabel as nib

import blobfile as bf

import torch as th


import torch.cuda.amp as amp

import torch.distributed as dist

from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torch.optim import AdamW



from . import dist_util, logger

from .fp16_util import MixedPrecisionTrainer

from .nn import update_ema

from .resample import LossAwareSampler, UniformSampler

# Diffusionblend

# For ImageNet experiments, this was a good default value.

# We found that the lg_loss_scale quickly climbed to

# 20-21 within the first ~1K steps of training.


INITIAL_LOG_LOSS_SCALE = 20.0

random.seed(10)



def visualize(img):

    _min = img.min()

    _max = img.max()

    normalized_img = (img - _min) / (_max - _min)

    return normalized_img





class TrainLoop:

    def __init__(

        self,

        *,

        model,

        classifier,

        diffusion,

        data,

        dataloader,

        batch_size,

        microbatch,

        lr,

        ema_rate,

        log_interval,

        save_interval,

        resume_checkpoint,

        use_fp16=False,

        fp16_scale_growth=1e-3,

        schedule_sampler=None,

        weight_decay=0.0,

        lr_anneal_steps=0,

        patch_size=3,  # Add this parameter

    ):

        self.model = model

        self.dataloader = dataloader

        self.classifier = classifier

        self.diffusion = diffusion

        self.data = data

        self.batch_size = batch_size

        self.microbatch = microbatch if microbatch > 0 else batch_size

        self.lr = lr

        self.ema_rate = (

            [ema_rate]

            if isinstance(ema_rate, float)

            else [float(x) for x in ema_rate.split(",")]

        )

        self.log_interval = log_interval

        self.save_interval = save_interval

        self.resume_checkpoint = resume_checkpoint

        self.use_fp16 = use_fp16

        self.fp16_scale_growth = fp16_scale_growth

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

        self.weight_decay = weight_decay

        self.lr_anneal_steps = lr_anneal_steps

        self.patch_size = patch_size  # Add this line



        self.step = 0

        self.resume_step = 0

        self.global_batch = self.batch_size * dist.get_world_size()



        self.sync_cuda = th.cuda.is_available()



        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(

            model=self.model,

            use_fp16=self.use_fp16,

            fp16_scale_growth=fp16_scale_growth,

        )



        self.opt = AdamW(

            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay

        )

        if self.resume_step:

            self._load_optimizer_state()

            # Model was resumed, either due to a restart or a checkpoint

            # being specified at the command line.

            self.ema_params = [

                self._load_ema_parameters(rate) for rate in self.ema_rate

            ]

        else:

            self.ema_params = [

                copy.deepcopy(self.mp_trainer.master_params)

                for _ in range(len(self.ema_rate))

            ]



        if th.cuda.is_available():

            self.use_ddp = True

            self.ddp_model = DDP(

                self.model,

                device_ids=[dist_util.dev()],

                output_device=dist_util.dev(),

                broadcast_buffers=False,

                bucket_cap_mb=128,

                find_unused_parameters=False,

            )

        else:

            if dist.get_world_size() > 1:

                logger.warn(

                    "Distributed training requires CUDA. "

                    "Gradients will not be synchronized properly!"

                )

            self.use_ddp = False

            self.ddp_model = self.model

        self.moving_avg_loss = 0.0
        self.ma_window = 100  # Adjust window size as needed
        self.loss_history = []

    def _load_and_sync_parameters(self):

        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint



        if resume_checkpoint:

            print("resume model")

            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            if dist.get_rank() == 0:

                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")

                self.model.load_state_dict(

                    dist_util.load_state_dict(

                        resume_checkpoint, map_location=dist_util.dev()

                    )

                )



        dist_util.sync_params(self.model.parameters())



    def _load_ema_parameters(self, rate):

        ema_params = copy.deepcopy(self.mp_trainer.master_params)



        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:

            if dist.get_rank() == 0:

                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

                state_dict = dist_util.load_state_dict(

                    ema_checkpoint, map_location=dist_util.dev()

                )

                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)



        dist_util.sync_params(ema_params)

        return ema_params



    def _load_optimizer_state(self):

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        opt_checkpoint = bf.join(

            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"

        )

        if bf.exists(opt_checkpoint):

            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")

            state_dict = dist_util.load_state_dict(

                opt_checkpoint, map_location=dist_util.dev()

            )

            self.opt.load_state_dict(state_dict)


    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        self.mp_trainer.zero_grad() # Zero gradients at start
        
        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            samples_in_batch = 0
            batchsize = 4
            while samples_in_batch < batchsize:
                try:
                    batch, cond, path, slicedict = next(data_iter)
                    
                    # Create patches for the entire volume
                    patches = self.create_volume_patches(batch, cond, slicedict, self.patch_size)
                    
                    if patches:
                        patch_idx = random.randint(0, len(patches) - 1)
                        patch_batch, patch_cond = patches[patch_idx]
                        # Run forward-backward but don't step optimizer yet
                        sample = self.forward_backward_accumulate(patch_batch, patch_cond, batchsize)
                        samples_in_batch += 1
                        i += 1

                except StopIteration:
                    print("Reached end of dataset, reinitializing data loader")
                    data_iter = iter(self.dataloader)
                    continue

            # After accumulating gradients for batch_size samples, take optimizer step
            took_step = self.optimize(self.opt)
            self.mp_trainer.zero_grad()  # Zero gradients after stepping
            
            self.log_step()
            
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                print(self.moving_avg_loss)
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            self._anneal_lr()

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def create_volume_patches(self, batch, cond, slicedict, patch_size):
        """Creates patches from the entire volume, ensuring each patch has exactly patch_size slices."""
        patches = []
        total_slices = len(slicedict)
        
        # Determine if we should use cross partition
        use_cross = (self.step + self.resume_step) % patch_size == 0
        
        # Random offset in range [0, patch_size)
        m = random.randint(0, patch_size - 1)
        
        if use_cross:
            # Create cross partitions
            # Can start from any position since we wrap around
            for start_pos in range(total_slices):
                # Create a patch with 3 slices at same relative positions
                patch_indices = []
                for i in range(patch_size):
                    idx = (start_pos + i * patch_size) % total_slices
                    patch_indices.append(idx)
                
                # We'll always have patch_size slices due to wrapping
                patch_batch = []
                patch_cond = []
                for idx in patch_indices:
                    patch_batch.append(batch[..., slicedict[idx]].clone().detach())
                    patch_cond.append(cond[..., slicedict[idx]].clone().detach())
                
                patch_batch = self.process_patch_tensor(th.stack(patch_batch))
                patch_cond = self.process_patch_tensor(th.stack(patch_cond))
                patches.append((patch_batch, patch_cond))
        else:
            # Create adjacent partitions
            for start_idx in range(total_slices):
                # Take exactly patch_size consecutive slices, wrapping around if needed
                patch_indices = [(start_idx + i) % total_slices for i in range(patch_size)]
                
                patch_batch = []
                patch_cond = []
                for idx in patch_indices:
                    patch_batch.append(batch[..., slicedict[idx]].clone().detach())
                    patch_cond.append(cond[..., slicedict[idx]].clone().detach())
                
                patch_batch = self.process_patch_tensor(th.stack(patch_batch))
                patch_cond = self.process_patch_tensor(th.stack(patch_cond))
                patches.append((patch_batch, patch_cond))
        
        return patches

    def process_patch_tensor(self, tensor):
        """Process tensors to match required dimensions."""
        return tensor.squeeze(1).squeeze(-1)
    
    def forward_backward_accumulate(self, batch, cond, batchsize):
        """Compute loss for a single patch without optimizer step."""
        micro = th.cat((batch, cond), dim=1).to(dist_util.dev())
        cond = {}
        # Get timesteps
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
        
        # Compute losses
        compute_losses = functools.partial(
            self.diffusion.training_losses_segmentation,
            self.ddp_model,
            self.classifier,
            micro,
            t,
        )
        
        losses = compute_losses()
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses[0]["loss"].detach() * batchsize)
        
        # Scale loss by batch_size to average gradients
        loss = (losses[0]["loss"] * weights).mean() / batchsize # Do not use self.batch_size, that would affect the default dataloader collate func's behavior--it will try to stack samples with different slicedicts thus errors.
        
        # Update moving average loss
        self.loss_history.append(loss.item() * self.batch_size)  # Un-scale for logging
        if len(self.loss_history) > self.ma_window:
            self.loss_history.pop(0)
        self.moving_avg_loss = sum(self.loss_history) / len(self.loss_history)
        
        # Backward pass
        self.mp_trainer.backward(loss)
        
        return losses[1]  # Return sample for visualization


    def _update_ema(self):

        for rate, params in zip(self.ema_rate, self.ema_params):

            update_ema(params, self.mp_trainer.master_params, rate=rate)



    def _anneal_lr(self):

        if not self.lr_anneal_steps:

            return

        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps

        lr = self.lr * (1 - frac_done)

        for param_group in self.opt.param_groups:

            param_group["lr"] = lr



    def log_step(self):

        logger.logkv("step", self.step + self.resume_step)

        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)



    def save(self):

        def save_checkpoint(rate, params):

            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            if dist.get_rank() == 0:

                logger.log(f"saving model {rate}...")

                if not rate:

                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"

                else:

                    filename = (

                        f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"

                    )

                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:

                    th.save(state_dict, f)



        save_checkpoint(0, self.mp_trainer.master_params)

        for rate, params in zip(self.ema_rate, self.ema_params):

            save_checkpoint(rate, params)



        if dist.get_rank() == 0:

            with bf.BlobFile(

                bf.join(

                    get_blob_logdir(),

                    f"optsavedmodel{(self.step+self.resume_step):06d}.pt",

                ),

                "wb",

            ) as f:

                th.save(self.opt.state_dict(), f)



        dist.barrier()


class AMPTrainLoop(TrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler()
        self.autocast_enabled = self.use_fp16
        
    def forward_backward_accumulate(self, batch, cond, batchsize):
        """Compute loss for a single patch with mixed precision."""
        micro = th.cat((batch, cond), dim=1).to(dist_util.dev())
        cond = {}
        # Get timesteps
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
        
        # Compute losses with autocast for mixed precision
        with amp.autocast(enabled=self.autocast_enabled):
            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
            )
            
            losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses[0]["loss"].detach() * batchsize)
            
            # Scale loss by batch_size to average gradients
            loss = (losses[0]["loss"] * weights).mean() / batchsize
        
        # Update moving average loss
        self.loss_history.append(loss.item() * self.batch_size)  # Un-scale for logging
        if len(self.loss_history) > self.ma_window:
            self.loss_history.pop(0)
        self.moving_avg_loss = sum(self.loss_history) / len(self.loss_history)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        return losses[1]  # Return sample for visualization

    def optimize(self, opt):
        """Optimizes parameters with gradient scaling for mixed precision."""
        took_step = False
        # Unscales gradients and skips updates if NaNs/Infs are found
        self.scaler.unscale_(opt)
        
        if self.use_fp16:
            # Check for gradient overflow
            grad_norm, param_norm = self._compute_norms()
            if not th.isfinite(grad_norm):
                logger.log(f"Found NaN or Inf in gradients, skipping update")
                self.mp_trainer.zero_grad()
                return False
        
        # Step optimizer with scaled gradients
        self.scaler.step(opt)
        
        # Update scale for next iteration
        self.scaler.update()
        
        # Update EMA parameters
        self._update_ema()
        took_step = True
        
        return took_step


def parse_resume_step_from_filename(filename):

    """

    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the

    checkpoint's number of steps.

    """

    split = filename.split("model")

    if len(split) < 2:

        return 0

    split1 = split[-1].split(".")[0]

    try:

        return int(split1)

    except ValueError:

        return 0





def get_blob_logdir():

    # You can change this to be a separate path to save checkpoints to

    # a blobstore or some external drive.

    return logger.get_dir()





def find_resume_checkpoint():

    # On your infrastructure, you may want to override this to automatically

    # discover the latest checkpoint on your blob storage, etc.

    return None





def find_ema_checkpoint(main_checkpoint, step, rate):

    if main_checkpoint is None:

        return None

    filename = f"ema_{rate}_{(step):06d}.pt"

    path = bf.join(bf.dirname(main_checkpoint), filename)

    if bf.exists(path):

        return path

    return None





def log_loss_dict(diffusion, ts, losses):

    for key, values in losses.items():

        logger.logkv_mean(key, values.mean().item())

        # Log the quantiles (four quartiles, in particular).

        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):

            quartile = int(4 * sub_t / diffusion.num_timesteps)

            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class ModifiedTrainLoop(TrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # If resuming from checkpoint, modify learning rate
        if self.resume_step:
            self._modify_learning_rate()
    
    def _modify_learning_rate(self):
        if dist.get_rank() == 0:
            logger.log("Modifying learning rate...")
            for param_group in self.opt.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] *= 1  # Reduce by factor of 1
                logger.log(f"Learning rate changed from {old_lr} to {param_group['lr']}")
