#!/bin/bash

# Set environment variables for model configuration
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

# Run the sampling script
python /scratch/bcsl/jyin15/DiffBlend/scripts/inpainting_sample_uses_interface.py \
  --data_dir /scratch/bcsl/jyin15/DiffBlend/toy_data \
  --log_dir /scratch/bcsl/jyin15/DiffBlend/sampling_results_playground \
  --model_path /scratch/bcsl/jyin15/DiffBlend/ckpt/emasavedmodel_0.9999_482000.pt \
  --adapted_samples /scratch/bcsl/jyin15/DiffBlend/adapted_samples \
  $MODEL_FLAGS $DIFFUSION_FLAGS