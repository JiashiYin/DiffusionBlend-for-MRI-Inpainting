#!/bin/bash



# Set environment variables

MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

TRAIN_FLAGS="--lr 1e-4 --batch_size 1"



# Navigate to the correct directory

cd /scratch/bcsl/jyin15/DiffBlend/guided_diffusion || exit



# Run the training script

python3 ../scripts/inpainting_train.py --data_dir /scratch/bcsl/jyin15/DiffBlend/preprocessed_data --log_dir /scratch/bcsl/jyin15/DiffBlend/ckpt $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS


# --resume_checkpoint /scratch/bcsl/jyin15/DiffBlend/ckpt/savedmodel432000.pt


