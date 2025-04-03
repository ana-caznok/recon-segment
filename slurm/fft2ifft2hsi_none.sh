#! /bin/bash
# ====================================
#SBATCH --job-name=segrecnone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=9GB
#SBATCH --time=0-28:00:00
#SBATCH --output=none_fft2ifft2hsi.out
# ====================================
# Activate Conda and then the environment.

source ~/software/init-conda
conda activate agrvai
export CODE_PATH="/home/ana.caznoksilveira/recon-segment"
export DATA_PATH="/home/ana.caznoksilveira/icasp/icasp/data/Link_2"
export WANDB_API_KEY=91726ad327981a84eb14736bd7e3800221958491

wandb login
cd /home/ana.caznoksilveira/recon-segment


python train_seg_rec.py --config configs/fft2ifft2hsi_none_cluster.yaml