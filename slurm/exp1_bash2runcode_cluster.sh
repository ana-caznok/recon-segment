#! /bin/bash
# ====================================
#SBATCH --job-name=segrec-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=9GB
#SBATCH --time=0-20:00:00
# ====================================
# Activate Conda and then the environment.

source ~/software/init-conda
conda activate agrvai
export CODE_PATH="/home/ana.caznoksilveira/recon-segment"
export DATA_PATH="/home/ana.caznoksilveira/icasp/icasp/data/Link_2"
export WANDB_API_KEY=91726ad327981a84eb14736bd7e3800221958491

wandb login
cd /home/ana.caznoksilveira/recon-segment

python train_seg_rec.py --config configs/fft2fft_transf2metrics_cluster.yaml
python train_seg_rec.py --config configs/fft2ifft2hsi_cluster.yaml
python train_seg_rec.py --config configs/fft2hsi_cluster.yaml

python train_seg_rec.py --config configs/fft2fft_transf2metrics_none_cluster.yaml
python train_seg_rec.py --config configs/fft2ifft2hsi_none_cluster.yaml