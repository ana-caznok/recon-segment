#! /bin/bash
# ====================================
#SBATCH --job-name=rainbow-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=5GB
#SBATCH --time=0-02:35:00
# ====================================
# Activate Conda and then the environment.

source ~/software/init-conda
conda activate agrvai
export CODE_PATH="/home/ana.caznoksilveira/recon-segment"
export DATA_PATH="/home/ana.caznoksilveira/icasp/icasp/data/Link_2"
export WANDB_API_KEY=91726ad327981a84eb14736bd7e3800221958491

wandb login
cd /home/ana.caznoksilveira/recon-segment

python train_seg_rec.py --config configs/fft2fft_cluster.yaml
