export CODE_PATH="/media/ana-caznok/SSD-08/recon-segment"
export DATA_PATH="/media/ana-caznok/SSD-08/NTIRE"

python train_seg_rec.py --config configs/ntire-restormer_rgb2hsi.yaml
python train_seg_rec.py --config configs/ntire-restormer_rgb2fft2ifft2hsi.yaml

