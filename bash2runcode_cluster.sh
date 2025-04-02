export CODE_PATH="/home/ana.caznoksilveira/recon-segment"
export DATA_PATH="/home/ana.caznoksilveira/icasp/icasp/data/Link_2"

#python train_seg_rec.py --config configs/test.yaml
#python train_seg_rec.py --config configs/test_double.yaml

python train_seg_rec.py --config configs/fft2ifft2hsi.yaml
python train_seg_rec.py --config configs/fft2hsi.yaml
python train_seg_rec.py --config configs/fft2fft.yaml
