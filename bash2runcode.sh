export CODE_PATH="/media/ana-caznok/SSD-08/recon-segment"
export DATA_PATH="/media/ana-caznok/SSD-08/icasp_4090/icasp/data/Link_2"

#python train_seg_rec.py --config configs/test.yaml
#python train_seg_rec.py --config configs/test_double.yaml

#python train_seg_rec.py --config configs/fft2ifft2hsi.yaml
#python train_seg_rec.py --config configs/fft2hsi.yaml
#python train_seg_rec.py --config configs/fft2fft.yaml

#python train_seg_rec.py --config configs/fft2fft_transf2metrics.yaml
#python train_seg_rec.py --config configs/fft2ifft2hsi_cluster.yaml
#python train_seg_rec.py --config configs/fft2hsi.yaml
#python train_seg_rec.py --config configs/test_stable.yaml

#python train_seg_rec.py --config configs/stable_fft2ifft2hsi_none_cluster_xmin.yaml

python train_seg_rec.py --config configs/pseudohsi2hsi_D40.yaml

