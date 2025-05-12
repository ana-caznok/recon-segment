export CODE_PATH="/media/ana-caznok/SSD-08/recon-segment"
export DATA_PATH="/media/ana-caznok/SSD-08/icasp_4090/icasp/data/Link_2/downsampled"

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

#python train_seg_rec.py --config configs/pseudohsi2hsi_D40.yaml


#python train_seg_rec.py --config configs/test_continue2.yaml

#python train_seg_rec.py --config configs/test_fft2hsi2.yaml

#python train_seg_rec.py --config configs/stable_fft2ifft2hsi_1000_none_cluster_xmin.yaml

#

#python train_seg_rec.py --config configs/test_segrec_rest2.yaml
#python train_seg_rec.py --config configs/test_overlap2.yaml
#python train_seg_rec.py --config configs/restormer_msi2hsi_mraessimsam.yaml
#python train_seg_rec.py --config configs/restormer_fft2hsi_mraessimsam.yaml
#python train_seg_rec.py --config configs/restormer_fft2ifft2hsi_mraessimsam.yaml
#python train_seg_rec.py --config configs/restormer-seg_msi2hsi.yaml
#python train_seg_rec_dualtask.py --config configs/restormer-seg_msi2mask.yaml
#python train_seg_rec.py --config configs/restormer_new-fft2ifft2hsi_mraessimsam.yaml
#python train_seg_rec.py --config configs/restormer_fft2ifft2hsi_byc_continuous.yaml
#python train_seg_rec.py --config configs/restormer_fft2ifft2hsi_noclip.yaml
#python train_seg_rec.py --config configs/restormer_fft2ifft2hsi_noclip_byc.yaml
#python train_seg_rec.py --config configs/restormer_fft2ifft2hsi_noclip_byimg.yaml
python train_seg_rec.py --config configs/unet_msi2hsi.yaml
