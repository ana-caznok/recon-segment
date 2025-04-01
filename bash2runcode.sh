export CODE_PATH="/media/ana-caznok/SSD-08/recon-segment"
export DATA_PATH="/media/ana-caznok/SSD-08/icasp_4090/icasp/data/Link_2"

#python train_seg_rec.py --config configs/test.yaml
#python train_seg_rec.py --config configs/test_double.yaml

python train_seg_rec.py --config configs/test_ifft.yaml
