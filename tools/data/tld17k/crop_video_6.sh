#!/usr/bin/env bash

cd /changde/code
source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch


python crop_video.py \
--csv_path /caok15/Pytorch_Retinaface/video_coords_infer_20220316_6.csv \
--out_dir /changde/data/tld17k/validation_crop_256_avi/