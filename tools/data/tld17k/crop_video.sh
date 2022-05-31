#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
# source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch


python tools/data/tld17k/crop_video.py \
--csv_path /caok15/Pytorch_Retinaface/video_coords_infer_1207.csv
