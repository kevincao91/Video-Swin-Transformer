#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
# source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/swin_base_patch244_window877_tld17k5c_22k_x3.py"
CHECKPOINT="mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/epoch_10.pth"
LABELFILE="data/tld17k/label_map_tld17k5c.txt"
# PATH_TO_LONG_VIDEO="data/tld17k/videos/210-C30-2021_11_14_10_21_49.avi"
PATH_TO_LONG_VIDEO="/changde/data/tld17k/videos_540_avi/220-C30-2021_11_23_09_00_05.avi"
PATH_TO_SAVED_VIDEO="mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/demo_video_540_220-C30-2021_11_23_09_00_05.avi"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/demo/long_video_demo.py $CONFIG \
      $CHECKPOINT \
      $PATH_TO_LONG_VIDEO \
      $LABELFILE \
      $PATH_TO_SAVED_VIDEO \
      --input-step 3 --threshold 0.2