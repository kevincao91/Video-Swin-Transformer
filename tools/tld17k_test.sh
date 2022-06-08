#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
# source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/swin_base_patch244_window877_tld17k3c_22k.py"
CHECKPOINT="mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/epoch_10.pth"
RESULT_FILE="mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/result.pkl"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py $CONFIG \
        $CHECKPOINT \
        --out $RESULT_FILE \
        --eval top_k_accuracy mean_class_accuracy

