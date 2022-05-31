#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
# source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/work_dirs/ucf101_swin_base_22k_patch244_window877/swin_base_patch244_window877_ucf101_22k_x4.py"
CHECKPOINT="mytrain/work_dirs/ucf101_swin_base_22k_patch244_window877/best_top1_acc_epoch_27.pth"
RESULT_FILE="mytrain/work_dirs/ucf101_swin_base_22k_patch244_window877/test_result.pkl"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py $CONFIG \
        $CHECKPOINT \
        --out $RESULT_FILE \
        --eval top_k_accuracy mean_class_accuracy

