#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/swin_base_patch244_window877_ucf101_22k_x4.py"
CHECKPOINT="mytrain/work_dirs/ucf101_swin_base_22k_patch244_window877/epoch_30.pth"
GPUS=3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --eval top_k_accuracy
