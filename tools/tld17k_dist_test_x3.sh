#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/swin_base_patch244_window877_tld17k5c_22k_x3.py"
CHECKPOINT="mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/epoch_10.pth"
RESULT_FILE="mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/test_result.pkl"
GPUS=3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --eval top_k_accuracy --out $RESULT_FILE
