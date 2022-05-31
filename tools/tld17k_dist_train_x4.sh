#!/usr/bin/env bash

cd /caok15/Video-Swin-Transformer
source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch

CONFIG="mytrain/swin_base_patch244_window877_tld17k_22k_x4.py"
GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch \
    --test-last

