#!/usr/bin/env bash

cd /changde/code
source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch


python resize_video.py --c_type 1 --rev 1
