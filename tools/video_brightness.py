# coding = utf-8
"""
@ Author: kevin
@ Date: 2021-12-15 16:21
@ Email:caok15@sany.com.cn
# Copyright (c) SANY, Group. and its affiliates.
"""

# import some common libraries
import pickle
import csv
import cv2
import numpy as np
from PIL import ImageStat
from PIL import Image
from tqdm import tqdm
import os.path as osp
from moviepy.editor import *


def bv(vi):
    clip = VideoFileClip(vi)
    clipColorx = clip.fx(vfx.colorx, 0.8)
    clipColorx.write_videofile(vi)


def compute_brightness(vi):
    tmp = './tmp.jpg'
    cap = cv2.VideoCapture(osp.join('/changde/data/tld17k/videos_crop_256_avi/', vi))
    cnt = 0
    while cnt < 10:
        flag, frame = cap.read()
        cnt += 1
    cv2.imwrite(tmp, frame)
    im = Image.open(tmp).convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]


# coding here ======
def main():
    print('hello world!')

    val_ = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_all_list_videos.txt'
    # val_ = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_line6_list_videos.txt'
    with open(val_, 'r') as val_f:
        lines = val_f.readlines()

    headers = ['video', 'brightness']
    # csv_path = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_val_infer.csv'
    csv_path = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_all_brightness.csv'
    with open(csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, headers)
        csv_writer.writeheader()

        for it in tqdm(lines):
            vi, label = it.split()
            brightness = compute_brightness(vi)
            row = {
                'video': vi,
                'brightness': brightness
            }
            csv_writer.writerow(row)


if __name__ == '__main__':
    main()
