# coding = utf-8
"""
@ Author: kevin
@ Date: 2021-12-15 16:21
@ Email:caok15@sany.com.cn
# Copyright (c) SANY, Group. and its affiliates.
"""

# import some common libraries
import os
import pickle
import csv
import numpy as np
from tqdm import tqdm
import csv
import shutil
from tqdm import tqdm


# coding here ======
def main():
    print('hello world!')
    src_dir = '/changde/data/tld17k/videos_crop_256_avi'
    dst_dir = '/changde/data/tld17k/初分类'

    label_dict = {
        1: '190-210',
        2: '220',
        3: '230-250',
    }

    for key in label_dict:
        dir_path = os.path.join(dst_dir, label_dict[key])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    csv_file = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_all_infer.csv'
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        for row in tqdm(csv_reader):
            video, _, infer = row
            # print(video, _, infer)
            src_ = os.path.join(src_dir, video)
            dst_ = os.path.join(dst_dir, infer, video)
            shutil.copy(src_, dst_)


if __name__ == '__main__':
    main()
