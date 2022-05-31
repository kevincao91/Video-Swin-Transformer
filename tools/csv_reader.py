# coding = utf-8
"""
@ Author: kevin
@ Date: 2021-12-15 16:21
@ Email:caok15@sany.com.cn
# Copyright (c) SANY, Group. and its affiliates.
"""

# import some common libraries
import csv
# from tqdm import tqdm
import os.path as osp
import shutil


# coding here ======
def main():

    src_dir = './1207'
    out_dir = './1207select'


    print('hello world!')
    file_ = r'D:\WorkSpace\Video-Swin-Transformer\mytrain\tld17k5c1207_val_infer_more_detail.csv'
    with open(file_, 'r') as f:
        csv_reader = csv.reader(f)
        _ = next(csv_reader)
        for row in csv_reader:
            video, label, infer, _, _, _, _, _, _ = row
            # video_path = osp.join(src_dir, video)
            video_path = src_dir + '/' + video

            # out_path = osp.join(out_dir, infer, video)
            out_path = out_dir + '/' + infer + '/'+ video
            shutil.copy(video_path, out_path)
            print(video_path,out_path,'done!')


if __name__ == '__main__':
    main()

