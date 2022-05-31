import argparse
import glob
import json
import os.path as osp
import random

from mmcv.runner import set_random_seed
from tools.data.anno_txt2json import lines2dictlist
from tools.data.parse_file_list import (parse_directory, parse_diving48_splits,
                                        parse_hmdb51_split,
                                        parse_jester_splits,
                                        parse_kinetics_splits,
                                        parse_mit_splits, parse_mmit_splits,
                                        parse_sthv1_splits, parse_sthv2_splits,
                                        parse_ucf101_splits)


def main():
    print('aaa')

    with open('/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_list_videos.txt', 'r') as f:
        lines = f.readlines()

    # print(lines)
    video_dic = {}
    for tld in range(190, 260, 10):
        video_dic[tld] = []

    for vi in lines:
        base = osp.basename(vi)
        field = base.split('-')
        # print(field)
        tld = int(field[0])
        video_dic[tld].append(base)
    # 分类表
    class_table = {190: 1,
                   200: 1,
                   210: 1,
                   220: 2,
                   230: 3,
                   240: 3,
                   250: 3
                   }
    se_dict = {
        190: 1,
        200: 12,
        210: 20,
        220: 50,
        230: 10,
        240: 6,
        250: 1
    }

    for tld in range(190, 260, 10):
        len_ = len(video_dic[tld])
        print('==> ', tld, len_)

    test_ = []
    for tld in range(190, 260, 10):
        len_ = len(video_dic[tld])
        print('se> ', tld, se_dict[tld])

        cnt = 0
        for vi in lines:
            base = osp.basename(vi)
            field = base.split('-')
            # print(field)
            tld_ = int(field[0])
            if tld_ == tld:
                test_.append(vi)
                cnt += 1
            if cnt >= se_dict[tld]:
                break

    # random.shuffle(train_)
    # random.shuffle(test_)

    with open('/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_test_list_videos.txt', 'w') as f:
        for it in test_:
            f.write(it)


if __name__ == '__main__':
    main()
