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
    video1_list = glob.glob(osp.join('/caok15/Video-Swin-Transformer/data/tld3/videos', '1', '*'))
    t1_num = len(video1_list)
    video2_list = glob.glob(osp.join('/caok15/Video-Swin-Transformer/data/tld3/videos', '2', '*'))
    t2_num = len(video2_list)
    video3_list = glob.glob(osp.join('/caok15/Video-Swin-Transformer/data/tld3/videos', '3', '*'))
    t3_num = len(video3_list)
    print(t1_num, t2_num, t3_num)

    train_ = []
    test_ = []

    for idx, file_ in enumerate(video1_list):
        if idx < 0.8 * t1_num:
            train_.append((file_, '1'))
        else:
            test_.append(file_)

    for idx, file_ in enumerate(video2_list):
        if idx < 0.8 * t2_num:
            train_.append((file_, '2'))
        else:
            test_.append(file_)

    for idx, file_ in enumerate(video3_list):
        if idx < 0.8 * t3_num:
            train_.append((file_, '3'))
        else:
            test_.append(file_)

    # random.shuffle(train_)
    # random.shuffle(test_)

    with open('/caok15/Video-Swin-Transformer/data/tld3/annotations/testlist01.txt', 'w') as f:
        for it in test_:
            f.write(it[48:]+'\n')
    with open('/caok15/Video-Swin-Transformer/data/tld3/annotations/trainlist01.txt', 'w') as f:
        for it in train_:
            f.write(' '.join(it)[48:]+'\n')


def test():
    import decord as de
    ctx = de.cpu(0)

    video1_list = glob.glob(osp.join('/caok15/Video-Swin-Transformer/data/tld3/videos_540_avi', '3', '*.avi'))
    t1_num = len(video1_list)


    for idx, file_ in enumerate(video1_list):
        try:
            print(idx)
            vr = de.VideoReader(file_)
            # print(len(vr))
            # print(vr[0].shape)
        except:
            print(idx, file_)








if __name__ == '__main__':
    test()
