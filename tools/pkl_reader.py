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
import numpy as np
from tqdm import tqdm


# coding here ======
def main():
    print('hello world!')
    # file_ = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/test_result.pkl'
    file_ = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/line2_test_result.pkl'

    with open(file_, 'rb') as pkl_f:
        data = pickle.load(pkl_f)

    # val_ = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_list_videos.txt'
    val_ = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_line2_list_videos.txt'
    with open(val_, 'r') as val_f:
        lines = val_f.readlines()

    # label_dict = {
    #     1: '190-200',
    #     2: '210',
    #     3: '220',
    #     4: '230',
    #     5: '240-250'
    # }
    label_dict = {
        -1: 'unlabeled',
        1: '190-210',
        2: '220',
        3: '230-250',
    }

    headers = ['video', 'label', 'infer']
    # csv_path = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_val_infer.csv'
    csv_path = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_val_line2_infer.csv'
    with open(csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, headers)
        csv_writer.writeheader()

        for it in tqdm(zip(lines, data)):
            line, pre = it
            vi, label = line.split()
            pre_label = np.argmax(pre)
            row = {
                'video': vi,
                'label': label_dict[int(label)],
                'infer': label_dict[pre_label]
            }
            csv_writer.writerow(row)


def more_info():
    print('hello world!')
    file_ = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/test_result.pkl'
    with open(file_, 'rb') as pkl_f:
        data = pickle.load(pkl_f)

    val_ = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_list_videos.txt'
    with open(val_, 'r') as val_f:
        lines = val_f.readlines()

    # label_dict = {
    #     1: '190-200',
    #     2: '210',
    #     3: '220',
    #     4: '230',
    #     5: '240-250'
    # }
    label_dict = {
        1: '190-210',
        2: '220',
        3: '230-250',
    }

    # headers = ['video', 'label', 'infer', '_background', '190-200', '210', '220', '230', '240-250']
    headers = ['video', 'label', 'infer', '_background', '190-210', '220', '230-250']
    csv_path = '/caok15/Video-Swin-Transformer/mytrain/tld17k3c_val_infer_more_detail.csv'
    with open(csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, headers)
        csv_writer.writeheader()

        for it in tqdm(zip(lines, data)):
            line, pre = it
            vi, label = line.split()
            pre_label = np.argmax(pre)

            # row = {
            #     'video': vi,
            #     'label': label_dict[int(label)],
            #     'infer': label_dict[pre_label],
            #     '_background': pre[0],
            #     '190-200': pre[1],
            #     '210': pre[2],
            #     '220': pre[3],
            #     '230': pre[4],
            #     '240-250': pre[5]
            # }
            row = {
                'video': vi,
                'label': label_dict[int(label)],
                'infer': label_dict[pre_label],
                '_background': pre[0],
                '190-210': pre[1],
                '220': pre[2],
                '230-250': pre[3]
            }
            csv_writer.writerow(row)


if __name__ == '__main__':
    main()
    # more_info()
