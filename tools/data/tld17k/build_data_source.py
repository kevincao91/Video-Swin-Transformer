import argparse
import glob
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import time

import random


def get_dh_info():
    print('获取单号信息')

    df = pd.read_excel('data_20220218.xlsx')
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    data = df.sort_values(by='开始时间', axis=0, ascending=True, inplace=False)
    data_num = len(df)
    print('total data num: ', data_num)
    assert data_num == 18061
    # drop_duplicates
    data = data.drop_duplicates(subset=['开始时间'], keep=False)

    video_list = glob.glob(osp.join('/changde/data/tld17k/videos_crop_256_avi/', '*.avi'))
    video_num = len(video_list)
    print('total video num: ', video_num)
    assert video_num == 15601

    video_list = sorted(video_list, key=lambda x: x.split('-')[-1])
    # print(data.iat[0, 1])
    # print(data.iat[-400, 1])
    # print(video_list[0])
    # print(video_list[-400:])
    # exit()

    matched_num = 0
    idx_data = 0
    idx_video = 0
    data['video_name'] = '-1'
    while idx_data < data_num and idx_video < video_num:
        row = data.iloc[idx_data, :]
        data_time = row['开始时间']
        # print(data_time)
        data_time_str = data_time.strftime("%Y_%m_%d_%H_%M_%S")
        print('data_time : ', data_time_str)
        video_path = video_list[idx_video]
        video_name = os.path.basename(video_path)
        video_txt, file_end = os.path.splitext(video_name)
        video_tld, video_qddj, video_time = video_txt.split('-')
        print('video_time: ', video_time)
        if data_time_str == video_time:
            matched_num += 1
            print('** index:', idx_data, idx_video, ' | matched num:', matched_num)
            data.iat[idx_data, -1] = video_name
            idx_data += 1
            idx_video += 1
        else:
            print('index:', idx_data, idx_video, ' | matched num:', matched_num)

            data_time_str = time.strptime(data_time_str, "%Y_%m_%d_%H_%M_%S")
            video_time = time.strptime(video_time, "%Y_%m_%d_%H_%M_%S")
            # print('data_time : ', data_time_str)
            # print('video_time: ', video_time)
            data_time_str = time.mktime(data_time_str)
            video_time = time.mktime(video_time)
            # print('data_time : ', data_time_str)
            # print('video_time: ', video_time)
            data_video_sec = data_time_str - video_time
            print('data_video_sec: ', data_video_sec)
            if data_video_sec > 0:
                idx_video += 1
            elif data_video_sec < 0:
                idx_data += 1
        # if idx_data > 17125:
        #     exit()

    print('to_excel')
    data.to_excel('data_matched.xlsx', index=False)


def build_by_dh():
    print("分单号进行数据集生成")
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
        190: 7,
        200: 406,
        210: 2149,
        220: 7000,  # 12641
        230: 343,
        240: 5,
        250: 1
    }

    df = pd.read_excel('data_matched.xlsx')
    df = df[df['video_name'] != '-1']

    groups = df.groupby(['坍落度值'])

    id_name = groups.size()
    print(id_name)

    danhao_dic = {}

    for tld in range(190, 260, 10):
        df_groups = groups.get_group(tld)
        sub_groups = df_groups.groupby('出货单号')
        danhao_list = sub_groups.size().index.to_list()
        random.shuffle(danhao_list)
        danhao_dic[tld] = danhao_list

    # selecting
    train_ = []
    test_ = []
    for tld in range(190, 260, 10):
        len_se = se_dict[tld]
        print('se> ', tld, len_se)
        danhao_ = danhao_dic[tld]
        df_groups = groups.get_group(tld)
        sub_groups = df_groups.groupby('出货单号')
        add_num = 0
        for danhao in danhao_:
            data = sub_groups.get_group(danhao)
            if add_num < 0.8 * len_se:
                for row in zip(data['坍落度值'], data['出货单号'], data['video_name']):
                    train_.append((row[-1], str(class_table[tld])))
                    add_num += 1
            elif add_num < len_se:
                for row in zip(data['坍落度值'], data['出货单号'], data['video_name']):
                    test_.append((row[-1], str(class_table[tld])))
                    add_num += 1
            else:
                break
    # exit()

    # random.shuffle(train_)
    # random.shuffle(test_)

    with open('/caok15/Video-Swin-Transformer/data/tld17k/annotations/trainlist3c.txt', 'w') as f:
        for it in train_:
            f.write(' '.join(it) + '\n')
    with open('/caok15/Video-Swin-Transformer/data/tld17k/annotations/testlist3c.txt', 'w') as f:
        for it in test_:
            f.write(' '.join(it) + '\n')


def main():
    print('不分单号进行数据集生成')
    video_list = glob.glob(osp.join('/changde/data/tld17k/videos_crop_256_avi/', '*.avi'))

    print('total num: ', len(video_list))
    assert len(video_list) == 15601

    video_dic = {}

    for tld in range(190, 260, 10):
        video_dic[tld] = []

    for vi in video_list:
        base = osp.basename(vi)
        field = base.split('-')
        # print(field)
        tld = int(field[0])
        video_dic[tld].append(base)

    # TLD分类时间排序
    for tld in range(190, 260, 10):
        list_tmp = video_dic[tld]
        list_tmp = sorted(list_tmp, key=lambda x: x.split('-')[-1])
        video_dic[tld] = list_tmp

    # 时间排序
    for tld in range(190, 260, 10):
        list_tmp = video_dic[tld]
        list_tmp = sorted(list_tmp, key=lambda x: x.split('-')[-1])
        video_dic[tld] = list_tmp

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
        190: 7,
        200: 409,
        210: 2191,
        220: 7000,  # 12641
        230: 347,
        240: 5,
        250: 1
    }

    train_ = []
    test_ = []
    for tld in range(190, 260, 10):
        len_total = len(video_dic[tld])
        print('==> ', tld, len_total)
        len_se = se_dict[tld]
        print('se> ', tld, len_se)

        se_point = np.linspace(0, len_total - 1, len_se)
        se_point = list(map(int, se_point))
        file_ = video_dic[tld]

        for idx in se_point:
            if idx < 0.8 * len_total:
                train_.append((file_[idx], str(class_table[tld])))
            elif idx < len_total:
                test_.append((file_[idx], str(class_table[tld])))
            else:
                break
    # exit()

    # random.shuffle(train_)
    # random.shuffle(test_)

    with open('/caok15/Video-Swin-Transformer/data/tld17k/annotations/trainlist3c.txt', 'w') as f:
        for it in train_:
            f.write(' '.join(it) + '\n')
    with open('/caok15/Video-Swin-Transformer/data/tld17k/annotations/testlist3c.txt', 'w') as f:
        for it in test_:
            f.write(' '.join(it) + '\n')


def all_build_test():
    print('所有数据生成测试集')

    video_list = glob.glob(osp.join('/changde/data/tld17k/videos_crop_256_avi/', '*.avi'))
    print('total num: ', len(video_list))
    assert len(video_list) == 15601

    video_dic = {}

    for tld in range(190, 260, 10):
        video_dic[tld] = []

    for vi in video_list:
        base = osp.basename(vi)
        field = base.split('-')
        # print(field)
        tld = int(field[0])
        video_dic[tld].append(base)

    # TLD分类时间排序
    for tld in range(190, 260, 10):
        list_tmp = video_dic[tld]
        list_tmp = sorted(list_tmp, key=lambda x: x.split('-')[-1])
        video_dic[tld] = list_tmp

    # 分类表
    class_table = {190: 1,
                   200: 1,
                   210: 1,
                   220: 2,
                   230: 3,
                   240: 3,
                   250: 3
                   }

    all_ = []
    for tld in range(190, 260, 10):
        len_total = len(video_dic[tld])
        print('==> ', tld, len_total)
        file_ = video_dic[tld]

        for it in file_:
            all_.append((it, str(class_table[tld])))
    # exit()
    all_sort = sorted(all_, key=lambda x: x[0].split('-')[-1])
    with open('/caok15/Video-Swin-Transformer/data/tld17k/annotations/tld17k3c_all_list_videos.txt', 'w') as f:
        for it in all_sort:
            f.write(' '.join(it) + '\n')


def dir_build_test():
    print('所有数据生成测试集')

    video_list = glob.glob(osp.join('/changde/data/tld17k/validation/line2_crop_256_avi/', '*.avi'))
    print('total num: ', len(video_list))
    # assert len(video_list) == 15601

    video_list = [(osp.basename(vi), '-1') for vi in video_list]

    all_sort = sorted(video_list, key=lambda x: x[0].split('-')[-1])
    with open('/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_line2_list_videos.txt', 'w') as f:
        for it in all_sort:
            f.write(' '.join(it) + '\n')


if __name__ == '__main__':
    # get_dh_info()
    # build_by_dh()
    # main()
    # all_build_test()
    dir_build_test()
