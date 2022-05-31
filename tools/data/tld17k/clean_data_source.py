import argparse
import glob
import json
import os
import os.path as osp
import numpy as np
import shutil


def main():
    print('aaa')
    video_list = glob.glob(osp.join('/changde/data/tld17k/videos_crop_256_avi/', '*.avi'))

    print('total num: ', len(video_list))
    assert len(video_list) == 15601

    for vi in video_list:
        bn = osp.basename(vi)
        splits = bn.split("-")

        # tld rep
        # if len(splits) == 3:
        #     tld, qd, date = splits
        # else:
        #     print(vi)
        #     to_vi = vi.replace(bn, bn[4:])
        #     print('to => ', to_vi)
        #     shutil.move(vi, to_vi)

        if len(splits) == 3:
            tld, qd, date = splits
        else:
            print(vi)
            break
        if '细石' in qd:
            print(vi)
            os.remove(vi)
            continue
        if not qd.startswith('C'):
            print(vi)
            os.remove(vi)


if __name__ == '__main__':
    main()
