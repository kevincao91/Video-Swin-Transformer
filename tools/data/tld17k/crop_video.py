import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool
import subprocess as sp
import time
import csv
from tqdm import tqdm


def resize_videos(vid_item, type_=1):
    """Generate resized video cache.

    Args:
        vid_item (dict): Video item containing video full path,
            video relative path.

    Returns:
        bool: Whether generate video cache successfully.
    """
    video = vid_item['video']
    w = vid_item['w']
    h = vid_item['h']
    x = vid_item['x']
    y = vid_item['y']

    if w == '-1':
        print(f'{video} miss hopper ==============')
        return True

    full_path, vid_path = video, osp.basename(video)
    out_full_path = osp.join(args.out_dir, vid_path.replace('.mp4', '.avi'))
    dir_name = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, dir_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    tin = time.time()

    if type_ == 1:
        # type1  cuda
        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'error',
               '-threads', '6',
               '-i', full_path,
               '-vf', 'crop=%s:%s:%s:%s' % (w, h, x, y),
               '-s', '256x256',
               '-r', '25',
               '-vcodec', 'h264_nvenc',
               '-an',
               out_full_path,
               '-n']
    else:
        # type2  cpu
        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'error',
               '-threads', '6',
               '-i', full_path,
               '-preset', 'fast',
               '-vf', 'crop=%s:%s:%s:%s' % (w, h, x, y),
               '-s', '256x256',
               '-r', '25',
               '-vcodec', 'libx264',
               '-an',
               out_full_path,
               '-n']

    # os.popen(cmd)
    s_ = sp.Popen(cmd)
    s_.wait()
    tou = time.time() - tin

    print(f'{vid_path} done', ' in %f s' % tou)
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the resized cache of original videos')
    parser.add_argument('--out_dir', type=str,
                        default='/changde/data/tld17k/1207_crop_256_avi/',
                        help='output video directory')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='/caok15/Pytorch_Retinaface/video_coords_infer_1207.csv')
    parser.add_argument(
        '--c_type',
        type=int,
        default=1,  # 1 cuda  2 cpu
        help='resize c_type')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    tin = time.time()

    headers = ['video', 'w', 'h', 'x', 'y']
    with open(args.csv_path, 'r') as f:
        csv_reader = csv.DictReader(f, headers)
        _ = next(csv_reader)

        for row in tqdm(csv_reader):
            resize_videos(row, args.c_type)

    tou = time.time() - tin
    print('Total time: %f' % tou)
