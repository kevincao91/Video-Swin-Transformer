import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool
import subprocess as sp
import time


def resize_videos(vid_item,type_):
    """Generate resized video cache.

    Args:
        vid_item (list): Video item containing video full path,
            video relative path.

    Returns:
        bool: Whether generate video cache successfully.
    """
    full_path, vid_path = vid_item
    out_full_path = osp.join(args.out_dir, vid_path.replace('.mp4', '.avi'))
    dir_name = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, dir_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    tin = time.time()
    
    if type_ == 1 :
        # type1  cuda
        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'error',
               '-threads', '6',
               '-i', full_path,
               '-s', '960x540',
               '-r', '25',
               '-vcodec', 'h264_nvenc',
               '-an',
               out_full_path,
               '-y']
    else:
        # type2  cpu
        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'error',
               '-threads', '6',
               '-i', full_path,
               '-preset', 'fast',
               '-s', '960x540',
               '-r', '25',
               '-vcodec', 'libx264',
               '-an',
               out_full_path,
               '-y']

    # os.popen(cmd)
    s_ = sp.Popen(cmd)
    s_.wait()
    tou = time.time() - tin
    
    print(f'{vid_path} done', ' in %f s'%tou)
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the resized cache of original videos')
    parser.add_argument('--src_dir', type=str,
                        default='/changde/data/new_tld/0#/',
                        help='source video directory')
    parser.add_argument('--out_dir', type=str,
                        default='/changde/data/new_tld/videos_540_avi/',
                        help='output video directory')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=1,
        help='directory level of data')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--scale',
        type=int,
        default=1080//2,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--num-worker', type=int, default=1, help='number of workers')
    parser.add_argument(
        '--c_type',
        type=int,
        default=2, # 1 cuda  2 cpu
        help='resize c_type')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                              args.ext)
    done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level + args.ext)
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos transfer finished: ',
          len(done_fullpath_list))
    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    tin = time.time()
    # pool = Pool(args.num_worker)
    # pool.map(resize_videos, zip(fullpath_list, vid_list))
    for it in zip(fullpath_list, vid_list):
        resize_videos(it,args.c_type)
    tou = time.time() - tin
    print('Total time: %f'%tou)
