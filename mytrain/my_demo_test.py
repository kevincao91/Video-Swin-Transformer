import torch
from mmaction.apis import init_recognizer, inference_recognizer
import os.path as osp
import glob
import csv
from tqdm import tqdm


def main():
    config_file = '/caok15/Video-Swin-Transformer/mytrain/swin_base_patch244_window877_tld17k5c_22k_x2.py'
    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)
    checkpoint_file = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/epoch_10.pth'

    model = init_recognizer(config=config_file, checkpoint=checkpoint_file, device=device)

    # test loop
    print('test loop')

    test_file = '/caok15/Video-Swin-Transformer/data/tld17k/tld17k5c_val_list_videos.txt'
    with open(test_file, 'r') as tf:
        lines = tf.readlines()

    headers = ['video', 'tld', 'infer']
    csv_path = '/caok15/Video-Swin-Transformer/mytrain/infer.csv'
    with open(csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, headers)
        csv_writer.writeheader()

        video_list = ['/changde/data/tld17k/videos_crop_256_avi/' + line.split()[0] for line in lines]
        print('total num: ', len(video_list))

        for vi in video_list:
            # inference the demo video
            results = inference_recognizer(model, video_path=vi,
                                           label_path='/caok15/Video-Swin-Transformer/data/tld17k/label_map_tld17k5c.txt')
            print(vi, results)
            result = results[0][0]

            tld = osp.basename(vi).split("-")[0]

            row = {
                'video': vi,
                'tld': tld,
                'infer': result,
            }
            csv_writer.writerow(row)

    print('ok!')


def single():
    config_file = '/caok15/Video-Swin-Transformer/mytrain/swin_base_patch244_window877_tld17k5c_22k_x3.py'
    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)
    checkpoint_file = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/epoch_10.pth'

    model = init_recognizer(config=config_file, checkpoint=checkpoint_file, device=device)

    # test loop
    print('test')
    video_list = ['/caok15/Video-Swin-Transformer/jp/jp_jbzj_0106_170.mp4',
                  '/caok15/Video-Swin-Transformer/jp/jp_jbzj_0106_190.mp4',
                  '/caok15/Video-Swin-Transformer/jp/jp_xld_0106_170.mp4',
                  '/caok15/Video-Swin-Transformer/jp/jp_xld_0106_180.mp4',
                  '/caok15/Video-Swin-Transformer/jp/jp_xlk_0106_170.mp4',
                  '/caok15/Video-Swin-Transformer/jp/jp_xlk_0106_190.mp4',
                  ]
    print('total num: ', len(video_list))

    for vi in video_list:
        # inference the demo video
        results = inference_recognizer(model, video_path=vi,
                                       label_path='/caok15/Video-Swin-Transformer/data/tld17k/label_map_tld17k5c.txt')
        print(vi, results)

    print('ok!')


if __name__ == '__main__':
    main()
    # single()
