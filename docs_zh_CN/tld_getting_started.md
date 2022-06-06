# 塌落度视觉模型训练基础教程

本文档提供基于 MMAction2 的 Swin-Transformer 相关用法的基本教程。对于安装说明，请参阅 [安装指南](install.md)。

<!-- TOC -->

- [数据集](#数据集)
  - [测试数据集](#测试数据集)
- [如何训练模型](#如何训练模型)
  - [数据集标注文件准备](#数据集标注文件准备)
  - [训练配置](#训练配置)
  - [使用单个 GPU 进行训练](#使用单个-GPU-进行训练)
  - [使用多个 GPU 进行训练](#使用多个-GPU-进行训练)
- [详细教程](#详细教程)

<!-- TOC -->

## 数据集

所有数据在/changde/data/tld17k目录下。
```
changde
├── data
│   ├── tld17k
│   │   ├── videos                            原始的摄像头视频
│   │   ├── videos_540_avi                    经过缩放到540x540的并转成avi的视频
│   │   ├── videos_crop_256_avi               经过裁剪到256x256的并转成avi的视频
│   │   ├── videos_comqd_crop_256_avi         经过裁剪到256x256的并转成avi的并保留常用强度的视频
│   │   ├── tld17k3c_train_list_videos.txt    训练集标注数据
│   │   ├── tld17k3c_val_list_videos.txt      验证集标注数据
│   │   ├── tld17k3c_test_list_videos.txt     测试集标注数据
```

建议将changde下参与训练的视频目录链接到 `$Video-Swin-Transformer/data` 下。
如果用户的文件夹结构与默认结构不同，则需要在配置文件中进行对应路径的修改。

```
Video-Swin-Transformer
├── data
│   ├── tld17k
│   │   ├── videos
│   │   ├── tld17k3c_train_list_videos.txt
│   │   ├── tld17k3c_val_list_videos.txt
│   │   ├── tld17k3c_test_list_videos.txt
│   ├── ucf101
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── ucf101_train_list.txt
│   │   ├── ucf101_val_list.txt
│   ├── ...
```


### 测试数据集

用户可使用以下命令进行数据集测试

```shell
# 单 GPU 测试
./tools/tld17k_test.sh

# 多 GPU 测试
./tools/tld17k_dist_test_x3.sh
./tools/tld17k_dist_test_x4.sh
```

具体细节：

自定义配置在 `./mytrain/swin_base_patch244_window877_tld17k3c_22k.py` 目录下。
自定义训练的结果文件夹在 `./mytrain/tld17k3c_swin_base_22k_patch244_window877` 目录下。

实例1. 在 tld17k 数据集下测试 已经训练好的模型`，并验证 `top-k accuracy` 和 `mean class accuracy` 指标

调用文件 ./tools/tld17k_test.sh

```shell
#!/usr/bin/env bash
cd /caok15/Video-Swin-Transformer
# source /root/anaconda3/bin/activate /root/anaconda3/envs/pytorch
# conda activate pytorch
CONFIG="mytrain/swin_base_patch244_window877_tld17k3c_22k_x2_4val.py"
CHECKPOINT="mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/epoch_10.pth"
RESULT_FILE="mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877/result.pkl"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py $CONFIG \
        $CHECKPOINT \
        --out $RESULT_FILE \
        --eval top_k_accuracy mean_class_accuracy
```

测试的时候，需要注意的配置内容：

```shell
# dataset settings
data_root = './data/tld17k/videos'
data_root_val = './data/tld17k/videos'
ann_file_train = f'./data/tld17k/tld17k3c_train_list_videos.txt'
ann_file_val = f'./data/tld17k/tld17k3c_val_list_videos.txt'
ann_file_test = f'./data/tld17k/tld17k3c_all_list_videos.txt'
```

## 如何训练模型

### 数据集标注文件准备

数据集相关工具在 `./tools/data/tld17k` 文件夹内

可以直接根据 `./data/tld17k/tld17k3c_train_list_videos.txt` 文件的结构，直接生成类似的数据集标注文件

或者通过先后调用 `build_data_source.py`， `build_file_list.py` 文件生成。


### 训练配置

使用的训练配置文件在 `./mytrain/swin_base_patch244_window877_tld17k3c_22k.py` 目录下，

训练的时候，需要注意的配置内容：
```shell

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/changde/data/tld17k/videos_crop_256_avi/'
data_root_val = '/changde/data/tld17k/videos_crop_256_avi/'
ann_file_train = f'/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_train_list_videos.txt'
ann_file_val = f'/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_val_list_videos.txt'
ann_file_test = f'/caok15/Video-Swin-Transformer/data/tld17k/tld17k3c_all_list_videos.txt'

evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-5, ...))
# learning policy
total_epochs = 10

# runtime settings
load_from = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877_bak/epoch_10.pth'
work_dir = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k3c_swin_base_22k_patch244_window877'

```

### 使用单个 GPU 进行训练

```shell
./tools/tld17k_train.sh
```

### 使用多个 GPU 进行训练

```shell
./tools/tld17k_dist_train_x2.sh
./tools/tld17k_dist_train_x3.sh
./tools/tld17k_dist_train_x4.sh
```


## 详细教程

目前, MMAction2 提供以下几种更详细的教程：

- [如何编写配置文件](tutorials/1_config.md)
- [如何微调模型](tutorials/2_finetune.md)
- [如何增加新数据集](tutorials/3_new_dataset.md)
- [如何设计数据处理流程](tutorials/4_data_pipeline.md)
- [如何增加新模块](tutorials/5_new_modules.md)
- [如何导出模型为 onnx 格式](tutorials/6_export_model.md)
- [如何自定义模型运行参数](tutorials/7_customize_runtime.md)
