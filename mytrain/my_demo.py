import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = '/caok15/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py'
device = 'cuda:0'  # or 'cpu'
device = torch.device(device)
checkpoint_file = '/caok15/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics400_22k.pth'

model = init_recognizer(config=config_file, checkpoint=checkpoint_file, device=device)
# inference the demo video
results = inference_recognizer(model, video_path='/caok15/Video-Swin-Transformer/demo/demo.mp4',
                               label_path='/caok15/Video-Swin-Transformer/demo/label_map_k400.txt')


print('result:')
for result in results:
    print(result)
