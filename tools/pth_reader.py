import torch
import collections


# step1
'''
checkpoint_file = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k_swin_base_22k_patch244_window877/epoch_20.pth'
pretrained = torch.load(checkpoint_file)['state_dict']
cnt = 0
for key in pretrained:
    print(cnt, key, pretrained[key].shape)
    cnt += 1

need_checkpoint_file = '/caok15/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics400_22k.pth'
need_pretrained = torch.load(need_checkpoint_file)['state_dict']
cnt = 0
for key in need_pretrained:
    print(cnt, key, need_pretrained[key].shape)
    cnt += 1
'''

# step2
'''
checkpoint_file = '/caok15/Video-Swin-Transformer/checkpoints/swin_large_patch4_window12_384_22k.pth'
pretrained = torch.load(checkpoint_file)['model']

new_state_dict = collections.OrderedDict()

for key in pretrained:
    new_key = 'backbone.' + key
    print(key, '    ==>    ', new_key)
    new_state_dict[new_key] = pretrained[key]

out = '/caok15/Video-Swin-Transformer/checkpoints/my_swin_large_patch4_window12_384_22k.pth'
torch.save(new_state_dict, out)
'''

# step3

out = '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/epoch_10.pth'

pretrained = torch.load(out)

for key in pretrained:
    print(key)

new_ = {
    'state_dict': pretrained['state_dict']
}
torch.save(new_, '/caok15/Video-Swin-Transformer/mytrain/work_dirs/tld17k5c_swin_base_22k_patch244_window877/tiny_model.pth')
