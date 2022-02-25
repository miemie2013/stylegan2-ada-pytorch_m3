
import torch
import paddle
import os
import numpy as np
from paddle_networks2 import StyleGANv2ADA_SynthesisNetwork

img_resolution = 512
img_channels = 3

synthesis_kwargs = dict(
    channel_base=32768,
    channel_max=512,
    num_fp16_res=4,
    conv_clamp=256,
)

z_dim = 512
c_dim = 0
w_dim = 512

mapping_kwargs = dict(
    num_layers=8,
)

synthesis = StyleGANv2ADA_SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
synthesis.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

synthesis_std = synthesis.state_dict()

ckpt_file = 'pytorch_synthesis.pth'
save_name = 'pytorch_synthesis.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


synthesis_dic = {}
for key, value in state_dict.items():
    synthesis_dic[key] = value.data.numpy()


map = {}

conv_i = 0
torgb_i = 0
for block_idx, res in enumerate(synthesis.block_resolutions):
    in_channels = synthesis.channels_dict[res // 2] if res > 4 else 0
    is_last = synthesis.is_lasts[block_idx]
    architecture = synthesis.architectures[block_idx]

    if in_channels == 0:
        map[f'b{res}.const'] = 'const'
    else:
        pass

    # Main layers.
    if in_channels == 0:
        map[f'b{res}.conv1'] = 'convs.%d'%(conv_i)
        conv_i += 1
    # elif self.architecture == 'resnet':
    #     y = self.skip(x, gain=np.sqrt(0.5))
    #     x = self.conv0(x, ws[:, i + 1], fused_modconv=fused_modconv, **layer_kwargs)
    #     x = self.conv1(x, ws[:, i + 1], fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
    #     x = y.add_(x)
    else:
        map[f'b{res}.conv0'] = 'convs.%d'%(conv_i)
        map[f'b{res}.conv1'] = 'convs.%d'%(conv_i + 1)
        conv_i += 2

    # ToRGB.
    map[f'b{res}.resample_filter'] = f"resample_filter_{block_idx}"
    if is_last or architecture == 'skip':
        map[f'b{res}.torgb'] = 'torgbs.%d'%(torgb_i)
        torgb_i += 1


for key in synthesis_dic.keys():
    name2 = None
    for key2 in map.keys():
        if key2 in key:
            name2 = key.replace(key2, map[key2])
            break
    w = synthesis_dic[key]
    if '.linear.weight' in key:
        print('.linear.weight...')
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print('noise_strength...')
        w = np.reshape(w, [1, ])
    # print(key)
    copy(name2, w, synthesis_std)
synthesis.set_state_dict(synthesis_std)

paddle.save(synthesis_std, save_name)

