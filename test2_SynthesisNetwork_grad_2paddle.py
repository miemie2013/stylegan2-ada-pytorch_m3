
import torch
import paddle
import os
import numpy as np
from paddle_networks import StyleGANv2ADA_SynthesisNetwork

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

for key in synthesis_dic.keys():
    name2 = key
    w = synthesis_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, synthesis_std)
synthesis.set_state_dict(synthesis_std)

paddle.save(synthesis_std, save_name)

