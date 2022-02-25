
import torch
import numpy as np
from training.networks import SynthesisNetwork

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

synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
synthesis.train()
torch.save(synthesis.state_dict(), "pytorch_synthesis.pth")

batch_size = 2
ws0 = torch.randn([batch_size, 1, 512])
ws = ws0.repeat([1, synthesis.num_ws, 1])
ws.requires_grad_(True)

dic = {}
pre_name = 'synthesis'

img = synthesis(ws, dic, pre_name + '.synthesis')
dimg_dws = torch.autograd.grad(outputs=[img.sum()], inputs=[ws], create_graph=True,
                               only_inputs=True)[0]
dic[pre_name + '.dimg_dws'] = dimg_dws.cpu().detach().numpy()
dic[pre_name + '.output'] = img.cpu().detach().numpy()
dic[pre_name + '.input'] = ws.cpu().detach().numpy()
np.savez('synthesis_grad', **dic)
print()
