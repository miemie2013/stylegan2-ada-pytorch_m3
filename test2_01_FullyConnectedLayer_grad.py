
import torch
import numpy as np
from training.networks import FullyConnectedLayer


in_channels = 512
w_dim = 512

activation = 'linear'
activation = 'lrelu'

fullyConnectedLayer = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
fullyConnectedLayer.train()
torch.save(fullyConnectedLayer.state_dict(), "pytorch_fullyConnectedLayer.pth")

batch_size = 2
ws = torch.randn([batch_size, 512])
ws.requires_grad_(True)

dic = {}
pre_name = 'fullyConnectedLayer'

styles = fullyConnectedLayer(ws, dic, pre_name + '.fullyConnectedLayer')
dstyles_dws = torch.autograd.grad(outputs=[styles.sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
dic[pre_name + '.dstyles_dws'] = dstyles_dws.cpu().detach().numpy()
dic[pre_name + '.output'] = styles.cpu().detach().numpy()
dic[pre_name + '.input'] = ws.cpu().detach().numpy()
np.savez('fullyConnectedLayer_grad', **dic)
print()
