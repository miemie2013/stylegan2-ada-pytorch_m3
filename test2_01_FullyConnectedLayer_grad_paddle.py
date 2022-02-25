
import paddle
import numpy as np
from paddle_networks2 import FullyConnectedLayer

in_channels = 512
w_dim = 512

activation = 'linear'
activation = 'lrelu'

fullyConnectedLayer = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
fullyConnectedLayer.train()
fullyConnectedLayer.set_state_dict(paddle.load("pytorch_fullyConnectedLayer.pdparams"))


pre_name = 'fullyConnectedLayer'
dic2 = np.load('fullyConnectedLayer_grad.npz')
ws = dic2[pre_name + '.input']
styles_pytorch = dic2[pre_name + '.output']
dstyles_dws_pytorch = dic2[pre_name + '.dstyles_dws']
ws = paddle.to_tensor(ws)
ws.stop_gradient = False


styles = fullyConnectedLayer(ws, dic2, pre_name + '.fullyConnectedLayer')

dstyles_dws = paddle.grad(
    outputs=[styles.sum()],
    inputs=[ws],
    create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
    retain_graph=True)[0]

styles_paddle = styles.numpy()
ddd = np.sum((styles_pytorch - styles_paddle)**2)
print('ddd=%.6f' % ddd)


dstyles_dws_paddle = dstyles_dws.numpy()
ddd = np.sum((dstyles_dws_pytorch - dstyles_dws_paddle)**2)
print('ddd=%.6f' % ddd)


print()
