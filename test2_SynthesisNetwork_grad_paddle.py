
import paddle
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
synthesis.set_state_dict(paddle.load("pytorch_synthesis.pdparams"))


pre_name = 'synthesis'
dic2 = np.load('synthesis_grad.npz')
ws = dic2[pre_name + '.input']
img_pytorch = dic2[pre_name + '.output']
dimg_dws_pytorch = dic2[pre_name + '.dimg_dws']
ws = paddle.to_tensor(ws)
ws.stop_gradient = False


# img = synthesis(ws, dic, pre_name + '.synthesis')
img = synthesis(ws)

dimg_dws = paddle.grad(
    outputs=[img.sum()],
    inputs=[ws],
    create_graph=True,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
    retain_graph=True)[0]

# paddle版SynthesisNetwork与pytorch版SynthesisNetwork有相同的输出
img_paddle = img.numpy()
ddd = np.sum((img_pytorch - img_paddle)**2)
print('ddd=%.6f' % ddd)


# paddle版SynthesisNetwork与pytorch版SynthesisNetwork有不同的梯度
dimg_dws_paddle = dimg_dws.numpy()
ddd = np.sum((dimg_dws_pytorch - dimg_dws_paddle)**2)
print('ddd=%.6f' % ddd)

# 只有最后一个ws(ws[:, -1, :])有梯度。
dimg_dws_paddle = dimg_dws.numpy()
ddd = np.sum((dimg_dws_pytorch[0][15] - dimg_dws_paddle[0][15])**2)
print('ddd=%.6f' % ddd)

# 只有最后一个ws(ws[:, -1, :])有梯度。
dimg_dws_paddle = dimg_dws.numpy()
ddd = np.sum((dimg_dws_pytorch[1][15] - dimg_dws_paddle[1][15])**2)
print('ddd=%.6f' % ddd)


print()
