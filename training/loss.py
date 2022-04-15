
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch.nn.parallel import DistributedDataParallel as DDP

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------


def save_tensor(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        dic[key] = tensor.cpu().detach().numpy()

def print_diff(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        ddd = np.sum((dic[key] - tensor.cpu().detach().numpy()) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))


class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        '''
        除了self.augment_pipe，其它3个 self.G_mapping、self.G_synthesis、self.D 都是DDP模型。
        只有DDP模型才能使用with module.no_sync():
        '''
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            self.style_mixing_prob = -1.0
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        '''
        除了self.augment_pipe，其它3个 self.G_mapping、self.G_synthesis、self.D 都是DDP模型。
        只有DDP模型才能使用with module.no_sync():
        '''
        if self.augment_pipe is not None:
            # img = self.augment_pipe(img)
            debug_percentile = 0.7
            img = self.augment_pipe(img, debug_percentile)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, dic, save_npz):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        '''
        多卡训练时，传进来的sync都是True。
        run_G()和run_D()中，都有
        with misc.ddp_sync(module, sync):
        这句的意思是：如果是单卡模式，什么都不做；
                    如果是多卡模式，且sync==False时，等于
        with module.no_sync():
        这用于DDP模型中的梯度累加，DDP模型训练时，每一次优化器step()之前，可能有K次loss.backward()，
        那么，前K-1次loss.backward()应该使用with module.no_sync():使得梯度可以累加，最后1次loss.backward()。
        '''
        aaaaaaaa1 = training_stats._counters

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            # 训练生成器，判别器应该冻结，而且希望fake_img的gen_logits越大越好（愚弄D，使其判断是真图片），所以损失是-log(sigmoid(gen_logits))
            # 每个step都做1次
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                if save_npz:
                    save_tensor(dic, phase + ' gen_img', gen_img)
                    save_tensor(dic, phase + ' _gen_ws', _gen_ws)
                else:
                    print_diff(dic, phase + ' gen_img', gen_img)
                    print_diff(dic, phase + ' _gen_ws', _gen_ws)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                if save_npz:
                    save_tensor(dic, phase + ' gen_logits', gen_logits)
                else:
                    print_diff(dic, phase + ' gen_logits', gen_logits)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                G_mapping = self.G_mapping.module if isinstance(self.G_mapping, DDP) else self.G_mapping
                G_synthesis = self.G_synthesis.module if isinstance(self.G_synthesis, DDP) else self.G_synthesis
                D = self.D.module if isinstance(self.D, DDP) else self.D
                m_w_grad = G_mapping.fc7.weight.grad
                m_b_grad = G_mapping.fc7.bias.grad
                s_w_grad = G_synthesis.b32.conv0.affine.weight.grad
                s_b_grad = G_synthesis.b32.conv0.affine.bias.grad
                d_w_grad = D.b32.conv0.weight.grad
                d_b_grad = D.b32.conv0.bias.grad
                if save_npz:
                    save_tensor(dic, phase + ' m_w_grad', m_w_grad)
                    save_tensor(dic, phase + ' m_b_grad', m_b_grad)
                    save_tensor(dic, phase + ' s_w_grad', s_w_grad)
                    save_tensor(dic, phase + ' s_b_grad', s_b_grad)
                    save_tensor(dic, phase + ' d_w_grad', d_w_grad)
                    save_tensor(dic, phase + ' d_b_grad', d_b_grad)
                else:
                    print_diff(dic, phase + ' m_w_grad', m_w_grad)
                    print_diff(dic, phase + ' m_b_grad', m_b_grad)
                    print_diff(dic, phase + ' s_w_grad', s_w_grad)
                    print_diff(dic, phase + ' s_b_grad', s_b_grad)
                    print_diff(dic, phase + ' d_w_grad', d_w_grad)
                    print_diff(dic, phase + ' d_b_grad', d_b_grad)

        # Gpl: Apply path length regularization.
        if do_Gpl:
            # 训练生成器，判别器应该冻结（其实也没有跑判别器），是生成器的梯度惩罚损失（一种高级一点的梯度裁剪）
            # 每4个step做1次
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                batch_size = max(batch_size, 1)
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                if save_npz:
                    save_tensor(dic, phase + ' gen_img', gen_img)
                    save_tensor(dic, phase + ' gen_ws', gen_ws)
                else:
                    print_diff(dic, phase + ' gen_img', gen_img)
                    print_diff(dic, phase + ' gen_ws', gen_ws)
                # pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                if save_npz:
                    save_tensor(dic, phase + ' pl_grads', pl_grads)
                else:
                    print_diff(dic, phase + ' pl_grads', pl_grads)
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
                G_mapping = self.G_mapping.module if isinstance(self.G_mapping, DDP) else self.G_mapping
                G_synthesis = self.G_synthesis.module if isinstance(self.G_synthesis, DDP) else self.G_synthesis
                D = self.D.module if isinstance(self.D, DDP) else self.D
                m_w_grad = G_mapping.fc7.weight.grad
                m_b_grad = G_mapping.fc7.bias.grad
                s_w_grad = G_synthesis.b32.conv0.affine.weight.grad
                s_b_grad = G_synthesis.b32.conv0.affine.bias.grad
                d_w_grad = D.b32.conv0.weight.grad
                d_b_grad = D.b32.conv0.bias.grad
                if save_npz:
                    save_tensor(dic, phase + ' m_w_grad', m_w_grad)
                    save_tensor(dic, phase + ' m_b_grad', m_b_grad)
                    save_tensor(dic, phase + ' s_w_grad', s_w_grad)
                    save_tensor(dic, phase + ' s_b_grad', s_b_grad)
                    save_tensor(dic, phase + ' d_w_grad', d_w_grad)
                    save_tensor(dic, phase + ' d_b_grad', d_b_grad)
                else:
                    print_diff(dic, phase + ' m_w_grad', m_w_grad)
                    print_diff(dic, phase + ' m_b_grad', m_b_grad)
                    print_diff(dic, phase + ' s_w_grad', s_w_grad)
                    print_diff(dic, phase + ' s_b_grad', s_b_grad)
                    print_diff(dic, phase + ' d_w_grad', d_w_grad)
                    print_diff(dic, phase + ' d_b_grad', d_b_grad)

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            # 训练判别器，生成器应该冻结，而且希望fake_img的gen_logits越小越好（判断是假图片），所以损失是-log(1 - sigmoid(gen_logits))
            # 每个step都做1次
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                if save_npz:
                    save_tensor(dic, phase + ' gen_img', gen_img)
                    save_tensor(dic, phase + ' _gen_ws', _gen_ws)
                else:
                    print_diff(dic, phase + ' gen_img', gen_img)
                    print_diff(dic, phase + ' _gen_ws', _gen_ws)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                if save_npz:
                    save_tensor(dic, phase + ' gen_logits', gen_logits)
                else:
                    print_diff(dic, phase + ' gen_logits', gen_logits)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
                G_mapping = self.G_mapping.module if isinstance(self.G_mapping, DDP) else self.G_mapping
                G_synthesis = self.G_synthesis.module if isinstance(self.G_synthesis, DDP) else self.G_synthesis
                D = self.D.module if isinstance(self.D, DDP) else self.D
                m_w_grad = G_mapping.fc7.weight.grad
                m_b_grad = G_mapping.fc7.bias.grad
                s_w_grad = G_synthesis.b32.conv0.affine.weight.grad
                s_b_grad = G_synthesis.b32.conv0.affine.bias.grad
                d_w_grad = D.b32.conv0.weight.grad
                d_b_grad = D.b32.conv0.bias.grad
                if save_npz:
                    save_tensor(dic, phase + ' backward0 m_w_grad', m_w_grad)
                    save_tensor(dic, phase + ' backward0 m_b_grad', m_b_grad)
                    save_tensor(dic, phase + ' backward0 s_w_grad', s_w_grad)
                    save_tensor(dic, phase + ' backward0 s_b_grad', s_b_grad)
                    save_tensor(dic, phase + ' backward0 d_w_grad', d_w_grad)
                    save_tensor(dic, phase + ' backward0 d_b_grad', d_b_grad)
                else:
                    print_diff(dic, phase + ' backward0 m_w_grad', m_w_grad)
                    print_diff(dic, phase + ' backward0 m_b_grad', m_b_grad)
                    print_diff(dic, phase + ' backward0 s_w_grad', s_w_grad)
                    print_diff(dic, phase + ' backward0 s_b_grad', s_b_grad)
                    print_diff(dic, phase + ' backward0 d_w_grad', d_w_grad)
                    print_diff(dic, phase + ' backward0 d_b_grad', d_b_grad)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                if save_npz:
                    save_tensor(dic, phase + ' real_logits', real_logits)
                else:
                    print_diff(dic, phase + ' real_logits', real_logits)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    # 训练判别器，生成器应该冻结，而且希望real_img的gen_logits越大越好（判断是真图片），所以损失是-log(sigmoid(real_logits))
                    # 每个step都做1次
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if save_npz:
                        save_tensor(dic, phase + ' loss_Dreal', loss_Dreal)
                    else:
                        print_diff(dic, phase + ' loss_Dreal', loss_Dreal)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    # 训练判别器，生成器应该冻结（其实也没有跑判别器），是判别器的梯度惩罚损失（一种高级一点的梯度裁剪）
                    # 每16个step做1次
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    if save_npz:
                        save_tensor(dic, phase + ' r1_grads', r1_grads)
                    else:
                        print_diff(dic, phase + ' r1_grads', r1_grads)
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                G_mapping = self.G_mapping.module if isinstance(self.G_mapping, DDP) else self.G_mapping
                G_synthesis = self.G_synthesis.module if isinstance(self.G_synthesis, DDP) else self.G_synthesis
                D = self.D.module if isinstance(self.D, DDP) else self.D
                m_w_grad = G_mapping.fc7.weight.grad
                m_b_grad = G_mapping.fc7.bias.grad
                s_w_grad = G_synthesis.b32.conv0.affine.weight.grad
                s_b_grad = G_synthesis.b32.conv0.affine.bias.grad
                d_w_grad = D.b32.conv0.weight.grad
                d_b_grad = D.b32.conv0.bias.grad
                if do_Dmain:
                    if save_npz:
                        save_tensor(dic, phase + ' backward1 m_w_grad', m_w_grad)
                        save_tensor(dic, phase + ' backward1 m_b_grad', m_b_grad)
                        save_tensor(dic, phase + ' backward1 s_w_grad', s_w_grad)
                        save_tensor(dic, phase + ' backward1 s_b_grad', s_b_grad)
                        save_tensor(dic, phase + ' backward1 d_w_grad', d_w_grad)
                        save_tensor(dic, phase + ' backward1 d_b_grad', d_b_grad)
                    else:
                        print_diff(dic, phase + ' backward1 m_w_grad', m_w_grad)
                        print_diff(dic, phase + ' backward1 m_b_grad', m_b_grad)
                        print_diff(dic, phase + ' backward1 s_w_grad', s_w_grad)
                        print_diff(dic, phase + ' backward1 s_b_grad', s_b_grad)
                        print_diff(dic, phase + ' backward1 d_w_grad', d_w_grad)
                        print_diff(dic, phase + ' backward1 d_b_grad', d_b_grad)
                if do_Dr1:
                    if save_npz:
                        save_tensor(dic, phase + ' m_w_grad', m_w_grad)
                        save_tensor(dic, phase + ' m_b_grad', m_b_grad)
                        save_tensor(dic, phase + ' s_w_grad', s_w_grad)
                        save_tensor(dic, phase + ' s_b_grad', s_b_grad)
                        save_tensor(dic, phase + ' d_w_grad', d_w_grad)
                        save_tensor(dic, phase + ' d_b_grad', d_b_grad)
                    else:
                        print_diff(dic, phase + ' m_w_grad', m_w_grad)
                        print_diff(dic, phase + ' m_b_grad', m_b_grad)
                        print_diff(dic, phase + ' s_w_grad', s_w_grad)
                        print_diff(dic, phase + ' s_b_grad', s_b_grad)
                        print_diff(dic, phase + ' d_w_grad', d_w_grad)
                        print_diff(dic, phase + ' d_b_grad', d_b_grad)

#----------------------------------------------------------------------------
