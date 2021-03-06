# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    save_npz                = 1,        #
    dist_url                = None,     #
    num_machines            = 1,        #
    machine_rank            = 0,        #
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    print('rrrrrrrrrrrrrrrrrrrrrrrrrrrr')
    print(ema_kimg)
    print(ema_rampup)
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    # allow_tf32 = False
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
        # for name, module in [('G_ema', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    # if rank == 0:
    #     z = torch.empty([batch_gpu, G.z_dim], device=device)
    #     c = torch.empty([batch_gpu, G.c_dim], device=device)
    #     img = misc.print_module_summary(G, [z, c])
    #     misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
        print(f'DDP models:')
    ddp_modules = dict()


    batch_idx = 0
    if not save_npz:
        if batch_idx == 0:
            G_ema.load_state_dict(torch.load("G_ema_00.pth", map_location="cpu"))
            G.load_state_dict(torch.load("G_00.pth", map_location="cpu"))
            D.load_state_dict(torch.load("D_00.pth", map_location="cpu"))
    if save_npz:
        if batch_idx == 0 and rank == 0:
            torch.save(G_ema.state_dict(), "G_ema_00.pth")
            torch.save(G.state_dict(), "G_00.pth")
            torch.save(D.state_dict(), "D_00.pth")


    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            '''
            除了augment_pipe，其它4个 G.mapping、G.synthesis、D、G_ema 都是DDP模型。
            '''
            print(name)
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            # opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if name == 'G':
                opt = torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.9)
            elif name == 'D':
                opt = torch.optim.SGD(module.parameters(), lr=0.002, momentum=0.9)
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    # if rank == 0:
    #     print('Exporting sample images...')
    #     grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    #     save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
    #     grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
    #     grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
    #     images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    #     save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    # while True:
    while batch_idx < 20:
        dic = {}
        print('======================== batch%.5d.npz ========================'%batch_idx)
        if not save_npz:
            dic = np.load('batch%.5d_rank%.2d.npz' % (batch_idx, rank))

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            if batch_idx == 0:
                '''
                假如训练命令的命令行参数是 --gpus=2 --batch 8 --cfg my32
                即总的批大小是8，每卡批大小是4，那么这里
                phase_real_img.shape = [4, 3, 32, 32]
                phase_real_c.shape   = [4, 0]
                batch_gpu            = 4
                即拿到的phase_real_img和phase_real_c是一张卡（一个进程）上的训练样本，（每张卡）批大小是4
                '''
                print('phase_real_img.shape =', phase_real_img.shape)
                print('phase_real_c.shape =', phase_real_c.shape)
                print('batch_gpu =', batch_gpu)
            if save_npz:
                dic['phase_real_img'] = phase_real_img.cpu().detach().numpy()
            else:
                aaaaaaaaa = dic['phase_real_img']
                phase_real_img = torch.Tensor(aaaaaaaaa).to(device).to(torch.float32)

            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            if batch_idx == 0:
                '''
                phase_real_c.split(batch_gpu)的意思是，
                把 phase_real_c 分成 phase_real_c.shape[0] // batch_gpu 份， 每一份的大小是 batch_gpu。
                
                我不太明白这个split()的意义？不管单卡还是多卡训练，len(phase_real_img)始终是1，len(phase_real_c)始终是1
                '''
                print('len(phase_real_img) =', len(phase_real_img))
                print('phase_real_img[0].shape =', phase_real_img[0].shape)
                print('len(phase_real_c) =', len(phase_real_c))
                print('phase_real_c[0].shape =', phase_real_c[0].shape)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            if batch_idx == 0:
                '''
                假如训练命令的命令行参数是 --gpus=2 --batch 8 --cfg my32
                all_gen_z.shape = [4*8, 512]
                '''
                print('all_gen_z.shape =', all_gen_z.shape)
            if save_npz:
                dic['all_gen_z'] = all_gen_z.cpu().detach().numpy()
            else:
                bbbbbbbbb = dic['all_gen_z']
                all_gen_z = torch.Tensor(bbbbbbbbb).to(device).to(torch.float32)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            if batch_idx == 0:
                '''
                all_gen_z.split(batch_size)的意思是，
                把 all_gen_z 分成 all_gen_z.shape[0] // batch_size 份， 每一份的大小是 batch_size。这里份数即 阶段phases 数4.
                
                phase_gen_z.split(batch_gpu)的意思是，
                把 phase_gen_z 分成 phase_gen_z.shape[0] // batch_gpu 份， 每一份的大小是 batch_gpu。这里份数即 显卡 数2.
                
                len(all_gen_z)        = 4
                len(all_gen_z[0])     = 2
                all_gen_z[0][0].shape = [4, 512]
                '''
                print('len(all_gen_z) =', len(all_gen_z))
                print('len(all_gen_z[0]) =', len(all_gen_z[0]))
                print('all_gen_z[0][0].shape =', all_gen_z[0][0].shape)
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            if batch_idx == 0:
                print('Accumulate gradients ...')
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                # if batch_idx > -1:
                if batch_idx == 0:
                    '''
                    破案了，不管单卡还是多卡训练，这个for round_idx ...循环只会循环1次，round_idx始终是0，sync始终是True。
                    假如训练命令的命令行参数是 --gpus=2 --batch 8 --cfg my32
                    
                    则len(phase_gen_z) == len(all_gen_z[0]) == 显卡数量 == 2
                    但是len(phase_real_img) == 1，len(phase_real_c) == 1，
                    即phase_gen_z中有 {显卡数量 - 1} 个噪声被浪费掉。（但是单卡模式时，不会有噪声被浪费，可能这样写是为了兼容单卡模式）
                    '''
                    print('round_idx =', round_idx)
                    print('sync =', sync)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain, dic=dic, save_npz=save_npz)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                # for param in phase.module.parameters():
                #     if param.grad is not None:
                #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
                pass
                # if 'G' in phase['name']:
                #     phase.opt.step()
                #     pass
                # elif 'D' in phase['name']:
                #     phase.opt.step()
                #     pass
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        if save_npz:
            save_npz_name = 'batch%.5d_rank%.2d'%(batch_idx, rank)
            dic['w_avg'] = G.mapping.w_avg.cpu().detach().numpy()
        else:
            kkk = 'w_avg'; ddd = np.sum((dic[kkk] - G.mapping.w_avg.cpu().detach().numpy()) ** 2)
            print('diff=%.6f (%s)' % (ddd, kkk))


        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        if save_npz:
            if batch_idx == 19 and rank == 0:
                if num_gpus > 1:
                    torch.save(G_ema.state_dict(), "G_ema_19.pth")
                    torch.save(G.state_dict(), "G_19.pth")
                    torch.save(D.state_dict(), "D_19.pth")
                else:
                    torch.save(G_ema.state_dict(), "G_ema_19.pth")
                    torch.save(G.state_dict(), "G_19.pth")
                    torch.save(D.state_dict(), "D_19.pth")

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            aaaaaaaaaa1 = ada_stats['Loss/signs/real']
            aaaaaaaaaa2 = ada_target
            aaaaaaaaaa3 = aaaaaaaaaa1 - aaaaaaaaaa2
            aaaaaaaaaa4 = np.sign(aaaaaaaaaa3)
            if save_npz:
                dic['augment_pipe_p'] = augment_pipe.p.cpu().detach().numpy()
                dic['aaaaaaaaaa1'] = np.array(aaaaaaaaaa1)
            else:
                kkk = 'augment_pipe_p'; ddd = np.sum((dic[kkk] - augment_pipe.p.cpu().detach().numpy()) ** 2)
                print('diff=%.6f (%s)' % (ddd, kkk))
                kkk = 'aaaaaaaaaa1'; ddd = np.sum((dic[kkk] - np.array(aaaaaaaaaa1)) ** 2)
                print('diff=%.6f (%s)' % (ddd, kkk))
            adjust = aaaaaaaaaa4 * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        if save_npz:
            np.savez(save_npz_name, **dic)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        # print('cur_tick: %d' % cur_tick)
        # print('cur_nimg: %d' % cur_nimg)
        # print('tick_start_nimg: %d' % tick_start_nimg)
        # print('kimg_per_tick: %d' % kimg_per_tick)
        # print()
        # 咩酱：意思是每训练 kimg_per_tick * 1000 张图片，打印一次。
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        # 咩酱：意思是每训练 10 张图片，打印一次。
        # if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + 10):
        #     continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        #     images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        #     save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        # 咩酱：程序会卡在这里。计算指标FID。所以先注释掉
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
