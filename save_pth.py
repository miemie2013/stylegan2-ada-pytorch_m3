# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--g_ema_name', type=str, required=True)
@click.option('--g_name', type=str, required=True)
@click.option('--d_name', type=str, required=True)
def generate_style_mix(
    network_pkl: str,
    g_ema_name: str,
    g_name: str,
    d_name: str
):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        models = legacy.load_network_pkl(f)
    torch.save(models['G'].state_dict(), g_name)
    torch.save(models['G_ema'].state_dict(), g_ema_name)
    torch.save(models['D'].state_dict(), d_name)
    print('Done.')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
