import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

sys.path.append('../')
sys.path.append('../../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from fastmri_utils2.metric import compute_psnr

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler

from utils_deepECpr.algo_utils import *


def run_HIO(y, true_alg_run, measurement_type = 'OSF', beta_HIO = 0.9, HIO_iter_trials = 50, HIO_iter_final = 1000, number_of_trials = 50):

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    imsize = 256
    
    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize
    SamplingRate = p

    if measurement_type == 'OSF':

        A_op_complex_foo = A_op_complex_HIO_OSF(p,imsize)
        # zero padding masks
        oversampled_image_size = int(np.sqrt(m))       
        support = torch.zeros(1,1,3,oversampled_image_size,oversampled_image_size)
        pad = (oversampled_image_size - imsize)
        pad_left = pad//2 # same amount at top
        pad_right = pad - pad_left # same amount at bottom
        support[:,:,:,pad_left:pad_left+imsize,pad_left:pad_left+imsize] = 1
    else:

        raise NotImplementedError
        
    y_complex = torch.view_as_complex(torch.stack([1*y, 0*y], dim=-1)).contiguous()

    x_rec_HIO_best = get_the_best_HIO_recon(1*y_complex, A_op_complex_foo, beta_HIO, HIO_iter_trials ,HIO_iter_final,  support, d, m, pad_left, imsize, number_of_trials = number_of_trials, algo_trial_num = true_alg_run)

    x_rec_HIO_best_orientation_corrected_correlation = fix_channel_orientation_using_correlation(1*x_rec_HIO_best)

    return x_rec_HIO_best_orientation_corrected_correlation

