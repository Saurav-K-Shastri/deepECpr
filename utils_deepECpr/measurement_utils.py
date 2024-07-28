import os, sys

sys.path.append(os.path.dirname(sys.path[0]))
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torchvision.transforms as transforms
from fastmri_utils2.fftc import fft2c_new as fft2c
from fastmri_utils2.fftc import ifft2c_new as ifft2c
from fastmri_utils2.math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)
import cv2
from data.dataloader import get_dataset, get_dataloader
sys.path.append('../utils_deepECpr/')
sys.path.append('../')
sys.path.append('../../')

from fastmri_utils2.metric import compute_psnr
from fastmri_utils2.metric import calc_SSIM

from utils_deepECpr.algo_utils import *

from PIL import Image
import imageio.v2 as imageio

import torch.nn.functional as F
import torch.nn as nn
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler, extract_and_expand
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

from scipy.optimize import minimize
from scipy import signal
import scipy.io as sio

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_FFHQ_measurement(image_number, measurement_type, alpha, data_config_name, verbose = False):

    if image_number >= 30:
        raise ValueError("Image number must be less than 30.")
    
    if verbose:
        print("Chosen Options ")
        print(" ")
        print("Image Number      : ", image_number)
        print("Measurement Type  : ", measurement_type)
        print("Alpha             : ", alpha)
        print(" ")

    data_config_foo = load_yaml(data_config_name)
    # Prepare dataloader
    data_config = data_config_foo['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)

    # main parameters
    seed = 0 # random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    imsize = 256


    Img = (dataset[image_number].unsqueeze(0))
    x0 = torch.zeros(1,2,3,imsize,imsize)
    x0[:,0,:,:,:] = 255*((Img)*0.5 + 0.5)

    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize

    if measurement_type == 'OSF':
        
        if alpha not in {4, 6, 8}:
            raise ValueError('Alpha value must be one of 4, 6, 8 for OSF measurements!')

        A_op_main_foo = A_op_OSF(p,imsize)
        z0 = A_op_main_foo.A(x0)

        torch.random.manual_seed(image_number)
        torch.manual_seed(image_number)

        z0_complex = torch.stack((z0[:,[0],:,:,:],z0[:,[1],:,:,:]), dim = -1)
        noise_mat = torch.randn(x0.shape[0], 1, z0.shape[2], z0.shape[3], z0.shape[4])
        intensity_noise = alpha*complex_abs(z0_complex)*noise_mat
        z2 = complex_abs(z0_complex)**2
        y2_full = 1.0*z2 + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (complex_abs(z0_complex))
        sig = torch.std(err) # Following prDeep's implementation where ground truth value is used; Alternatively one could use average value of sig for each alpha averaged over the training set

        fixed_rand_mat_3_chan_6_dim = None
        
    elif measurement_type == 'CDP':

        if alpha not in {5, 15, 45}:
            raise ValueError('Alpha value must be one of 5, 15, 45 for CDP measurements!')

        SamplingRate = p
        fixed_rand_mat = torch.load('utils_deepECpr/fixed_rand_mat_CDP.pt')
        fixed_rand_mat_3_chan_6_dim = torch.zeros(1,SamplingRate,3,imsize,imsize,2)
        for i in range(SamplingRate):
            for k in range(3):
                fixed_rand_mat_3_chan_6_dim[:,i,k,:,:,:] = fixed_rand_mat[:,i,:,:,:]

        A_op_main_foo = A_op_CDP(fixed_rand_mat_3_chan_6_dim)
        z0 = A_op_main_foo.A(x0)

        torch.random.manual_seed(image_number)
        torch.manual_seed(image_number)

        z0_complex = torch.stack((z0[:,0:SamplingRate,:,:,:],z0[:,SamplingRate:,:,:,:]), dim = -1)
        noise_mat = torch.randn(x0.shape[0], SamplingRate, z0.shape[2], z0.shape[3], z0.shape[4])
        intensity_noise = alpha*complex_abs(z0_complex)*noise_mat
        z2 = complex_abs(z0_complex)**2
        y2_full = 1.0*z2 + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (complex_abs(z0_complex))
        sig = torch.std(err) # Following prDeep's implementation where ground truth value is used; Alternatively one could use average value of sig for each alpha averaged over the training set
        

    else:

        raise ValueError('Measurement type not recognized!')


    return y, sig, x0, z0, fixed_rand_mat_3_chan_6_dim



def get_grayscale_measurement(image_number, image_type, measurement_type, alpha, data_config_name, verbose = False):
    
    if image_number >= 6:
        raise ValueError("Image number must be less than 30.")
    
    if verbose:
        print("Chosen Options ")
        print(" ")
        print("Image Type        : ", image_type)
        print("Image Number      : ", image_number)
        print("Measurement Type  : ", measurement_type)
        print("Alpha             : ", alpha)
        print(" ")

    data_config_foo = load_yaml(data_config_name)
    # Prepare dataloader
    data_config = data_config_foo['data']
    file_dir_natural = data_config_foo['data']['root'] + 'natural/'
    file_dir_unnatural = data_config_foo['data']['root'] + 'unnatural/'

    if image_type == 'natural':
        file = file_dir_natural+str(image_number)+'.png'
    else:
        file = file_dir_unnatural+str(image_number)+'.png'

    img = cv2.imread(file)

    # main parameters
    seed = 0 # random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    imsize = 128
    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize

    Img = cv2.resize(img, (int(resize_size), int(resize_size)), interpolation=cv2.INTER_AREA) # use when shrinking image
    Img = np.expand_dims(Img[:, :, 0].copy(), (0,1)) / 255.0

    x0 = torch.zeros(1,2,resize_size,resize_size)
    x0[:,0,:,:] = 255*torch.from_numpy(Img)
    

    if measurement_type == 'OSF':
        
        if alpha not in {4, 6, 8}:
            raise ValueError('Alpha value must be one of 4, 6, 8 for OSF measurements!')

        Omn_func_foo = Omn_func_new(p,resize_size)
        A_op_foo = A_op_Fourier_grayscale(p,resize_size)
        A_op_main_foo = A_op_OSF_grayscale(p,resize_size,Omn_func_foo)

        z0 = A_op_foo.A(Omn_func_foo.A(x0))

        torch.random.manual_seed(image_number)
        torch.manual_seed(image_number)

        z0_complex = torch.stack((z0[:,[0],:,:],z0[:,[1],:,:]), dim = -1)
        noise_mat = torch.randn(x0.shape[0], 1, z0.shape[2], z0.shape[3])
        intensity_noise = alpha*complex_abs(z0_complex)*noise_mat
        z2 = complex_abs(z0_complex)**2
        y2_full = z2.clone() + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (complex_abs(z0_complex))
        sig = torch.std(err) # Following prDeep's implementation where ground truth value is used; Alternatively one could use average value of sig for each alpha averaged over the training set

        fixed_rand_mat = None
        
    elif measurement_type == 'CDP':

        if alpha not in {5, 15, 45}:
            raise ValueError('Alpha value must be one of 5, 15, 45 for CDP measurements!')

        SamplingRate = p
        fixed_rand_mat = torch.from_numpy(sio.loadmat('utils_deepECpr/fixed_rand_mat_CDP_grayscale.mat')['fixed_rand_mat']).contiguous()

        A_op_main_foo = A_op_CDP_grayscale(fixed_rand_mat)
        z0 = A_op_main_foo.A(x0)

        torch.random.manual_seed(image_number)
        torch.manual_seed(image_number)

        z0_complex = torch.stack((z0[:,0:SamplingRate,:,:],z0[:,SamplingRate:,:,:]), dim = -1)
        noise_mat = torch.randn(x0.shape[0], SamplingRate, z0.shape[2], z0.shape[3])
        intensity_noise = alpha*complex_abs(z0_complex)*noise_mat
        z2 = complex_abs(z0_complex)**2
        y2_full = 1.0*z2 + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (complex_abs(z0_complex))
        sig = torch.std(err) # Following prDeep's implementation where ground truth value is used; Alternatively one could use average value of sig for each alpha averaged over the training set
        
    else:

        raise ValueError('Measurement type not recognized!')

    return y, sig, x0, z0, fixed_rand_mat

