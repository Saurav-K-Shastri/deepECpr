import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.functional import mse_loss
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity

from fastmri_utils2.math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)


"""Default data type"""
dtype = torch.float32

"""Transform PIL image to torch.Tensor (C, H, W)"""
to_tensor = transforms.ToTensor()

"""Transform torch.Tensor (C, H, W) to PIL Image"""
to_PIL = transforms.ToPILImage()

def compute_psnr(target, img2):
    target = np.clip(target, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((target - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def psnr(clean, noisy):
    
    return np.array([calc_psnr_complex_single_image(n, c, max=torch.max(complex_abs(c.permute(1,2,0)))) for c, n in zip(clean, noisy)]).mean()

def psnr_real(clean, noisy):
    
    return np.array([calc_psnr(n, c, max=torch.max(c)) for c, n in zip(clean, noisy)]).mean()

def psnr_all_im(clean, noisy):
    
    return np.array([calc_psnr_complex_single_image(n, c, max=torch.max(complex_abs(c.permute(1,2,0)))) for c, n in zip(clean, noisy)])

def psnr_all_im_abs(clean, noisy):
    
    return np.array([calc_psnr(complex_abs(n.permute(1,2,0)), complex_abs(c.permute(1,2,0)), max=torch.max(complex_abs(c.permute(1,2,0)))) for c, n in zip(clean, noisy)])

def calc_psnr(test_image, target_image, max=1.):
    """Calculate PSNR of images."""
    mse = mse_loss(test_image, target_image)
    return 20 * torch.log10(max / torch.sqrt(mse)).item()

def calc_rsnr(test_image, target_image):
    """Calculate rSNR of images."""
    mse = mse_loss(test_image, target_image)
    x_sq = torch.mean(target_image**2)
    rSNR = x_sq/(mse)
    return rSNR.item()

def calc_SSIM(x_rec,x_gt):
    clean = x_gt.detach().numpy().astype(np.float32)
    noisy = x_rec.detach().numpy().astype(np.float32)
    SSIM_val = structural_similarity(clean, noisy, data_range=clean.max())
    return SSIM_val


def calc_psnr_complex_single_image(test_image, target_image, max=1.):
    """Calculate PSNR of images."""
    mse = mse_loss(test_image, target_image)*2 # multiplied with 2 bcs complex valued tensor has extra dimension
    return 10 * torch.log10(max**2 / mse).item()





def ssim_complex(gt: np.ndarray, pred: np.ndarray, maxval = None ) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim >= 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.shape[-1] == 2:
        is_complex = True
    else:
        is_complex = False

    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        if is_complex:
            #Note: the channel axis should correspond to the dimension with real and imaginary
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], channel_axis=-1, data_range=maxval
            )
        else:
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], data_range=maxval
            )

    return ssim / gt.shape[0]

