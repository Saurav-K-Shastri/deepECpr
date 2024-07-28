'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import numpy as np
from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, fft2_mul_chan
from util.fastmri_utils import complex_mul

# =================
# Operation classes
# =================

def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude
    

# @register_operator(name='phase_retrieval_coded_diffraction')
# class PhaseRetrievalOperatorCD(NonLinearOperator):
#     def __init__(self, fixed_rand_mat_3_chan, SamplingRate, device):
#         self.fixed_rand_mat_3_chan = fixed_rand_mat_3_chan
#         self.SamplingRate = SamplingRate
#         self.device = device
        
#     def forward(self, data, **kwargs):
#         my_complex_image = torch.stack((data,torch.zeros_like(data)),dim=-1)
#         my_complex_image_new = my_complex_image.repeat(1,self.SamplingRate,1,1,1).contiguous()
#         inp_data = complex_mul(my_complex_image_new,self.fixed_rand_mat_3_chan)
#         amplitude = fft2_mul_chan(inp_data).abs()
#         return amplitude
#         # amplitude_sq = fft2_mul_chan(inp_data)
#         # return amplitude_sq

# @register_operator(name='phase_retrieval_coded_diffraction_0_1_scaling')
# class PhaseRetrievalOperatorCD(NonLinearOperator):
#     def __init__(self, fixed_rand_mat_3_chan, SamplingRate, device):
#         self.fixed_rand_mat_3_chan = fixed_rand_mat_3_chan
#         self.SamplingRate = SamplingRate
#         self.device = device
        
#     def forward(self, data2, **kwargs):
#         data = (data2)*0.5 + 0.5
#         my_complex_image = torch.stack((data,torch.zeros_like(data)),dim=-1)
#         my_complex_image_new = my_complex_image.repeat(1,self.SamplingRate,1,1,1).contiguous()
#         inp_data = complex_mul(my_complex_image_new,self.fixed_rand_mat_3_chan)
#         amplitude = fft2_mul_chan(inp_data).abs()
#         return amplitude
#         # amplitude_sq = fft2_mul_chan(inp_data)
#         # return amplitude_sq
    
@register_operator(name='phase_retrieval_cdp_0_255_scaling')
class PhaseRetrievalOperatorCD_CDP(NonLinearOperator):
    def __init__(self, fixed_rand_mat_3_chan, SamplingRate):
        self.fixed_rand_mat_3_chan = fixed_rand_mat_3_chan
        self.SamplingRate = SamplingRate
        
    def forward(self, data2, **kwargs):
        data = 255*((data2)*0.5 + 0.5).unsqueeze(1).contiguous()
        my_complex_image = torch.stack((data,torch.zeros_like(data)),dim=-1)
        my_complex_image_new = my_complex_image.repeat(1,self.SamplingRate,1,1,1,1).contiguous()
        inp_data = complex_mul(my_complex_image_new,self.fixed_rand_mat_3_chan)
        op_data = fft2_mul_chan(inp_data)*np.sqrt(1/self.SamplingRate)
        amplitude = op_data.abs()
        return amplitude

    
@register_operator(name='phase_retrieval_osf_0_255_scaling')
class PhaseRetrievalOperatorCD(NonLinearOperator):
    def __init__(self, oversampling_factor, image_size):
        
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.n = self.image_size*self.image_size
        self.m = int(self.n*self.oversampling_factor)
        self.oversampled_image_size = int(np.sqrt(self.m))
        self.pad = (self.oversampled_image_size - self.image_size)
        self.pad_left = self.pad//2 # same amount at top
        self.pad_right = self.pad - self.pad_left # same amount at bottom
        
    def forward(self, data2, **kwargs):
        data = 255*((data2)*0.5 + 0.5)
        X1 = torch.nn.functional.pad(data, (self.pad_left, self.pad_right, self.pad_left, self.pad_right))
        X2 = (torch.stack((X1,torch.zeros_like(X1)),dim=-1))
        amplitude = fft2_mul_chan(X2).abs()
        return amplitude

# @register_operator(name='phase_retrieval_osf_0_1_scaling')
# class PhaseRetrievalOperatorCD(NonLinearOperator):
#     def __init__(self, oversampling_factor, image_size):
        
#         self.oversampling_factor = oversampling_factor
#         self.image_size = image_size
#         self.n = self.image_size*self.image_size
#         self.m = int(self.n*self.oversampling_factor)
#         self.oversampled_image_size = int(np.sqrt(self.m))
#         self.pad = (self.oversampled_image_size - self.image_size)
#         self.pad_left = self.pad//2 # same amount at top
#         self.pad_right = self.pad - self.pad_left # same amount at bottom
        
#     def forward(self, data2, **kwargs):
#         data = ((data2)*0.5 + 0.5)
#         X1 = torch.nn.functional.pad(data, (self.pad_left, self.pad_right, self.pad_left, self.pad_right))
#         X2 = (torch.stack((X1,torch.zeros_like(X1)),dim=-1))*np.sqrt(self.m/self.n)
#         amplitude = fft2_mul_chan(X2).abs()
#         return amplitude
    





@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data, i):
        return self.forward(data,i)
    
    @abstractmethod
    def forward(self, data, i):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data, i):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data, i):
        torch.random.manual_seed(i)
        w0 = self.sigma * torch.randn_like(data, device=data.device)
        # print(w0.shape)
        # print(torch.sum(data.cpu()**2))
        # print(torch.norm(w0.cpu()))
        # SNRdB_test = 20*np.log10(torch.norm(data.cpu())/torch.norm(w0.cpu()))-3
        # print(" ")
        # print('SNRdB_test = ', SNRdB_test)
        # print(" ")
        return data + w0
    
@register_noise(name='gaussian_poisson')
class GaussianPoissonNoise(Noise):
    def __init__(self, alpha,SamplingRate):
        self.alpha = alpha
        self.SamplingRate = SamplingRate
    
    def forward(self, z0, i):

        torch.random.manual_seed(i)

        z0_complex = 255*torch.stack((z0[:,0:self.SamplingRate,:,:,:],z0[:,self.SamplingRate:,:,:,:]), dim = -1)

        noise_mat = torch.randn(z0.shape[0], self.SamplingRate, z0.shape[2], z0.shape[3], z0.shape[4])
        intensity_noise = self.alpha*complex_abs(z0_complex)*noise_mat
        z2 = complex_abs(z0_complex)**2
        y2_full = z2.clone() + intensity_noise
        y2 = y2_full*torch.gt(y2_full, 0).type_as(y2_full)
        y = torch.sqrt(y2)/255
        err = y - (complex_abs(z0_complex)/255)
        sig = torch.std(err)

        return y, sig

@register_noise(name='gaussian_poisson_cdp_255')
class GaussianPoissonNoise(Noise):
    def __init__(self, alpha, SamplingRate):
        self.alpha = alpha
        self.SamplingRate = SamplingRate

    def forward(self, z0, i):

        torch.random.manual_seed(i)
        torch.manual_seed(i)

        z0_abs = complex_abs(z0)
        noise_mat = torch.randn(z0.shape[0], self.SamplingRate, z0.shape[2], z0.shape[3], z0.shape[4]).to(z0_abs.device)
        intensity_noise = self.alpha*z0_abs*noise_mat
        z2 = z0_abs**2
        y2_full = z2.clone() + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (z0_abs)
        sig = torch.std(err)
        

        return y, sig
    

@register_noise(name='gaussian_poisson_osf_255')
class GaussianPoissonNoise(Noise):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def forward(self, z0, i):

        torch.random.manual_seed(i)
        torch.manual_seed(i)
        z0_abs = complex_abs(z0)
        noise_mat = (torch.randn(z0_abs.shape[0], 1, z0_abs.shape[1], z0_abs.shape[2], z0_abs.shape[3]).to(z0_abs.device))[:,0,:,:,:] # had to do this so as to match the measurements exactly to the ones used in deepECpr and other algos
        # noise_mat = torch.randn_like(z0_abs)
        intensity_noise = self.alpha*z0_abs*noise_mat
        z2 = z0_abs**2
        y2_full = z2.clone() + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))
        err = y - (z0_abs)
        sig = torch.std(err)
        

        return y, sig
    
@register_noise(name='gaussian_poisson_osf')
class GaussianPoissonNoise(Noise):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def forward(self, z0, i):

        torch.random.manual_seed(i)

        z0_abs = 255*complex_abs(z0)
        noise_mat = torch.randn_like(z0_abs)
        intensity_noise = self.alpha*z0_abs*noise_mat
        z2 = z0_abs**2
        y2_full = z2.clone() + intensity_noise
        y2 = torch.abs(y2_full)
        y = torch.abs(torch.sqrt(y2))/255
        err = y - ((z0_abs)/255)
        sig = torch.std(err)
        
        return y, sig
    

    





# @register_noise(name='gaussian_SNR')
# class GaussianNoise(Noise):
#     def __init__(self, SNR,SamplingRate):
#         self.SNRdB = SNRdB
#         self.SamplingRate = SamplingRate
    
#     def forward(self, z0):

#         sig_1 = torch.sqrt(torch.mean(z0.pow(2)))*torch.pow(torch.tensor(10), -self.SNRdB/20)
#         w0 = sig_1*torch.randn(z0.shape[0], self.SamplingRate, x0.shape[2], x0.shape[3], x0.shape[4])
#         SNRdB_test = 20*np.log10(torch.norm(z0)/torch.norm(w0))-3
#         print('SNRdB_test = ', SNRdB_test)
#         y = torch.zeros(x0.shape[0], SamplingRate, x0.shape[2], x0.shape[3])
#         y = complex_abs(torch.stack((z0[:,0:SamplingRate,:,:,:],z0[:,SamplingRate:,:,:,:]), dim = -1)) + w0
#         z0_complex = 255*torch.stack((z0[:,0:SamplingRate,:,:,:],z0[:,SamplingRate:,:,:,:]), dim = -1)
#         err = y - (complex_abs(z0_complex)/255)
#         sig = torch.std(err)


#         return data + torch.randn_like(data, device=data.device) * self.sigma

# @register_noise(name='poisson_prDeep')
# class PoissonNoise_prDeep(Noise):
#     def __init__(self, alpha):
#         self.alpha = alpha

#     def forward(self, data):
        

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data, i):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)