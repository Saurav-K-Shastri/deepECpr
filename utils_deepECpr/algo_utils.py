import os, sys

sys.path.append(os.path.dirname(sys.path[0]))
import yaml
import numpy as np
import torch
from fastmri_utils2.fftc import fft2c_new as fft2c
from fastmri_utils2.fftc import ifft2c_new as ifft2c
sys.path.append('../')
sys.path.append('../../')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
from guided_diffusion.gaussian_diffusion import extract_and_expand
from fastmri_utils2.math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)

from models.generator_new import DnCNN_BF 


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def find_best_shift(img1, img2):
    best_mse = float('inf')
    best_shift = (0, 0, 0)

    for my_flip in [0 , 1]:
        if my_flip:
            img1 = 1*img1.flip(0).flip(1)
        for row_shift in range(img1.size(0)):
            for col_shift in range(img1.size(1)):
                shifted_img1 = torch.roll(img1, shifts=(row_shift, col_shift), dims=(0, 1))
                current_mse = F.mse_loss(shifted_img1, img2)

                if current_mse < best_mse:
                    best_mse = current_mse
                    best_shift = (row_shift, col_shift,my_flip)

    return best_shift

def find_best_flip(img1, img2):
    best_mse = float('inf')
    best_flip = 0

    for my_flip in [0 , 1]:

        if my_flip:
            img1 = 1*img1.flip(0).flip(1)

        current_mse = F.mse_loss(img1, img2)

        if current_mse < best_mse:
            best_mse = current_mse
            best_flip = my_flip

    return best_flip

def fix_flip(img1, x0_gt):
    
    x_out = torch.zeros_like(x0_gt)

    for channel in range(3):

        x_rec_chan = img1[0,0,channel,:,:]

        x_out_inter = 1*x_rec_chan

        flip_op_foo = find_best_flip(x_rec_chan, x0_gt[0,0,channel, :,:])

        if flip_op_foo:
            x_rec_chan = 1*x_out_inter.flip(0).flip(1)
        else:
            x_rec_chan = 1*x_out_inter       

            # x_rec_chan = torch.roll(x_rec_chan, shifts=(row_shift_foo, column_shift_foo), dims=(0, 1))

        x_out[0,0,channel,:,:] = x_rec_chan

    return x_out


class A_op_OSF:
    
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.n = self.image_size*self.image_size
        self.m = int(self.n*self.oversampling_factor)
        self.oversampled_image_size = int(np.sqrt(self.m))
        self.pad = (self.oversampled_image_size - self.image_size)
        self.pad_left = self.pad//2 # same amount at top
        self.pad_right = self.pad - self.pad_left # same amount at bottom

    def A(self,X):
        X1  = torch.nn.functional.pad(X, (self.pad_left, self.pad_right, self.pad_left, self.pad_right))
        X2 = X1.permute(0,2,3,4,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:,:], X[:,1,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out = X2.permute(0,4,1,2,3).contiguous()
        return out[:,:,:,self.pad_left:self.pad_left+self.image_size,self.pad_left:self.pad_left+self.image_size].contiguous()


class A_op_Fourier:
    
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size

    def A(self,X):
        X2 = X.permute(0,2,3,4,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:,:], X[:,1,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out = X2.permute(0,4,1,2,3).contiguous()
        return out.contiguous()
    
class A_op_CDP:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X1 = X.permute(0,2,3,4,1).unsqueeze(1)
        X1_new = X1.repeat(1,self.Sampling_Rate,1,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1,1).contiguous()
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0:self.Sampling_Rate,:,:,:], X[:,self.Sampling_Rate:,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1,1)
        out = (torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1))*np.sqrt(1/self.Sampling_Rate)
        return out.permute(0,4,1,2,3).contiguous()


    
class A_op_complex_HIO_CDP:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X1 = torch.view_as_real(X.contiguous())

        X1_new = X1.repeat(1,self.Sampling_Rate,1,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1,1).contiguous()
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)

        return torch.view_as_complex(out.contiguous())

    def H(self,X):
        X1 = torch.view_as_real(X.contiguous())

        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1,1)
        out = (torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1))*np.sqrt(1/self.Sampling_Rate)
        out2 = out.unsqueeze(1).contiguous()
        out2[:,:,:,:,:,1] = 0
        return torch.view_as_complex(out2.contiguous())
    

class A_op_CDP_grayscale:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X1 = X.permute(0,2,3,1).unsqueeze(1)
        X1_new = X1.repeat(1,self.Sampling_Rate,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1).contiguous()
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)
        return torch.cat([out[:,:,:,:,0], out[:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0:self.Sampling_Rate,:,:], X[:,self.Sampling_Rate:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1)
        out = (torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1))*np.sqrt(1/self.Sampling_Rate)
        return out.permute(0,3,1,2).contiguous()
    
class A_op_complex_CDP_HIO_grayscale:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X = torch.view_as_real(X.contiguous())
        X1_new = X.repeat(1,self.Sampling_Rate,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1)
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)
        return torch.view_as_complex(out.contiguous())

    def H(self,X):
        X1 = torch.view_as_real(X.contiguous())
        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1)
        out = torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1)*np.sqrt(1/self.Sampling_Rate)
        out2 = out.unsqueeze(1).contiguous()
        out2[:,:,:,:,1] = 0
        return torch.view_as_complex(out2.contiguous())
    

class Omn_func_new:
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.n = self.image_size*self.image_size
        self.m = int(self.n*self.oversampling_factor)
        self.oversampled_image_size = int(np.sqrt(self.m))
        self.pad = (self.oversampled_image_size - self.image_size)
        self.pad_left = self.pad//2 # same amount at top
        self.pad_right = self.pad - self.pad_left # same amount at bottom
    
    def A(self,X):
        X1  = torch.nn.functional.pad(X, (self.pad_left, self.pad_right, self.pad_left, self.pad_right))
        return X1

    def H(self,X):
        X1 = X[:,:,self.pad_left:self.pad_left+self.image_size,self.pad_left:self.pad_left+self.image_size].contiguous()
        return X1
    
class A_op_Fourier_grayscale:
    
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size

    def A(self,X):
        X2 = X.permute(0,2,3,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,0], out[:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:], X[:,1,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out = X2.permute(0,3,1,2).contiguous()
        return out.contiguous()
    

class A_op_OSF_grayscale:
    
    def __init__(self,oversampling_factor, image_size, O_func_foo):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.O_func_foo = O_func_foo


    def A(self,X):
        X1 = self.O_func_foo.A(X)
        X2 = X1.permute(0,2,3,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,0], out[:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:], X[:,1,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out1 = X2.permute(0,3,1,2).contiguous()
        out = self.O_func_foo.H(out1)
        return out
    



class MAP_chan_stage_deepECpr_OSF:
    
    def __init__(self, y, sig):
        self.y_inp = 1*y.squeeze(1)
        self.oversampled_image_size = y.shape[-1]
        self.sig = sig
        self.v = self.sig**2

    def z_hat_and_var_hat_est(self, z_inp, v_bar_half, fixed_var=True):  # note v_bar is total noise i.e. both real and imag chan noise combined variance
        
        v_bar = 2*v_bar_half[0,:,:,:,:]
        
        z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0,:,:,:], 1*z_inp[:,1,:,:,:]], dim=-1).contiguous())

        z_angle_inp = torch.angle(z_complex)
        z_abs_inp = torch.abs(z_complex)

        z_foo = torch.view_as_real(torch.polar(self.y_inp, z_angle_inp)).permute(0,4,1,2,3).contiguous()

        z_hat = ((v_bar)/(v_bar + 2*self.v))*(z_foo) + ((2*self.v)/(v_bar + 2*self.v))*(z_inp)

        if fixed_var:
            var_hat_half = 1*v_bar_half

            return z_hat, var_hat_half
        
        else:

            v_hat_real_axis = ((self.v)*v_bar)/(v_bar + 2*self.v)

            v_hat_imag_axis = (v_bar/(2*z_abs_inp))*(((v_bar)/(v_bar + 2*self.v))*(self.y_inp) + ((2*self.v)/(v_bar + 2*self.v))*(z_abs_inp))
           
            v_hat_full = v_hat_real_axis + v_hat_imag_axis
            
            mask_pos_ext = (z_abs_inp>(self.y_inp/2))[0,:,:,:]

            v_hat = torch.sum(v_hat_full*mask_pos_ext)/(torch.sum(mask_pos_ext) + 1e-18)
            var_hat_half = (v_hat/2)*torch.ones_like(v_bar_half)
            
            return z_hat, var_hat_half
        
class MAP_chan_stage_deepECpr_CDP:
    
    def __init__(self, y, sig):
        self.y_inp = 1*y
        self.SamplingRate = y.shape[1]
        self.sig = sig
        self.v = self.sig**2

    def z_hat_and_var_hat_est(self, z_inp, v_bar_half, fixed_var=True):  # note v_bar is total noise i.e. both real and imag chan noise combined variance
        
        v_bar = 2*v_bar_half[0,:,:,:,:]
        
        z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0:self.SamplingRate,:,:,:], 1*z_inp[:,self.SamplingRate:,:,:,:]], dim=-1).contiguous())

        z_angle_inp = torch.angle(z_complex)
        z_abs_inp = torch.abs(z_complex)

        z_foo_1 = torch.view_as_real(torch.polar(self.y_inp, z_angle_inp))
        
        z_foo = torch.cat([z_foo_1[:,:,:,:,:,0], z_foo_1[:,:,:,:,:,1]], dim=1).contiguous()

        z_hat = ((v_bar)/(v_bar + 2*self.v))*(z_foo) + ((2*self.v)/(v_bar + 2*self.v))*(z_inp)

        if fixed_var:
            var_hat_half = 1*v_bar_half

            return z_hat, var_hat_half
        
        else:

            v_hat_real_axis = ((self.v)*v_bar[:self.SamplingRate,:,:,:])/(v_bar[:self.SamplingRate,:,:,:] + 2*self.v)

            v_hat_imag_axis = (v_bar[:self.SamplingRate,:,:,:]/(2*z_abs_inp))*(((v_bar[:self.SamplingRate,:,:,:])/(v_bar[:self.SamplingRate,:,:,:] + 2*self.v))*(self.y_inp) + ((2*self.v)/(v_bar[:self.SamplingRate,:,:,:] + 2*self.v))*(z_abs_inp))
           
            v_hat_full = v_hat_real_axis + v_hat_imag_axis
            
            mask_pos_ext = (z_abs_inp>(self.y_inp/2))[0,:,:,:,:]

            v_hat = torch.sum(v_hat_full*mask_pos_ext)/(torch.sum(mask_pos_ext) + 1e-18)
            var_hat_half = (v_hat/2)*torch.ones_like(v_bar_half)
            
            return z_hat, var_hat_half
        
        

class Deep_FFHQ_denoiser_deepECpr_with_divergence_and_EM:
    
    def __init__(self, diff_model, sampler, device, L_trials = 10, EM_total_iters = 10):
        
        self.device = device
        self.denoiser = diff_model
        self.L_trials = L_trials
        self.EM_total_iters = EM_total_iters
        self.sampler = sampler

    def get_diff_time_step_from_0_255_std(self,given_std_0_255, coef1_list):
        given_std_0_1 = given_std_0_255/255
        given_std_diff = given_std_0_1/0.5 # required since diffusion model images are in range -1 to 1
        est_coeff1 = np.sqrt(1/(1+(given_std_diff**2)))
        # find the index of coef1_list that is closest to est_coeff1
        coef1_list_np = np.array(coef1_list)
        idx = (np.abs(coef1_list_np - est_coeff1)).argmin()

        return idx, est_coeff1
    
    def denoise(self,x, noise_est_std):
        input_to_denoiser_new = x.to(self.device)
        with torch.no_grad():
            t_val, est_coeff1 = self.get_diff_time_step_from_0_255_std(noise_est_std.numpy(), self.sampler.sqrt_alphas_cumprod)
            coef1 = extract_and_expand(self.sampler.sqrt_alphas_cumprod, t_val, input_to_denoiser_new/255)
            my_noisy_image = ((input_to_denoiser_new/255) - 0.5)*coef1/0.5
            batched_times = torch.full((1,), t_val, device = self.device, dtype = torch.long)
            pred_noise = self.denoiser(my_noisy_image, batched_times)[:,:3,:,:]
            x_denoised = extract_and_expand(self.sampler.sqrt_recip_alphas_cumprod, batched_times, my_noisy_image) * my_noisy_image - extract_and_expand(self.sampler.sqrt_recipm1_alphas_cumprod, batched_times, my_noisy_image) * pred_noise

            denoised_image = 255*((x_denoised.detach().cpu())*0.5+ 0.5)

        return denoised_image.cpu()
    
    def denoise_and_compute_inp_op_variance_using_EM(self, x_noisy, approx_inp_var=100):
        noise_est_std_assumption = torch.sqrt(approx_inp_var)
        N_foo = torch.numel(x_noisy)
        eta = torch.max(torch.abs(x_noisy))/((10**6)) + 2e-16
        den = N_foo*eta
        divergence_accum = 0
        x_denoised = self.denoise(x_noisy,noise_est_std_assumption)

        for ell in range(self.L_trials):
            torch.random.manual_seed(ell)
            pm_vec = (2*(torch.rand_like(x_noisy)<0.5)-1)
            x_noisy_2 = x_noisy + eta*pm_vec
            x_denoised_2 = self.denoise(x_noisy_2,noise_est_std_assumption)
            num = torch.sum(pm_vec*(x_denoised_2 - 1*x_denoised))
            divergence_accum += num/den
        divergence = divergence_accum/self.L_trials

        pred_input_var = approx_inp_var
        mse_foo = torch.sum((x_denoised - x_noisy)**2)/(N_foo)

        for em_iter in range(self.EM_total_iters):
            
            nabla_inv = divergence*pred_input_var

            pred_input_var = mse_foo + nabla_inv

        pred_out_var = divergence*pred_input_var

        return x_denoised, pred_input_var, pred_out_var, divergence

class prox_PnP_freq_deepECpr_with_EM:
    
    def __init__(self, denoiser, A_op_CDP_OSF, clamp_and_pass_imag = False):
        self.denoiser = denoiser
        self.A_op_CDP_OSF = A_op_CDP_OSF
        self.clamp_and_pass_imag = clamp_and_pass_imag

    def prox(self, z_noisy, approx_inp_var, perform_EM = True):
        
        x_noisy = self.A_op_CDP_OSF.H(z_noisy)
        
        x_out = torch.zeros_like(x_noisy)

        x_noisy_real = 1*x_noisy[:,0,:,:,:]
        
        pred_inp_var_x = 0
        pred_op_var_x = 0
        divergence_x = 0

        if perform_EM:
            denoised_image, pred_inp_var_x, pred_op_var_x, divergence_x = self.denoiser.denoise_and_compute_inp_op_variance_using_EM(x_noisy_real, approx_inp_var)
        else:
            noise_est_std_assumption = torch.sqrt(approx_inp_var)
            denoised_image = self.denoiser.denoise(x_noisy_real, noise_est_std_assumption)


        x_out_real = denoised_image

        x_out[:,0,:,:,:] = x_out_real

        if self.clamp_and_pass_imag:
            x_out = torch.clamp(x_out, min=0)

        return self.A_op_CDP_OSF.A(x_out), pred_inp_var_x, pred_op_var_x, divergence_x
    


#############################################
######### Grayscale Image Functions #########
#############################################

class MAP_chan_stage_deepECpr_CDP_grayscale:
    
    def __init__(self, y, sig):
        self.y_inp = 1*y
        self.SamplingRate = y.shape[1]
        self.sig = sig
        self.v = self.sig**2

    def z_hat_and_var_hat_est(self, z_inp, v_bar_half, fixed_var=True):  # note v_bar is total noise i.e. both real and imag chan noise combined variance
        
        v_bar = 2*v_bar_half[0,:,:,:]
        
        z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0:self.SamplingRate,:,:], 1*z_inp[:,self.SamplingRate:,:,:]], dim=-1).contiguous())

        z_angle_inp = torch.angle(z_complex)
        z_abs_inp = torch.abs(z_complex)

        z_foo_1 = torch.view_as_real(torch.polar(self.y_inp, z_angle_inp))
        
        z_foo = torch.cat([z_foo_1[:,:,:,:,0], z_foo_1[:,:,:,:,1]], dim=1).contiguous()

        z_hat = ((v_bar)/(v_bar + 2*self.v))*(z_foo) + ((2*self.v)/(v_bar + 2*self.v))*(z_inp)

        if fixed_var:
            var_hat_half = 1*v_bar_half

            return z_hat, var_hat_half
        
        else:

            v_hat_real_axis = ((self.v)*v_bar[:self.SamplingRate,:,:])/(v_bar[:self.SamplingRate,:,:] + 2*self.v)

            v_hat_imag_axis = (v_bar[:self.SamplingRate,:,:]/(2*z_abs_inp))*(((v_bar[:self.SamplingRate,:,:])/(v_bar[:self.SamplingRate,:,:] + 2*self.v))*(self.y_inp) + ((2*self.v)/(v_bar[:self.SamplingRate,:,:] + 2*self.v))*(z_abs_inp))
           
            v_hat_full = v_hat_real_axis + v_hat_imag_axis
            
            mask_pos_ext = (z_abs_inp>(self.y_inp/2))[0,:,:,:]

            v_hat = torch.sum(v_hat_full*mask_pos_ext)/(torch.sum(mask_pos_ext) + 1e-18)
            var_hat_half = (v_hat/2)*torch.ones_like(v_bar_half)

            return z_hat, var_hat_half
        

class prox_PnP_freq_deepECpr_with_EM_grayscale_CDP:
    
    def __init__(self, denoiser, A_op_CDP, clamp_and_pass_imag = False):
        self.denoiser = denoiser
        self.A_op_CDP = A_op_CDP
        self.clamp_and_pass_imag = clamp_and_pass_imag

    def prox(self, z_noisy_scaled, approx_inp_var, perform_EM = True):
        
        x_noisy = self.A_op_CDP.H(z_noisy_scaled)
      
        x_out = torch.zeros_like(x_noisy)

        x_noisy_real = 1*x_noisy[:,[0],:,:]
        
        pred_inp_var_x = 0
        pred_op_var_x = 0
        divergence_x = 0
        
        if perform_EM:
            denoised_image, pred_inp_var_x, pred_op_var_x, divergence_x = self.denoiser.denoise_and_compute_inp_op_variance_using_EM(x_noisy_real, approx_inp_var)
        else:
            denoised_image = self.denoiser.denoise(x_noisy_real)

        x_out_real = denoised_image

        x_out[:,[0],:,:] = x_out_real

        if self.clamp_and_pass_imag:
            x_out = torch.clamp(x_out, min=0)

        return self.A_op_CDP.A(x_out), pred_inp_var_x, pred_op_var_x, divergence_x
    

class MAP_chan_stage_deepECpr_OSF_grayscale:
    
    def __init__(self, y, sig):
        self.y_inp = 1*y.squeeze(1)
        self.oversampled_image_size = y.shape[-1]
        self.sig = sig
        self.v = self.sig**2


    def z_hat_and_var_hat_est(self, z_inp, v_bar_half, fixed_var=True):  # note v_bar is total noise i.e. both real and imag chan noise combined variance
        
        v_bar = 2*v_bar_half[0,:,:,:]
        
        z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0,:,:], 1*z_inp[:,1,:,:]], dim=-1).contiguous())

        z_angle_inp = torch.angle(z_complex)
        z_abs_inp = torch.abs(z_complex)

        z_foo = torch.view_as_real(torch.polar(self.y_inp, z_angle_inp)).permute(0,3,1,2).contiguous()

        z_hat = ((v_bar)/(v_bar + 2*self.v))*(z_foo) + ((2*self.v)/(v_bar + 2*self.v))*(z_inp)

        if fixed_var:
            var_hat_half = 1*v_bar_half
            return z_hat, var_hat_half
        
        else:

            v_hat_real_axis = ((self.v)*v_bar)/(v_bar + 2*self.v)
            v_hat_imag_axis = (v_bar/(2*z_abs_inp))*(((v_bar)/(v_bar + 2*self.v))*(self.y_inp) + ((2*self.v)/(v_bar + 2*self.v))*(z_abs_inp))

            v_hat_full = v_hat_real_axis + v_hat_imag_axis
            mask_pos_ext = (z_abs_inp>(self.y_inp/2))[0,:,:]

            v_hat = torch.sum(v_hat_full*mask_pos_ext)/(torch.sum(mask_pos_ext) + 1e-18)
            var_hat_half = (v_hat/2)*torch.ones_like(v_bar_half)

            return z_hat, var_hat_half
        

class Deep_denoiser_deepECpr_with_divergence_and_EM:
    
    def __init__(self, model_path, device, L_trials = 10, EM_total_iters = 10):
        
        dnCNN_L2_state_dict = torch.load(model_path, map_location=device)
        dnCNN_L2 = DnCNN_BF(image_channels = 1, out_channel = 1)
        dnCNN_L2.load_state_dict(dnCNN_L2_state_dict["model"][0])
        dnCNN_L2.to(device)
        dnCNN_L2.eval()
        self.device = device
        self.denoiser = dnCNN_L2
        self.L_trials = L_trials
        self.EM_total_iters = EM_total_iters

    def denoise(self,x):
        with torch.no_grad():
            denoised_image = 255*(self.denoiser(x.to(self.device)/255))
        return denoised_image.cpu()
    
    def denoise_and_compute_inp_op_variance_using_EM(self, x_noisy, approx_inp_var=100):
        N_foo = torch.numel(x_noisy)
        eta = torch.max(torch.abs(x_noisy))/((10**4)) + 2e-16
        den = N_foo*eta
        divergence_accum = 0
        x_denoised = self.denoise(x_noisy)

        for ell in range(self.L_trials):
            torch.random.manual_seed(ell)
            pm_vec = (2*(torch.rand_like(x_noisy)<0.5)-1)
            x_noisy_2 = x_noisy + eta*pm_vec
            x_denoised_2 = self.denoise(x_noisy_2)
            num = torch.sum(pm_vec*(x_denoised_2 - 1*x_denoised))
            divergence_accum += num/den
        divergence = divergence_accum/self.L_trials

        pred_input_var = approx_inp_var
        mse_foo = torch.sum((x_denoised - x_noisy)**2)/(N_foo)

        for em_iter in range(self.EM_total_iters):
            
            nabla_inv = divergence*pred_input_var

            pred_input_var = mse_foo + nabla_inv

        pred_out_var = divergence*pred_input_var

        return x_denoised, pred_input_var, pred_out_var, divergence
    

class prox_PnP_freq_deepECpr_with_EM_grayscale:
    
    def __init__(self, denoiser, lambda_tune, Omn_foo, A_op, clamp_and_pass_imag = False):
        self.denoiser = denoiser
        self.Omn_foo = Omn_foo
        self.lambda_tune = lambda_tune 
        self.A_op = A_op
        self.clamp_and_pass_imag = clamp_and_pass_imag

    def prox(self, z_noisy_scaled, approx_inp_var, perform_EM = True):
        
        x_noisy_scaled = self.A_op.H(z_noisy_scaled)

        x_noisy = self.Omn_foo.H(x_noisy_scaled)
        
        x_out = torch.zeros_like(x_noisy)

        x_noisy_real = 1*x_noisy[:,[0],:,:]
        
        pred_inp_var_x = 0
        pred_op_var_x = 0
        divergence_x = 0
        
        if perform_EM:
            denoised_image, pred_inp_var_x, pred_op_var_x, divergence_x = self.denoiser.denoise_and_compute_inp_op_variance_using_EM(x_noisy_real, approx_inp_var)
        else:
            denoised_image = self.denoiser.denoise(x_noisy_real)

        x_out_real = denoised_image

        x_out[:,[0],:,:] = x_out_real

        if self.clamp_and_pass_imag:
            # x_out = torch.clamp(x_out, min=0, max = 255)
            x_out = torch.clamp(x_out, min=0)
            # x_out[:,[1],:,:] = x_noisy[:,[1],:,:]/(1+self.lambda_tune)

        return self.A_op.A(self.Omn_foo.A(x_out)), pred_inp_var_x, pred_op_var_x, divergence_x
    



