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


def run_deepECpr(y, sig, x0, z0, model_config_name, diffusion_config_name, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune, x_hat_init = None, fixed_CDP_rand_mat=None, measurement_type = 'OSF', verbose = False, my_device = 'cuda:0'):
    
    # Load diffusion denoiser model
    model_config = load_yaml(model_config_name)
    diffusion_config = load_yaml(diffusion_config_name)
    my_device = my_device
    diff_model = create_model(**model_config)
    diff_model = diff_model.to(my_device)
    diff_model.eval()
    sampler = create_sampler(**diffusion_config) 

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
        A_op_main_foo = A_op_OSF(p,imsize)
        A_op_F_foo = A_op_Fourier(p,imsize)

        # zero padding masks
        x_dummy = torch.ones(1,2,3,resize_size,resize_size)
        ones_zero_padded  = torch.nn.functional.pad(x_dummy, (A_op_main_foo.pad_left, A_op_main_foo.pad_right, A_op_main_foo.pad_left, A_op_main_foo.pad_right))
        zero_padded_mask = torch.ones_like(ones_zero_padded) - ones_zero_padded
        oversampled_image_size = int(np.sqrt(m))       
        support = torch.zeros(1,1,3,oversampled_image_size,oversampled_image_size)
        pad = (oversampled_image_size - imsize)
        pad_left = pad//2 # same amount at top
        pad_right = pad - pad_left # same amount at bottom
        support[:,:,:,pad_left:pad_left+imsize,pad_left:pad_left+imsize] = 1
        x_hat_init = x_hat_init

    elif measurement_type == 'CDP':

        A_op_main_foo = A_op_CDP(fixed_CDP_rand_mat)
        A_op_complex_foo = A_op_complex_HIO_CDP(fixed_CDP_rand_mat)

        y_complex = torch.view_as_complex(torch.stack([1*y, 0*y], dim=-1)).contiguous()
        x_dummy = torch.rand(A_op_complex_foo.H(1*y_complex).shape) + 0*1j
        x_init_complex = A_op_complex_foo.H(1*y_complex * torch.exp(1j * torch.angle(A_op_complex_foo.A(x_dummy))))

        x_hat_init = torch.zeros_like(x0)
        x_hat_init[:,0,:,:,:] = torch.clamp(torch.real(x_init_complex)[:,0,:,:,:],0,255)
    
    else:

        raise ValueError("Measurement type should be either 'OSF' or 'CDP'")
        
    use_EM_estimated_vbar2 = True
    

    # Intialization for deepECpr
    z_bar2 = A_op_main_foo.A(x_hat_init) + std_input*torch.randn_like(z0) # very noisy initial guess

    if measurement_type == 'OSF':
        v_bar2 = 1.2*(std_input**2)*torch.ones(1,1,1,2*resize_size,2*resize_size) # variance initialization
    else:
        v_bar2 = 1.2*(std_input**2)*torch.ones(1,2*SamplingRate,3,resize_size,resize_size) # variance initialization
 

    v_bar2_old = 1*v_bar2
    z_bar2_old = 1*z_bar2

    if measurement_type == 'OSF':
        chan_stage_posterior_est = MAP_chan_stage_deepECpr_OSF(y, sig)
    else:
        chan_stage_posterior_est = MAP_chan_stage_deepECpr_CDP(y, sig)

    
    my_Denoiser = Deep_FFHQ_denoiser_deepECpr_with_divergence_and_EM(diff_model, sampler, my_device, L_trials = 10, EM_total_iters = 10)

    for alg_iter in range(total_iterations):

        ############################################################################################
        #################################### Stochastic Damping ####################################
        ############################################################################################
        # Note damping doesnot happen at first iteraton

        v_bar2_damped = (damp_bar2*torch.sqrt(torch.abs(v_bar2)) + (1 - damp_bar2)*torch.sqrt(v_bar2_old))**2 # Note damping doesnot happen at first iteraton since v_bar2_old = v_bar2
        std_of_noise_to_add = torch.sqrt(torch.abs(v_bar2_damped - v_bar2))

        v_bar2 = 1*v_bar2_damped
        z_bar2 = 1*z_bar2 + std_of_noise_to_add*torch.randn_like(z_bar2) 

        v_bar2_old = 1*v_bar2


        ####################################################################################################
        #################################### zhat2 and vhat2 estimation ####################################
        ####################################################################################################
        #Since certain regions of the signal is known to be zero, we estimate the true noise variance in those regions and use it to improve the accuracy of our noise variance estimates

        if measurement_type == 'OSF':

            x_bar2 = A_op_main_foo.H(1*z_bar2)
            x_bar2_oversampled = A_op_F_foo.H(1*z_bar2)
            x_bar2_zero_padded_region = x_bar2_oversampled*zero_padded_mask

            x_bar_2_non_support_noise_var_real = torch.sum((x_bar2_zero_padded_region[0,0,:,:,:])**2)/torch.sum(zero_padded_mask[0,0,:,:,:])  # we know signal is zero in real part in non-support region
            x_bar_2_non_support_noise_var_imag = torch.sum((x_bar2_zero_padded_region[0,1,:,:,:])**2)/torch.sum(zero_padded_mask[0,1,:,:,:])  # we know signal is zero in imaginary part in non-support region

            x_bar_2_support_noise_var_imag = torch.mean(((x_bar2)[0,1,:,:,:])**2) # we know signal is zero in imaginary part in support region
            
            image_domain_noise_var = 4*((2*v_bar2) - (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag) - (1/4)*x_bar_2_support_noise_var_imag) # support region is 1/4th of the total region; non-support region is 3/4th of the total region

        else:

            col_space_z_bar2 = A_op_main_foo.A(A_op_main_foo.H(z_bar2))
            null_space_z_bar2 = z_bar2 - col_space_z_bar2
            z_vbar2_null_space_noise_var_real = torch.mean((null_space_z_bar2[0,0:SamplingRate,:,:,:])**2)/(3/4) # bcs the null space covers only 3/4th of the total space so we need to account this in the division (i.e. the correct the mean computation)
            z_vbar2_null_space_noise_var_imag = torch.mean((null_space_z_bar2[0,SamplingRate:,:,:,:])**2)/(3/4)
            x_bar2 = A_op_main_foo.H(1*z_bar2)
            x_bar_2_support_noise_var_imag = torch.mean(x_bar2[0,1,:,:,:]**2)
            image_domain_noise_var = 4*((2*v_bar2) - (3/4)*(z_vbar2_null_space_noise_var_real + z_vbar2_null_space_noise_var_imag) - (1/4)*x_bar_2_support_noise_var_imag)



        my_prox_foo = prox_PnP_freq_deepECpr_with_EM(my_Denoiser, A_op_main_foo, clamp_and_pass_imag = True)

        if use_EM_estimated_vbar2 and alg_iter<EM_iteration_stop:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = True)
            if measurement_type == 'OSF':
                v_bar2 = 0.5*torch.ones_like(v_bar2)*((1/4)*(pred_inp_var_x + x_bar_2_support_noise_var_imag) + (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag))
            else:
                v_bar2 = 0.5*torch.ones_like(v_bar2)*((1/4)*(pred_inp_var_x + x_bar_2_support_noise_var_imag) + (3/4)*(z_vbar2_null_space_noise_var_real + z_vbar2_null_space_noise_var_imag))

            if alg_iter==0:
                v_bar2_old = 1*v_bar2
        else:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = False)
        
        v_hat2 = linear_tune*v_bar2

        if verbose:
            x_rec_deepECpr = A_op_main_foo.H(1*z_hat2)
            print("Algorithm Iteration "+str(alg_iter)+"; PSNR deepECpr : ", compute_psnr(x0[0,0,:,:,:].numpy()/255, fix_flip(x_rec_deepECpr, x0)[0,0,:,:,:].numpy()/255))

        ######################################################################################################
        ##################################### vbar1 and zbar1 computation ####################################
        ######################################################################################################

        if alg_iter>0:
            v_bar1_old = 1*v_bar1
            z_bar1_old = 1*z_bar1

        v_bar1 = 1/((1/v_hat2) - (1/v_bar2))
        z_bar1 = ((z_hat2/v_hat2) - (z_bar2/v_bar2))*v_bar1

        #################################################################################
        #################################### Damping ####################################
        #################################################################################
        # Note damping doesnot happen at first iteraton

        if alg_iter>0:
            z_bar1 = damp_bar1 * z_bar1 + (1 - damp_bar1) * z_bar1_old
            v_bar1 = (damp_bar1*torch.sqrt(torch.abs(v_bar1)) + (1 - damp_bar1)*torch.sqrt(v_bar1_old))**2

        ####################################################################################################
        #################################### zhat1 and vhat1 estimation ####################################
        ####################################################################################################

        z_hat1, v_hat1 = chan_stage_posterior_est.z_hat_and_var_hat_est(z_bar1, v_bar1, fixed_var=False)

        ######################################################################################################
        ##################################### vbar2 and zbar2 computation ####################################
        ######################################################################################################

        v_bar2 = 1/((1/v_hat1) - (1/v_bar1))
        z_bar2 = ((z_hat1/v_hat1) - (z_bar1/v_bar1))*v_bar2

        ##################################### end of one iteration ####################################
      
    # Residual error
    trial_recon = torch.zeros(1,2,3,resize_size,resize_size)
    trial_recon[0,0,:,:,:] = (A_op_main_foo.H(1*z_hat2))[0,0,:,:,:]
    trial_recon = torch.clamp(trial_recon,0,255)
    trial_z0 = A_op_main_foo.A(trial_recon)
    trial_z0_complex = torch.stack((trial_z0[:,[0],:,:,:],trial_z0[:,[1],:,:,:]), dim = -1)
    trial_y2_full = complex_abs(trial_z0_complex)**2
    trial_y2 = torch.abs(trial_y2_full)
    trial_y = torch.abs(torch.sqrt(trial_y2))
    error_trial = torch.mean(torch.abs(trial_y[0,0,:,:,:] - y[0,0,:,:,:])**2).cpu().numpy()

    x_rec_deepECpr = A_op_main_foo.H(1*z_hat2)
    x_rec_deepECpr_corrected = fix_flip(x_rec_deepECpr, x0)
    x_rec_deepECpr = x_rec_deepECpr_corrected

    return x_rec_deepECpr, error_trial


def run_deepECpr_OSF_for_demo(y, sig, x0, z0, model_config_name, diffusion_config_name, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune, x_hat_HIO_init, verbose = True, my_device = 'cuda:0'):
    
    # Load diffusion denoiser model
    model_config = load_yaml(model_config_name)
    diffusion_config = load_yaml(diffusion_config_name)
    my_device = my_device
    diff_model = create_model(**model_config)
    diff_model = diff_model.to(my_device)
    diff_model.eval()
    sampler = create_sampler(**diffusion_config) 

    # lists for tracking quantities
    v_bar1_evolution = []
    v_bar2_evolution = []
    v_hat1_evolution = []
    v_hat2_evolution = []
    true_v_bar1_evolution = []
    true_v_bar2_evolution = []
    true_v_hat1_evolution = []
    true_v_hat2_evolution = []
    added_noise_std_evolution = []
    PSNR_list = []

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    imsize = 256
    
    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize

    A_op_main_foo = A_op_OSF(p,imsize)
    A_op_F_foo = A_op_Fourier(p,imsize)


    # zero padding masks
    x_dummy = torch.ones(1,2,3,resize_size,resize_size)
    ones_zero_padded  = torch.nn.functional.pad(x_dummy, (A_op_main_foo.pad_left, A_op_main_foo.pad_right, A_op_main_foo.pad_left, A_op_main_foo.pad_right))
    zero_padded_mask = torch.ones_like(ones_zero_padded) - ones_zero_padded
    oversampled_image_size = int(np.sqrt(m))       
    support = torch.zeros(1,1,3,oversampled_image_size,oversampled_image_size)
    pad = (oversampled_image_size - imsize)
    pad_left = pad//2 # same amount at top
    pad_right = pad - pad_left # same amount at bottom
    support[:,:,:,pad_left:pad_left+imsize,pad_left:pad_left+imsize] = 1

    use_EM_estimated_vbar2 = True

    # Intialization for deepECpr
    z_bar2 = A_op_main_foo.A(x_hat_HIO_init) + std_input*torch.randn_like(z0) # very noisy initial guess
    v_bar2 = 1.2*(std_input**2)*torch.ones(1,1,1,2*resize_size,2*resize_size) # variance initialization

    v_bar2_old = 1*v_bar2
    z_bar2_old = 1*z_bar2

    chan_stage_posterior_est = MAP_chan_stage_deepECpr_OSF(y, sig)
    my_Denoiser = Deep_FFHQ_denoiser_deepECpr_with_divergence_and_EM(diff_model, sampler, my_device, L_trials = 10, EM_total_iters = 10)

    for alg_iter in range(total_iterations):

        ############################################################################################
        #################################### Stochastic Damping ####################################
        ############################################################################################
        # Note damping doesnot happen at first iteraton

        v_bar2_damped = (damp_bar2*torch.sqrt(torch.abs(v_bar2)) + (1 - damp_bar2)*torch.sqrt(v_bar2_old))**2 # Note damping doesnot happen at first iteraton since v_bar2_old = v_bar2
        std_of_noise_to_add = torch.sqrt(torch.abs(v_bar2_damped - v_bar2))

        v_bar2 = 1*v_bar2_damped
        z_bar2 = 1*z_bar2 + std_of_noise_to_add*torch.randn_like(z_bar2) 

        v_bar2_old = 1*v_bar2

        # Track quantities
        if alg_iter>0:
            added_noise_std_evolution.append(torch.mean(std_of_noise_to_add))
        elif alg_iter==0:
            added_noise_std_evolution.append(torch.mean(torch.tensor(std_input)))

        ####################################################################################################
        #################################### zhat2 and vhat2 estimation ####################################
        ####################################################################################################
        #Since certain regions of the signal is known to be zero, we estimate the true noise variance in those regions and use it to improve the accuracy of our noise variance estimates

        x_bar2 = A_op_main_foo.H(1*z_bar2)
        x_bar2_oversampled = A_op_F_foo.H(1*z_bar2)
        x_bar2_zero_padded_region = x_bar2_oversampled*zero_padded_mask

        x_bar_2_non_support_noise_var_real = torch.sum((x_bar2_zero_padded_region[0,0,:,:,:])**2)/torch.sum(zero_padded_mask[0,0,:,:,:])  # we know signal is zero in real part in non-support region
        x_bar_2_non_support_noise_var_imag = torch.sum((x_bar2_zero_padded_region[0,1,:,:,:])**2)/torch.sum(zero_padded_mask[0,1,:,:,:])  # we know signal is zero in imaginary part in non-support region

        x_bar_2_support_noise_var_imag = torch.mean(((x_bar2)[0,1,:,:,:])**2) # we know signal is zero in imaginary part in support region
        
        image_domain_noise_var = 4*((2*v_bar2) - (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag) - (1/4)*x_bar_2_support_noise_var_imag) # support region is 1/4th of the total region; non-support region is 3/4th of the total region

        my_prox_foo = prox_PnP_freq_deepECpr_with_EM(my_Denoiser, A_op_main_foo, clamp_and_pass_imag = True)

        if use_EM_estimated_vbar2 and alg_iter<EM_iteration_stop:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = True)
            v_bar2 = 0.5*torch.ones_like(v_bar2)*((1/4)*(pred_inp_var_x + x_bar_2_support_noise_var_imag) + (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag))
            if alg_iter==0:
                v_bar2_old = 1*v_bar2
        else:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = False)
        
        v_hat2 = linear_tune*v_bar2

        # Track quantities
        v_bar2_evolution.append(torch.mean(v_bar2))
        true_v_bar2_evolution.append(torch.mean((z_bar2 - z0)**2))
        v_hat2_evolution.append(torch.mean(v_hat2))
        true_v_hat2_evolution.append(torch.mean((z_hat2 - z0)**2))
        x_rec_deepECpr = A_op_main_foo.H(1*z_hat2)
        PSNR_list.append(compute_psnr(x0[0,0,:,:,:].numpy()/255, fix_flip(x_rec_deepECpr, x0)[0,0,:,:,:].numpy()/255))
        if verbose:
            print("Algorithm Iteration "+str(alg_iter)+"; PSNR deepECpr : ", compute_psnr(x0[0,0,:,:,:].numpy()/255, fix_flip(x_rec_deepECpr, x0)[0,0,:,:,:].numpy()/255))



        ######################################################################################################
        ##################################### vbar1 and zbar1 computation ####################################
        ######################################################################################################

        if alg_iter>0:
            v_bar1_old = 1*v_bar1
            z_bar1_old = 1*z_bar1

        v_bar1 = 1/((1/v_hat2) - (1/v_bar2))
        z_bar1 = ((z_hat2/v_hat2) - (z_bar2/v_bar2))*v_bar1

        #################################################################################
        #################################### Damping ####################################
        #################################################################################
        # Note damping doesnot happen at first iteraton

        if alg_iter>0:
            z_bar1 = damp_bar1 * z_bar1 + (1 - damp_bar1) * z_bar1_old
            v_bar1 = (damp_bar1*torch.sqrt(torch.abs(v_bar1)) + (1 - damp_bar1)*torch.sqrt(v_bar1_old))**2

        # Track quantities
        v_bar1_evolution.append(torch.mean(v_bar1))
        true_v_bar1_evolution.append(torch.mean((z_bar1 - z0)**2))


        ####################################################################################################
        #################################### zhat1 and vhat1 estimation ####################################
        ####################################################################################################

        z_hat1, v_hat1 = chan_stage_posterior_est.z_hat_and_var_hat_est(z_bar1, v_bar1, fixed_var=False)

        # Track quantities
        v_hat1_evolution.append(torch.mean(v_hat1))
        true_v_hat1_evolution.append(torch.mean((z_hat1 - z0)**2))


        ######################################################################################################
        ##################################### vbar2 and zbar2 computation ####################################
        ######################################################################################################

        v_bar2 = 1/((1/v_hat1) - (1/v_bar1))
        z_bar2 = ((z_hat1/v_hat1) - (z_bar1/v_bar1))*v_bar2

        ##################################### end of one iteration ####################################

    # Track quantities
        
    x_rec_deepECpr_corrected = fix_flip(x_rec_deepECpr, x0)
    x_rec_deepECpr = x_rec_deepECpr_corrected

    print("-------------------------------------------")
    print("FINAL PSNR deepECpr : ", compute_psnr(x0[0,0,:,:,:].numpy()/255, x_rec_deepECpr[0,0,:,:,:].numpy()/255))
    print("-------------------------------------------")


    #############################################################################################
    ##################################### deepECpr Recovery  ####################################
    #############################################################################################

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(x0[0,0,:,:,:].permute(1,2,0).numpy()/255)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth', fontsize=16)
    plt.subplot(1,2,2)
    x_rec_deepECpr = torch.clamp(x_rec_deepECpr, min=0, max=255)
    plt.imshow(x_rec_deepECpr[0,0,:,:,:].permute(1,2,0).numpy()/255)
    plt.xticks([])
    plt.yticks([])
    plt.title('deepECpr Recon; PSNR: ' + str(np.round(compute_psnr(x0[0,0,:,:,:].numpy()/255, x_rec_deepECpr[0,0,:,:,:].numpy()/255),2)), fontsize=16)
    plt.show()

    ########################################################################################################################
    ##################################### Evolution of the true and estimated z errors  ####################################
    ########################################################################################################################
    
    plt.figure()
    plt.plot(np.sqrt(2*np.array(v_bar2_evolution)), 'b--o', label = r'estimated SD of error in $\overline{\mathbf{z}}^{(2)}$', linewidth = 1, markersize = 4)
    plt.plot(np.sqrt(2*np.array(true_v_bar2_evolution)), 'b--', label = r'true SD of error in $\overline{\mathbf{z}}^{(2)}$', linewidth = 2, markersize = 3)
    plt.plot(np.sqrt(2*np.array(v_bar1_evolution)), 'c--o', label = r'estimated SD of error in $\overline{\mathbf{z}}^{(1)}$', linewidth = 1, markersize = 4)
    plt.plot(np.sqrt(2*np.array(true_v_bar1_evolution)), 'c--', label = r'true SD of error in $\overline{\mathbf{z}}^{(1)}$', linewidth = 2, markersize = 3)
    plt.plot(np.sqrt(2*np.array(v_hat2_evolution)), 'r--o', label = r'estimated SD of error in $\widehat{\mathbf{z}}^{(2)}$', linewidth = 1, markersize = 4)
    plt.plot(np.sqrt(2*np.array(true_v_hat2_evolution)), 'r--', label = r'true SD of error in $\widehat{\mathbf{z}}^{(2)}$', linewidth = 2, markersize = 3)
    plt.plot(np.sqrt(2*np.array(v_hat1_evolution)), 'm--o', label = r'estimated SD of error in $\widehat{\mathbf{z}}^{(1)}$', linewidth = 1, markersize = 4)
    plt.plot(np.sqrt(2*np.array(true_v_hat1_evolution)), 'm--', label = r'true SD of error in $\widehat{\mathbf{z}}^{(1)}$', linewidth = 2, markersize = 3)
    plt.yscale('log')
    plt.legend( fontsize = 11)
    plt.xlabel('Iteration', fontsize = 16)
    plt.ylabel("Standard Deviation", fontsize = 16)
    plt.title("Evolution of the true and estimated z errors", fontsize = 16)
    plt.grid() 
    plt.show()

    ########################################################################################################################################################
    ##################################### Standard deviation of the noise added by deepECpr’s stochastic damping scheme ####################################
    ########################################################################################################################################################

    plt.figure()
    plt.plot(np.sqrt(2*np.array(added_noise_std_evolution)**2))
    plt.title("Standard deviation of the noise added", fontsize = 16)
    plt.xlabel("Iteration", fontsize = 16)
    plt.ylabel("Standard Deviation", fontsize = 16)
    plt.grid()
    plt.show()


    return x_rec_deepECpr


def run_deepECpr_grayscale_CDP(y, sig, x0, z0, model_config_name, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = None, fixed_CDP_rand_mat=None, measurement_type = 'CDP', verbose = False, my_device = 'cuda:0'):
    
    # Load diffusion denoiser model
    model_config = load_yaml(model_config_name)
    model_path_foo = model_config['model_path']

    BF_DnCNN_40_60_std_path = model_path_foo + 'BF_DnCNN_denoiser_40_60/checkpoint_last.pt'
    BF_DnCNN_20_40_std_path = model_path_foo + 'BF_DnCNN_denoiser_20_40/checkpoint_last.pt'
    BF_DnCNN_10_20_std_path = model_path_foo + 'BF_DnCNN_denoiser_10_20/checkpoint_last.pt'
    BF_DnCNN_0_10_std_path = model_path_foo + 'BF_DnCNN_denoiser_0_10/checkpoint_last.pt'

    model_paths = []
    model_paths.append(BF_DnCNN_40_60_std_path)
    model_paths.append(BF_DnCNN_20_40_std_path)
    model_paths.append(BF_DnCNN_10_20_std_path)
    model_paths.append(BF_DnCNN_0_10_std_path)
    
    my_device = my_device

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    imsize = 128
    
    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize
    SamplingRate = p

    if measurement_type == 'CDP':

        A_op_main_foo = A_op_CDP_grayscale(fixed_CDP_rand_mat)
        A_op_complex_foo = A_op_complex_CDP_HIO_grayscale(fixed_CDP_rand_mat)

        y_complex = torch.view_as_complex(torch.stack([1*y, 0*y], dim=-1)).contiguous()
        x_dummy = torch.rand(A_op_complex_foo.H(y_complex).shape) + 0*1j
        x_init_complex = A_op_complex_foo.H(y_complex * torch.exp(1j * torch.angle(A_op_complex_foo.A(x_dummy))))

        x_hat_init = torch.zeros_like(x0)
        x_hat_init[:,0,:,:] = torch.clamp(torch.real(x_init_complex)[:,0,:,:],0,255)

    else:

        raise ValueError("Measurement type should be 'CDP'")
        
    use_EM_estimated_vbar2 = True
    
    # Intialization for deepECpr
    z_bar2 = A_op_main_foo.A(x_hat_init) + std_input*torch.randn_like(z0) # very noisy initial guess
    v_bar2 = 1.2*(std_input**2)*torch.ones(1,2*SamplingRate,resize_size,resize_size) # variance initialization

    v_bar2_old = 1*v_bar2
    z_bar2_old = 1*z_bar2

    chan_stage_posterior_est = MAP_chan_stage_deepECpr_CDP_grayscale(y, sig)

    for alg_iter in range(total_iterations):

        ############################################################################################
        #################################### Stochastic Damping ####################################
        ############################################################################################
        # Note damping doesnot happen at first iteraton

        v_bar2_damped = (damp_bar2*torch.sqrt(torch.abs(v_bar2)) + (1 - damp_bar2)*torch.sqrt(v_bar2_old))**2 # Note damping doesnot happen at first iteraton since v_bar2_old = v_bar2
        std_of_noise_to_add = torch.sqrt(torch.abs(v_bar2_damped - v_bar2))

        v_bar2 = 1*v_bar2_damped
        z_bar2 = 1*z_bar2 + std_of_noise_to_add*torch.randn_like(z_bar2) 

        v_bar2_old = 1*v_bar2


        ####################################################################################################
        #################################### zhat2 and vhat2 estimation ####################################
        ####################################################################################################
        #Since certain regions of the signal is known to be zero, we estimate the true noise variance in those regions and use it to improve the accuracy of our noise variance estimates


        col_space_z_bar2 = A_op_main_foo.A(A_op_main_foo.H(z_bar2))
        null_space_z_bar2 = z_bar2 - col_space_z_bar2
        z_vbar2_null_space_noise_var_real = torch.mean((null_space_z_bar2[0,0:SamplingRate,:,:])**2)/(3/4) # bcs the null space covers only 3/4th of the total space so we need to account this in the division (i.e. the correct the mean computation)
        z_vbar2_null_space_noise_var_imag = torch.mean((null_space_z_bar2[0,SamplingRate:,:,:])**2)/(3/4)
        x_bar2 = A_op_main_foo.H(1*z_bar2)
        x_bar_2_support_noise_var_imag = torch.mean(((x_bar2)[0,1,:,:])**2)
        image_domain_noise_var = 4*((2*v_bar2) - (3/4)*(z_vbar2_null_space_noise_var_real + z_vbar2_null_space_noise_var_imag) - (1/4)*x_bar_2_support_noise_var_imag)
        std_bar2_in_image_dom = torch.sqrt(torch.abs(torch.mean(image_domain_noise_var)))


        if torch.max(std_bar2_in_image_dom).item()>40:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[0], my_device)
            linear_tune = linear_tune_list[0]
        elif torch.max(std_bar2_in_image_dom).item()>20:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[1], my_device)
            linear_tune = linear_tune_list[1]
        elif torch.max(std_bar2_in_image_dom).item()>10:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[2], my_device)
            linear_tune = linear_tune_list[2]
        else:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[3], my_device)
            linear_tune = linear_tune_list[3]

        my_prox_foo = prox_PnP_freq_deepECpr_with_EM_grayscale_CDP(my_Denoiser, A_op_main_foo, clamp_and_pass_imag = True)

        if use_EM_estimated_vbar2 and alg_iter<EM_iteration_stop:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = True)
            v_bar2 = 0.5*torch.ones_like(v_bar2)*((1/4)*(pred_inp_var_x + x_bar_2_support_noise_var_imag) + (3/4)*(z_vbar2_null_space_noise_var_real + z_vbar2_null_space_noise_var_imag))
            if alg_iter==0:
                v_bar2_old = 1*v_bar2
        else:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = False)
        
        v_hat2 = linear_tune*v_bar2

        ######################################################################################################
        ##################################### vbar1 and zbar1 computation ####################################
        ######################################################################################################

        if alg_iter>0:
            v_bar1_old = 1*v_bar1
            z_bar1_old = 1*z_bar1

        v_bar1 = 1/((1/v_hat2) - (1/v_bar2))
        z_bar1 = ((z_hat2/v_hat2) - (z_bar2/v_bar2))*v_bar1

        #################################################################################
        #################################### Damping ####################################
        #################################################################################
        # Note damping doesnot happen at first iteraton

        if alg_iter>0:
            z_bar1 = damp_bar1 * z_bar1 + (1 - damp_bar1) * z_bar1_old
            v_bar1 = (damp_bar1*torch.sqrt(torch.abs(v_bar1)) + (1 - damp_bar1)*torch.sqrt(v_bar1_old))**2

        ####################################################################################################
        #################################### zhat1 and vhat1 estimation ####################################
        ####################################################################################################

        z_hat1, v_hat1 = chan_stage_posterior_est.z_hat_and_var_hat_est(z_bar1, v_bar1, fixed_var=False)

        ######################################################################################################
        ##################################### vbar2 and zbar2 computation ####################################
        ######################################################################################################

        v_bar2 = 1/((1/v_hat1) - (1/v_bar1))
        z_bar2 = ((z_hat1/v_hat1) - (z_bar1/v_bar1))*v_bar2

        ##################################### end of one iteration ####################################
      
    # Residual error
    trial_recon = torch.zeros(1,2,resize_size,resize_size)
    trial_recon[0,0,:,:] = (A_op_main_foo.H(1*z_hat2))[0,0,:,:]
    trial_recon = torch.clamp(trial_recon,0,255)
    trial_z0 = A_op_main_foo.A(trial_recon)
    trial_z0_complex = torch.stack((trial_z0[:,[0],:,:],trial_z0[:,[1],:,:]), dim = -1)
    trial_y2_full = complex_abs(trial_z0_complex)**2
    trial_y2 = torch.abs(trial_y2_full)
    trial_y = torch.abs(torch.sqrt(trial_y2))
    error_trial = torch.mean(torch.abs(trial_y[0,0,:,:] - y[0,0,:,:])**2).cpu().numpy()

    x_rec_deepECpr = A_op_main_foo.H(1*z_hat2)

    return x_rec_deepECpr, error_trial



def run_deepECpr_grayscale_OSF(y, sig, x0, z0, model_config_name, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = None, fixed_CDP_rand_mat=None, measurement_type = 'OSF', verbose = False, my_device = 'cuda:0'):
    
    # Load diffusion denoiser model
    model_config = load_yaml(model_config_name)
    model_path_foo = model_config['model_path']

    BF_DnCNN_40_60_std_path = model_path_foo + 'BF_DnCNN_denoiser_40_60/checkpoint_last.pt'
    BF_DnCNN_20_40_std_path = model_path_foo + 'BF_DnCNN_denoiser_20_40/checkpoint_last.pt'
    BF_DnCNN_10_20_std_path = model_path_foo + 'BF_DnCNN_denoiser_10_20/checkpoint_last.pt'
    BF_DnCNN_0_10_std_path = model_path_foo + 'BF_DnCNN_denoiser_0_10/checkpoint_last.pt'

    model_paths = []
    model_paths.append(BF_DnCNN_40_60_std_path)
    model_paths.append(BF_DnCNN_20_40_std_path)
    model_paths.append(BF_DnCNN_10_20_std_path)
    model_paths.append(BF_DnCNN_0_10_std_path)
    
    my_device = my_device

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    imsize = 128
    
    d = imsize*imsize
    p = 4
    m = p*d
    resize_size = imsize
    SamplingRate = p

    if measurement_type == 'OSF':

        Omn_func_foo = Omn_func_new(p,resize_size)
        Omn_func_zero_pad = Omn_func_new(p,resize_size)
        ones_zero_padded = Omn_func_zero_pad.A(torch.ones(1,2,resize_size,resize_size))
        zero_padded_mask = torch.ones_like(ones_zero_padded) - ones_zero_padded

        A_op_foo = A_op_Fourier_grayscale(p,resize_size)
        A_op_main_foo = A_op_OSF_grayscale(p,resize_size,Omn_func_foo)

        # zero padding masks
        oversampled_image_size = int(np.sqrt(m))       
        support = torch.zeros(1,1,oversampled_image_size,oversampled_image_size)
        pad = (oversampled_image_size - resize_size)
        pad_left = pad//2 # same amount at top
        pad_right = pad - pad_left # same amount at bottom
        support[:,:,pad_left:pad_left+resize_size,pad_left:pad_left+resize_size] = 1

        x_hat_init = x_hat_init

    else:

        raise ValueError("Measurement type should be 'OSF'")
        
    use_EM_estimated_vbar2 = True
    
    # Intialization for deepECpr
    z_bar2 = A_op_foo.A(Omn_func_foo.A(x_hat_init)) + std_input*torch.randn_like(z0) # very noisy initial guess

    v_bar2 = 1.2*(std_input**2)*torch.ones(1,1,2*resize_size,2*resize_size) # variance initialization


    v_bar2_old = 1*v_bar2
    z_bar2_old = 1*z_bar2

    chan_stage_posterior_est = MAP_chan_stage_deepECpr_OSF_grayscale(y, sig)

    for alg_iter in range(total_iterations):

        ############################################################################################
        #################################### Stochastic Damping ####################################
        ############################################################################################
        # Note damping doesnot happen at first iteraton

        v_bar2_damped = (damp_bar2*torch.sqrt(torch.abs(v_bar2)) + (1 - damp_bar2)*torch.sqrt(v_bar2_old))**2 # Note damping doesnot happen at first iteraton since v_bar2_old = v_bar2
        std_of_noise_to_add = torch.sqrt(torch.abs(v_bar2_damped - v_bar2))

        v_bar2 = 1*v_bar2_damped
        z_bar2 = 1*z_bar2 + std_of_noise_to_add*torch.randn_like(z_bar2) 

        v_bar2_old = 1*v_bar2


        ####################################################################################################
        #################################### zhat2 and vhat2 estimation ####################################
        ####################################################################################################
        #Since certain regions of the signal is known to be zero, we estimate the true noise variance in those regions and use it to improve the accuracy of our noise variance estimates

        x_bar2 = A_op_main_foo.H(1*z_bar2)
        x_bar2_oversampled = A_op_foo.H(1*z_bar2)
        x_bar2_zero_padded_region = x_bar2_oversampled*zero_padded_mask

        x_bar_2_non_support_noise_var_real = torch.sum((x_bar2_zero_padded_region[0,0,:,:])**2)/torch.sum(zero_padded_mask[0,0,:,:])  # we know signal is zero in real part in non-support region
        x_bar_2_non_support_noise_var_imag = torch.sum((x_bar2_zero_padded_region[0,1,:,:])**2)/torch.sum(zero_padded_mask[0,1,:,:])  # we know signal is zero in imaginary part in non-support region

        x_bar_2_support_noise_var_imag = torch.mean(((x_bar2)[0,1,:,:])**2) # we know signal is zero in imaginary part in support region
        
        image_domain_noise_var = 4*((2*v_bar2) - (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag) - (1/4)*x_bar_2_support_noise_var_imag) # support region is 1/4th of the total region; non-support region is 3/4th of the total region


        if alg_iter<150:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[0], my_device)
            linear_tune = linear_tune_list[0]
        elif alg_iter<250:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[1], my_device)
            linear_tune = linear_tune_list[1]
        elif alg_iter<325:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[2], my_device)
            linear_tune = linear_tune_list[2]
        else:
            my_Denoiser = Deep_denoiser_deepECpr_with_divergence_and_EM(model_paths[3], my_device)
            linear_tune = linear_tune_list[3]

        my_prox_foo = prox_PnP_freq_deepECpr_with_EM_grayscale(my_Denoiser, 1, Omn_func_foo, A_op_foo, clamp_and_pass_imag = True)

        if use_EM_estimated_vbar2 and alg_iter<EM_iteration_stop:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = True)
            v_bar2 = 0.5*torch.ones_like(v_bar2)*((1/4)*(pred_inp_var_x + x_bar_2_support_noise_var_imag) + (3/4)*(x_bar_2_non_support_noise_var_real + x_bar_2_non_support_noise_var_imag))
            if alg_iter==0:
                v_bar2_old = 1*v_bar2
        else:
            z_hat2, pred_inp_var_x, pred_op_var_x, denoiser_divergence = my_prox_foo.prox(1*z_bar2, torch.abs(torch.mean(image_domain_noise_var)), perform_EM = False)
        
        v_hat2 = linear_tune*v_bar2

        ######################################################################################################
        ##################################### vbar1 and zbar1 computation ####################################
        ######################################################################################################

        if alg_iter>0:
            v_bar1_old = 1*v_bar1
            z_bar1_old = 1*z_bar1

        v_bar1 = 1/((1/v_hat2) - (1/v_bar2))
        z_bar1 = ((z_hat2/v_hat2) - (z_bar2/v_bar2))*v_bar1

        #################################################################################
        #################################### Damping ####################################
        #################################################################################
        # Note damping doesnot happen at first iteraton

        if alg_iter>0:
            z_bar1 = damp_bar1 * z_bar1 + (1 - damp_bar1) * z_bar1_old
            v_bar1 = (damp_bar1*torch.sqrt(torch.abs(v_bar1)) + (1 - damp_bar1)*torch.sqrt(v_bar1_old))**2

        ####################################################################################################
        #################################### zhat1 and vhat1 estimation ####################################
        ####################################################################################################

        z_hat1, v_hat1 = chan_stage_posterior_est.z_hat_and_var_hat_est(z_bar1, v_bar1, fixed_var=False)

        ######################################################################################################
        ##################################### vbar2 and zbar2 computation ####################################
        ######################################################################################################

        v_bar2 = 1/((1/v_hat1) - (1/v_bar1))
        z_bar2 = ((z_hat1/v_hat1) - (z_bar1/v_bar1))*v_bar2

        ##################################### end of one iteration ####################################
      
    # Residual error
    trial_recon = torch.zeros(1,2,resize_size,resize_size)
    trial_recon[0,0,:,:] = (A_op_main_foo.H(1*z_hat2))[0,0,:,:]
    trial_recon = torch.clamp(trial_recon,0,255)
    trial_z0 = A_op_main_foo.A(trial_recon)
    trial_z0_complex = torch.stack((trial_z0[:,[0],:,:],trial_z0[:,[1],:,:]), dim = -1)
    trial_y2_full = complex_abs(trial_z0_complex)**2
    trial_y2 = torch.abs(trial_y2_full)
    trial_y = torch.abs(torch.sqrt(trial_y2))
    error_trial = torch.mean(torch.abs(trial_y[0,0,:,:] - y[0,0,:,:])**2).cpu().numpy()


    x_rec_deepECpr = Omn_func_foo.H(A_op_foo.H(1*z_hat2))
    (row_shift, column_shift, flip_op) = find_best_shift(x_rec_deepECpr[0,0,:,:], x0[0,0,:,:])
    if flip_op:
        x_rec_deepECpr_corrected = 1*x_rec_deepECpr[0,0,:,:].flip(0).flip(1)
    else:
        x_rec_deepECpr_corrected = 1*x_rec_deepECpr[0,0,:,:]
    x_rec_deepECpr_corrected = (torch.roll(x_rec_deepECpr_corrected, shifts=(row_shift, column_shift), dims=(0, 1))).unsqueeze(0).unsqueeze(0).contiguous()
    x_rec_deepECpr = x_rec_deepECpr_corrected

    return x_rec_deepECpr, error_trial

