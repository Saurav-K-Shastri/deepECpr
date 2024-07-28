# Runs deepECpr on all FFHQ test images for all noise levels
# Saurav (shastri.19@osu.edu)

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import argparse
import numpy as np
from utils_deepECpr.algo_utils import *
from algorithms.deepECpr_algo import *
from utils_deepECpr.measurement_utils import *
from evaluation_scripts.metrics import ssim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, default='configs/FFHQ_OSF_config.yaml')
    parser.add_argument('--data_config', type=str, default='configs/testdata_config.yaml')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    my_device = torch.device(device_str)  

    task_config = load_yaml(args.task_config)

    data_config_location = args.data_config
    model_config_location = args.model_config
    diffusion_config_location = args.diffusion_config
    

    measurement_type = task_config.get('measurement_type')
    damp_bar1 = task_config.get('damp_bar1')
    damp_bar2 = task_config.get('damp_bar2')
    EM_iteration_stop = task_config.get('EM_iteration_stop')
    total_iterations = task_config.get('total_iterations')
    linear_tune = task_config.get('linear_tune')

    std_input = 120/np.sqrt(2) # this is the standard deviation of the initialization noise # ./np.sqrt(2) becasue the input is complex

    if measurement_type == "OSF":
        pre_run_HIO_file_location = task_config.get('pre_run_HIO_file_path')
        HIO_dict = np.load(pre_run_HIO_file_location)
        HIO_recon_channel_corrected_using_correlation_combined = HIO_dict['HIO_recon_channel_corrected_using_correlation_combined'] # this has been flipped and rotated to align with channels and not with ground truth
        alpha_list = [4,6,8]
        num_alg_runs = 3
    else:
        alpha_list = [5,15,45]
        num_alg_runs = 1

    PSNR_all_images = np.zeros((30,3))
    SSIM_all_images = np.zeros((30,3))

    # save PSNR and SSIM in args.save_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # add measurement type to the save file name
    save_file = os.path.join(save_dir, f"PSNR_SSIM_{measurement_type}_all_FFHQ_test_images_all_noise_levels.npz")

    print("=====================================================================================")
    print("==============Running deepECpr on all test images and noise levels===================")
    print("=====================================================================================")
    print(" ")
    print("Measurement Type  : ", measurement_type)
    for alpha in alpha_list:
        print(" ")
        print("alpha: ", alpha)
        if measurement_type == "OSF":
            alpha_count = {4: 0, 6: 1}.get(alpha, 2)
        else:
            alpha_count = {5: 0, 15: 1}.get(alpha, 2)

        for image_number in range(30):

            print("Image number: ", image_number)

            best_alg_run_error = np.inf

            for alg_run in range(num_alg_runs):

                y, sig, x0, z0, CDP_rand_mat = get_FFHQ_measurement(image_number, measurement_type, alpha, data_config_location, verbose = False)

                if measurement_type == "OSF":
                    x_hat_init = torch.zeros_like(x0)
                    x_hat_init[0,0,:,:,:] = torch.from_numpy(HIO_recon_channel_corrected_using_correlation_combined[image_number,alg_run,alpha_count,:,:,:])
                else:
                    x_hat_init = torch.zeros_like(x0) # dummy # actual initialization for CDP case is done side the 'run_deepECpr' function

                # Run deepECpr  

                deepECpr_recon, residual_error = run_deepECpr(y, sig, x0, z0, model_config_location, diffusion_config_location, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune, x_hat_init = x_hat_init, fixed_CDP_rand_mat=CDP_rand_mat, measurement_type = measurement_type , verbose = False, my_device=my_device)

                if residual_error < best_alg_run_error:
                    best_alg_run_error = residual_error
                    best_alg_run_recon = deepECpr_recon

            recon_img = best_alg_run_recon[0,0,:,:,:].permute(1,2,0).contiguous().cpu().numpy()/255
            GT_img = x0[0,0,:,:,:].permute(1,2,0).contiguous().cpu().numpy()/255
            
            SSIM_all_images[image_number,alpha_count] = ssim(GT_img, recon_img, maxval = 1.0, multichannel = True)
            PSNR_all_images[image_number,alpha_count] = compute_psnr(GT_img, recon_img)

            np.savez(save_file, PSNR_all_images = PSNR_all_images, SSIM_all_images = SSIM_all_images)

    print(" ")    
    print("=====================================================================")
    print("==============Average PSNR and SSIM for all test images==============")
    print("=====================================================================")
    print(" ")
    for alpha in alpha_list:
        if measurement_type == "OSF":
            alpha_count = {4: 0, 6: 1}.get(alpha, 2)
        else:
            alpha_count = {5: 0, 15: 1}.get(alpha, 2)
        print(f"Alpha: {alpha}")
        print("Avg PSNR: ", np.round(np.mean(PSNR_all_images[:,alpha_count]),2))
        print("Avg SSIM: ", np.round(np.mean(SSIM_all_images[:,alpha_count]),4))
        print(" ")
    print("=====================================================================")
    print("=====================================================================")


if __name__ == '__main__':
    main()