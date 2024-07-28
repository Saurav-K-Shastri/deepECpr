# Runs deepECpr on all Grayscale test images for all noise levels
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
    parser.add_argument('--task_config', type=str, default='configs/Grayscale_CDP_config.yaml')
    parser.add_argument('--data_config', type=str, default='configs/testdata_config_grayscale.yaml')
    parser.add_argument('--model_config', type=str, default='configs/model_config_grayscale.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    my_device = torch.device(device_str)  

    task_config = load_yaml(args.task_config)

    data_config_location = args.data_config
    model_config_location = args.model_config
   
    measurement_type = task_config.get('measurement_type')
    damp_bar1 = task_config.get('damp_bar1')
    damp_bar2 = task_config.get('damp_bar2')
    EM_iteration_stop = task_config.get('EM_iteration_stop')
    total_iterations = task_config.get('total_iterations')
    linear_tune_list = task_config.get('linear_tune_list')

    std_input = 70/np.sqrt(2) # this is the standard deviation of the initialization noise # ./np.sqrt(2) becasue the input is complex

    if measurement_type == "OSF":
        pre_run_natural_HIO_file_path = task_config.get('pre_run_natural_HIO_file_path')
        HIO_dict_natural = np.load(pre_run_natural_HIO_file_path)
        HIO_alg_natural_init = HIO_dict_natural["HIO_alg_init"]

        pre_run_unnatural_HIO_file_path = task_config.get('pre_run_unnatural_HIO_file_path')
        HIO_dict_unnatural = np.load(pre_run_unnatural_HIO_file_path)
        HIO_alg_unnatural_init = HIO_dict_unnatural["HIO_alg_init"]
        
        alpha_list = [4,6,8]
        num_alg_runs = 3
    else:
        alpha_list = [5,15,45]
        num_alg_runs = 1

    #################################################
    ########## Run for natural test images ##########
    #################################################

    image_type = "natural"

    PSNR_all_images = np.zeros((6,3))
    SSIM_all_images = np.zeros((6,3))

    # save PSNR and SSIM in args.save_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # add measurement type to the save file name
    save_file_natural = os.path.join(save_dir, f"PSNR_SSIM_{measurement_type}_all_Grayscale_natural_test_images_all_noise_levels.npz")

    print("=============================================================================================")
    print("==============Running deepECpr on all natural test images and noise levels===================")
    print("=============================================================================================")
    print(" ")
    print("Measurement Type  : ", measurement_type)
    for alpha in alpha_list:
        print(" ")
        print("alpha: ", alpha)
        if measurement_type == "OSF":
            alpha_count = {4: 0, 6: 1}.get(alpha, 2)
        else:
            alpha_count = {5: 0, 15: 1}.get(alpha, 2)

        for image_number in range(6):

            print("Image number: ", image_number)

            best_alg_run_error = np.inf

            for alg_run in range(num_alg_runs):

                y, sig, x0, z0, CDP_rand_mat = get_grayscale_measurement(image_number, image_type, measurement_type, alpha, data_config_location, verbose = False)

                if measurement_type == "OSF":
                    x_hat_init = torch.zeros_like(x0)
                    x_hat_init[0,0,:,:] = torch.from_numpy(HIO_alg_natural_init[image_number,alg_run,alpha_count,:,:])
                    deepECpr_recon, residual_error = run_deepECpr_grayscale_OSF(y, sig, x0, z0, model_config_location, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = x_hat_init, fixed_CDP_rand_mat=CDP_rand_mat, measurement_type = measurement_type , verbose = False, my_device=my_device)
                elif measurement_type == "CDP":
                    x_hat_init = torch.zeros_like(x0) # dummy # actual initialization for CDP case is done side the 'run_deepECpr' function
                    deepECpr_recon, residual_error = run_deepECpr_grayscale_CDP(y, sig, x0, z0, model_config_location, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = x_hat_init, fixed_CDP_rand_mat=CDP_rand_mat, measurement_type = measurement_type , verbose = False, my_device=my_device)
                else:
                    raise ValueError("Invalid measurement type")
                # Run deepECpr  

                if residual_error < best_alg_run_error:
                    best_alg_run_error = residual_error
                    best_alg_run_recon = deepECpr_recon

            recon_img = best_alg_run_recon[0,0,:,:].contiguous().cpu().numpy()/255
            GT_img = x0[0,0,:,:].contiguous().cpu().numpy()/255
            
            SSIM_all_images[image_number,alpha_count] = ssim(GT_img, recon_img, maxval = 1.0)
            PSNR_all_images[image_number,alpha_count] = compute_psnr(GT_img, recon_img)

            np.savez(save_file_natural, PSNR_all_images = PSNR_all_images, SSIM_all_images = SSIM_all_images)

    print(" ")    
    print("=====================================================================")
    print("==============Average PSNR and SSIM for all test images==============")
    print("=====================================================================")
    print(" ")
    print(" Image type: ", image_type)
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

    #################################################
    ########## Run for natural test images ##########
    #################################################

    image_type = "unnatural"

    PSNR_all_images = np.zeros((6,3))
    SSIM_all_images = np.zeros((6,3))

    # save PSNR and SSIM in args.save_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # add measurement type to the save file name
    save_file_unnatural = os.path.join(save_dir, f"PSNR_SSIM_{measurement_type}_all_Grayscale_unnatural_test_images_all_noise_levels.npz")

    print("===============================================================================================")
    print("==============Running deepECpr on all unnatural test images and noise levels===================")
    print("===============================================================================================")
    print(" ")
    print("Measurement Type  : ", measurement_type)
    for alpha in alpha_list:
        print(" ")
        print("alpha: ", alpha)
        if measurement_type == "OSF":
            alpha_count = {4: 0, 6: 1}.get(alpha, 2)
        else:
            alpha_count = {5: 0, 15: 1}.get(alpha, 2)

        for image_number in range(6):

            print("Image number: ", image_number)

            best_alg_run_error = np.inf

            for alg_run in range(num_alg_runs):

                y, sig, x0, z0, CDP_rand_mat = get_grayscale_measurement(image_number, image_type, measurement_type, alpha, data_config_location, verbose = False)

                if measurement_type == "OSF":
                    x_hat_init = torch.zeros_like(x0)
                    x_hat_init[0,0,:,:] = torch.from_numpy(HIO_alg_unnatural_init[image_number,alg_run,alpha_count,:,:])
                    deepECpr_recon, residual_error = run_deepECpr_grayscale_OSF(y, sig, x0, z0, model_config_location, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = x_hat_init, fixed_CDP_rand_mat=CDP_rand_mat, measurement_type = measurement_type , verbose = False, my_device=my_device)
                elif measurement_type == "CDP":
                    x_hat_init = torch.zeros_like(x0) # dummy # actual initialization for CDP case is done side the 'run_deepECpr' function
                    deepECpr_recon, residual_error = run_deepECpr_grayscale_CDP(y, sig, x0, z0, model_config_location, damp_bar1, damp_bar2, EM_iteration_stop, total_iterations, std_input, linear_tune_list, x_hat_init = x_hat_init, fixed_CDP_rand_mat=CDP_rand_mat, measurement_type = measurement_type , verbose = False, my_device=my_device)
                else:
                    raise ValueError("Invalid measurement type")
                # Run deepECpr  

                if residual_error < best_alg_run_error:
                    best_alg_run_error = residual_error
                    best_alg_run_recon = deepECpr_recon

            recon_img = best_alg_run_recon[0,0,:,:].contiguous().cpu().numpy()/255
            GT_img = x0[0,0,:,:].contiguous().cpu().numpy()/255
            
            SSIM_all_images[image_number,alpha_count] = ssim(GT_img, recon_img, maxval = 1.0)
            PSNR_all_images[image_number,alpha_count] = compute_psnr(GT_img, recon_img)

            np.savez(save_file_unnatural, PSNR_all_images = PSNR_all_images, SSIM_all_images = SSIM_all_images)

    print(" ")    
    print("=====================================================================")
    print("==============Average PSNR and SSIM for all test images==============")
    print("=====================================================================")
    print(" ")
    print(" Image type: ", image_type)
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