# Runs HIO and saves reconstructions on all FFHQ test images for all noise levels and all runs
# This code is for the FFHQ dataset. Please follow a similar approach for the Grayscale dataset.
# Saurav (shastri.19@osu.edu)

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import argparse
import numpy as np
from utils_deepECpr.algo_utils import *
from algorithms.deepECpr_algo import *
from algorithms.HIO_algo import *
from utils_deepECpr.measurement_utils import *
from evaluation_scripts.metrics import ssim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='configs/testdata_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_results/HIO_pre_run/')
    args = parser.parse_args()

    # Device setting

    data_config_location = args.data_config
    
    measurement_type = "OSF"

    if measurement_type == "OSF":
        alpha_list = [4,6,8]
        num_alg_runs = 3
    else:
        raise NotImplementedError


    # save PSNR and SSIM in args.save_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # add measurement type to the save file name
    save_file = os.path.join(save_dir, f"FFHQ_OSF_HIO_pre_run.npz")

    print("=====================================================================================")
    print("=============== Running HIO on all test images and noise levels =====================")
    print("=====================================================================================")
    print(" ")
    print("Measurement Type  : ", measurement_type)

    HIO_recon_channel_corrected_using_correlation_combined = torch.zeros((30,3,3,3,256,256))

    for alpha in alpha_list:
        print(" ")
        print("alpha: ", alpha)
        if measurement_type == "OSF":
            alpha_count = {4: 0, 6: 1}.get(alpha, 2)
        else:
            raise NotImplementedError

        for image_number in range(30):

            print("Image number: ", image_number)

            for alg_run in range(num_alg_runs):

                y, sig, x0, z0, CDP_rand_mat = get_FFHQ_measurement(image_number, measurement_type, alpha, data_config_location, verbose = False)

                # Run HIO  

                HIO_recon = run_HIO(y, true_alg_run = alg_run, measurement_type = measurement_type)

                HIO_recon_channel_corrected_using_correlation_combined[image_number,alg_run,alpha_count,:,:,:] = HIO_recon[0,0,:,:,:]

            np.savez(save_file, HIO_recon_channel_corrected_using_correlation_combined = HIO_recon_channel_corrected_using_correlation_combined)

    print("=====================================================================")
    print("=====================================================================")


if __name__ == '__main__':
    main()