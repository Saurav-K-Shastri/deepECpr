This repository contains the code associated with the paper "[Fast and Robust Phase Retrieval via Deep Expectation-Consistent Approximation](https://arxiv.org/pdf/2407.09687)," by Saurav K. Shastri and Philip Schniter.

### Abstract

Accurately recovering images from phaseless measurements is a challenging and long-standing problem.  In this work, we present "deepECpr," which combines expectation-consistent (EC) approximation with deep denoising networks to surpass state-of-the-art phase-retrieval methods in both speed and accuracy.  In addition to applying EC in a non-traditional manner, deepECpr includes a novel stochastic damping scheme that is inspired by recent diffusion methods.  Like existing phase-retrieval methods based on plug-and-play priors, regularization by denoising, or diffusion, deepECpr iterates a denoising stage with a measurement-exploitation stage.  But unlike existing methods, deepECpr requires far fewer denoiser calls.  We compare deepECpr to the state-of-the-art prDeep (Metzler et al., 2018), Deep-ITA (Wang et al., 2020), and Diffusion Posterior Sampling (Chung et al., 2023) methods for noisy phase-retrieval of color, natural, and unnatural grayscale images on oversampled-Fourier (OSF) and coded-diffraction-pattern (CDP) measurements and find improvements in both PSNR and SSIM with 5x fewer denoiser calls. 

### Dependencies

Please download the data and the pre-trained denoisers required for the demo here: https://drive.google.com/drive/folders/1XZOdsoeFcgCZiCaHom7hcylPK-qRhnpv?usp=sharing

To make sure that the versions of the packages match those we tested the code with, we recommend creating a new virtual environment and installing packages using `conda` with the following command.

```bash
conda env create -f environment.yml
```

### Demo

The Jupyter Notebook "Example_OSF_FFHQ_Recovery_Demo.ipynb" demonstrates colored image recovery from noisy, phaseless OSF measurements. Before running the notebook, ensure you update the file paths according to where you've saved the downloaded files.

To obtain the average performance metrics, execute the corresponding commands for each experiment after updating the file paths in the relevant configuration files located in the 'config/' directory:

1. OSF Phase Retrieval of FFHQ Images
```
python3 run_FFHQ_deepECpr_all_test_images_all_noise_levels.py --task_config=configs/FFHQ_OSF_config.yaml 
```

2. CDP Phase Retrieval of FFHQ Images
```
python3 run_FFHQ_deepECpr_all_test_images_all_noise_levels.py --task_config=configs/FFHQ_CDP_config.yaml
```

3. CDP Phase Retrieval of Grayscale Images
```
python3 run_Grayscale_deepECpr_all_test_images_all_noise_levels.py --task_config=configs/Grayscale_CDP_config.yaml
```

4. OSF Phase Retrieval of Grayscale Images
```
python3 run_Grayscale_deepECpr_all_test_images_all_noise_levels.py --task_config=configs/Grayscale_OSF_config.yaml 
```

### Acknowledgments
- [prDeep](https://github.com/ricedsp/prDeep/tree/master)
- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling)
