import torch
import numpy as np
import pydicom
import time
import os
import argparse
from pathlib import Path
from torchvision import utils
from scipy.io import savemat
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from torch_radon import RadonFanbeam

from model import Unet, FBPConvNet
from diffusion import GaussianDiffusion, get_named_eta_schedule, ModelMeanType, LossType

size = 512
angles = torch.linspace(0, 2 * torch.pi, 64)
resolution = 512
radon = RadonFanbeam(
    resolution,
    angles,
    source_distance=590.0,
    det_distance=490.0,
    det_count=736,
    det_spacing=2.1,
    clip_to_circle=False
)


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x


def compare(recon0, recon1, verbose=True):
    mse_recon = mean_squared_error(recon0, recon1)

    small_side = np.min(recon0.shape)
    if small_side < 7:
        if small_side % 2:
            win_size = small_side
        else:
            win_size = small_side - 1
    else:
        win_size = None

    ssim_recon = ssim1(recon0, recon1,
                       data_range=recon0.max() - recon0.min(), win_size=win_size)

    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                                         data_range=recon0.max() - recon0.min())

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)


def display_images(images, titles=None, figsize=(12, 6), save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = clear(image)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


def calculate_difference_image(image1, image2, amplification_factor=1.0):
    diff = np.abs(image1 - image2) * amplification_factor
    return diff


def test_single_image(image_path, results_folder, model_path, high_freq_model_path=None, timesteps=450):
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet(
        dim=64,
        input_channels=1,
        hf_channels=1,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
    )
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
        print(f"Loaded EMA model weights from {model_path}")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model weights from {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {model_path}")

    model_high = None
    if high_freq_model_path is not None:
        model_high = FBPConvNet().to(device)
        model_high.load_state_dict(torch.load(high_freq_model_path, map_location=device))
        model_high.eval()
        print(f"High frequency model loaded: {high_freq_model_path}")

    min_noise_level = 0.02
    etas_end = 0.99
    kappa = 0.001
    power = 0.3

    sqrt_etas = get_named_eta_schedule(
        'exponential',
        timesteps,
        min_noise_level,
        etas_end=etas_end,
        kappa=kappa,
        kwargs={'power': power}
    )

    diffusion_model = GaussianDiffusion(
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=ModelMeanType.START_X,
        loss_type=LossType.MSE,
        sf=1,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True
    )

    start_time = time.time()
    print(f"Processing image: {image_path}")

    dicom_image = pydicom.dcmread(image_path)
    image = dicom_image.pixel_array.astype(np.float32)
    image[image > 2500] = 0
    image = image / 2500.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    sinogram = radon.forward(image)
    filtered_sinogram = radon.filter_sinogram(sinogram)
    fbp_recon = radon.backward(filtered_sinogram)

    print("Input images:")
    display_images(
        [image, fbp_recon],
        titles=["Original Image", "FBP Reconstruction"],
        save_path=str(results_folder / "input_images.png")
    )

    model.eval()
    with torch.no_grad():
        print("Running diffusion sampling...")
        test_samples = diffusion_model.p_sample_loop(
            x_0=image,
            y=fbp_recon,
            model=model,
            model_high=model_high,
            noise=torch.randn_like(image),
            device=device,
            progress=True,
            regularize=True
        )

    end_time = time.time()
    processing_time = end_time - start_time

    test_samples_normalized = clear(test_samples)
    image_normalized = clear(image)
    fbp_normalized = clear(fbp_recon)

    diff_image = calculate_difference_image(image_normalized, test_samples_normalized, amplification_factor=5.0)

    print("Compared to original image:")
    mse, ssim, psnr = compare(test_samples_normalized, image_normalized, verbose=True)

    print("Compared to FBP reconstruction:")
    fbp_mse, fbp_ssim, fbp_psnr = compare(fbp_normalized, image_normalized, verbose=True)

    print(f"FBP - MSE: {fbp_mse:.4f}, SSIM: {fbp_ssim:.4f}, PSNR: {fbp_psnr:.4f}")
    print(f"Diffusion - MSE: {mse:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}")
    print(f"Improvement - PSNR: {psnr - fbp_psnr:.2f} dB, SSIM: {ssim - fbp_ssim:.4f}")
    print(f"Processing time: {processing_time:.4f} seconds")

    print("\nFinal reconstruction result:")
    display_images(
        [image_normalized, fbp_normalized, test_samples_normalized, diff_image],
        titles=["Original Image", "FBP Reconstruction", "Diffusion Model", "Difference x5"],
        figsize=(20, 5),
        save_path=str(results_folder / "final_comparison.png")
    )

    file_name = os.path.basename(image_path)
    save_prefix = os.path.join(results_folder, file_name)

    utils.save_image(test_samples, save_prefix + '_diffusion.png', nrow=1)
    utils.save_image(fbp_recon, save_prefix + '_fbp.png', nrow=1)
    utils.save_image(image, save_prefix + '_original.png', nrow=1)

    plt.imsave(save_prefix + '_diff.png', diff_image, cmap='jet')

    savemat(save_prefix + '_original.mat', {'recon': image_normalized})
    savemat(save_prefix + '_diffusion.mat', {'recon': test_samples_normalized})
    savemat(save_prefix + '_fbp.mat', {'recon': fbp_normalized})
    savemat(save_prefix + '_diff.mat', {'recon': diff_image})

    print(f"Results saved to: {results_folder}")

    return {
        'mse': mse,
        'ssim': ssim,
        'psnr': psnr,
        'fbp_mse': fbp_mse,
        'fbp_ssim': fbp_ssim,
        'fbp_psnr': fbp_psnr,
        'processing_time': processing_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single IMA image diffusion model test')
    parser.add_argument('--image_path', type=str, required=True, help='Test IMA image path')
    parser.add_argument('--results_folder', type=str, default='./results-test', help='Results save folder')
    parser.add_argument('--model_path', type=str, required=True, help='Trained model path')
    parser.add_argument('--high_freq_model_path', type=str, default=None, help='High frequency model path')
    parser.add_argument('--timesteps', type=int, default=100, help='Diffusion timesteps')

    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    print(f"Using parameters:")
    print(f"  Image path: {args.image_path}")
    print(f"  Results folder: {args.results_folder}")
    print(f"  Model path: {args.model_path}")
    print(f"  High frequency model path: {args.high_freq_model_path}")
    print(f"  Diffusion timesteps: {args.timesteps}")

    test_single_image(
        image_path=args.image_path,
        results_folder=args.results_folder,
        model_path=args.model_path,
        high_freq_model_path=args.high_freq_model_path,
        timesteps=args.timesteps
    )
