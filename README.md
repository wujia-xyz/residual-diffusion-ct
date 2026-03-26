# Residual Diffusion for Sparse-View CT Reconstruction

## Abstract

Sparse-view computed tomography (CT) reduces radiation exposure by acquiring fewer projections, making it valuable for dose-sensitive imaging scenarios. However, this often results in severe streak artifacts and detail loss due to incomplete angular sampling. While diffusion models provide effective generative priors for reconstruction, conventional approaches require extensive sampling steps from Gaussian noise, resulting in computationally expensive inference. In this paper, we propose a residual diffusion framework for sparse-view CT reconstruction that models the degradation pathway from clean images to filtered back-projection (FBP) reconstructions. Our method enables efficient sampling initialization from structured FBP estimates rather than pure noise, while incorporating multi-scale high-frequency conditioning and adaptive data consistency constraints to preserve anatomical detail and maintain measurement fidelity. Extensive experiments on both simulated and real datasets demonstrate that the proposed method consistently outperforms state-of-the-art diffusion-based approaches, achieving superior reconstruction quality with significantly reduced computational overhead.

## Requirements

This project requires [torch-radon](https://github.com/matteo-ronchetti/torch-radon) for CT forward and backward projection operations.

```bash
pip install torch-radon
```

## Project Structure

```
├── basic_ops.py      # Basic neural network operations
├── losses.py         # Loss functions
├── edgeloss.py       # Eagle loss for edge preservation
├── model.py          # UNet and FBPConvNet models
├── dataset.py        # Dataset classes for DICOM/IMA files
├── diffusion.py      # Gaussian diffusion model with adaptive data consistency
├── train.py          # Training script
├── test.py           # Testing script
└── README.md
```

## Usage

### Training

```bash
python train.py
```

### Testing

```bash
python test.py \
    --image_path /path/to/test/image.IMA \
    --model_path /path/to/model.pt \
    --high_freq_model_path /path/to/high_freq_model.pth \
    --timesteps 100
```
## Usage

### Training

```bash
python train.py
```

### Testing

```bash
python test.py \
    --image_path /path/to/test/image.IMA \
    --model_path /path/to/model.pt \
    --high_freq_model_path /path/to/high_freq_model.pth \
    --timesteps 100
```
## Checkpoints and Assets

Trained model checkpoints and related assets are provided via OneDrive:

https://1drv.ms/f/c/93483cb9d8985636/IgDY5PG1VOOkQJyj6xYh3mggAbJFxQGTeUYdx2jw9DJ8REc?e=FPNtza


