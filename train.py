import torch
import copy
import math
import itertools
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils
from tqdm import tqdm
from scipy.io import savemat
import numpy as np
from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

from model import Unet, FBPConvNet
from dataset import IMADataset
from diffusion import GaussianDiffusion, get_named_eta_schedule, ModelMeanType, LossType


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


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:
    def __init__(
        self,
        model,
        diffusion_model,
        train_dataset,
        test_dataset,
        *,
        train_batch_size=16,
        test_batch_size=1,
        gradient_accumulate_every=1,
        train_lr=1e-5,
        train_num_steps=100000,
        ema_decay=0.995,
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results',
        amp=False,
        max_grad_norm=1.,
        high_freq_model_path=None
    ):
        super().__init__()

        self.model = model
        self.diffusion_model = diffusion_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.model_high = None
        if high_freq_model_path is not None:
            self.model_high = FBPConvNet().to(self.device)
            self.model_high.load_state_dict(torch.load(high_freq_model_path))
            self.model_high.eval()
            print(f"High frequency model loaded from {high_freq_model_path}")

        self.opt = Adam(self.model.parameters(), lr=train_lr)

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.train_dl = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_dl = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False)

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.opt.load_state_dict(data['opt'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0

                for _ in range(self.gradient_accumulate_every):
                    batch = next(iter(self.train_dl))
                    x_0, y = batch
                    x_0, y = x_0.to(self.device), y.to(self.device)

                    t = torch.randint(0, self.diffusion_model.num_timesteps, (x_0.shape[0],), device=self.device).long()

                    with autocast(enabled=self.amp):
                        losses = self.diffusion_model.training_losses(
                            self.model,
                            x_0,
                            y,
                            t,
                            model_high=self.model_high
                        )
                        loss = losses['loss'].mean() / self.gradient_accumulate_every

                    total_loss += loss.item()

                    self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.ema.update_model_average(self.ema_model, self.model)

                pbar.set_description(f'loss: {total_loss:.8f}')
                pbar.update(1)
                self.step += 1

                if self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every

                    self.ema_model.eval()
                    with torch.no_grad():
                        train_sample = self.diffusion_model.p_sample_loop(
                            x_0=x_0[0:1],
                            y=y[0:1],
                            model=self.ema_model,
                            model_high=self.model_high,
                            noise=torch.randn_like(y[0:1]),
                            device=self.device,
                            progress=True,
                            regularize=True,
                        )

                    selected_batch = 81
                    test_batch = next(itertools.islice(self.test_dl, selected_batch, selected_batch + 1))
                    test_x_0, test_y = test_batch

                    test_x_0, test_y = test_x_0.to(self.device), test_y.to(self.device)
                    with torch.no_grad():
                        test_samples = self.diffusion_model.p_sample_loop(
                            x_0=test_x_0,
                            y=test_y,
                            model=self.ema_model,
                            model_high=self.model_high,
                            noise=torch.randn_like(test_y),
                            device=self.device,
                            regularize=False,
                        )

                    test_samples_normalized = clear(test_samples)
                    test_x_0_normalized = clear(test_x_0)

                    mse, ssim, psnr = compare(test_samples_normalized, test_x_0_normalized, verbose=False)

                    print(f"Step {self.step}: testMSE: {mse:.4f}, testSSIM: {ssim:.4f}, testPSNR: {psnr:.4f}")

                    self.ema_model.train()

                    utils.save_image(train_sample, str(self.results_folder / f'train_sample.png'), nrow=int(math.sqrt(self.num_samples)))
                    utils.save_image(test_samples, str(self.results_folder / f'test_sample.png'), nrow=1)
                    utils.save_image(test_y, str(self.results_folder / f'test_input.png'), nrow=1)
                    savemat(str(self.results_folder / f'test_ground_truth.mat'), {'recon': test_x_0_normalized})
                    savemat(str(self.results_folder / f'test_sample.mat'), {'recon': test_samples_normalized})

                    utils.save_image(test_x_0, str(self.results_folder / f'test_ground_truth.png'), nrow=1)

                    self.save(milestone)

                if self.step >= self.train_num_steps:
                    print('training complete')
                    return


if __name__ == "__main__":
    timesteps = 30
    min_noise_level = 0.02
    etas_end = 0.99
    kappa = 0.1
    power = 0.3

    initial_lr = 1e-3
    use_lr_scheduler = True

    resume_training = False
    resume_checkpoint = 2
    results_folder = './results'

    data_folder = '/home/zqq/usb_disk/wujia/aapmdata/full_3mm'
    test_data_folder = '/home/zqq/usb_disk/wujia/aapmdata/test'
    high_freq_model_path = '/home/zqq/usb_disk/wujia/diffusion_ct_my/高频权重/high_32_full_0428.pth'

    model = Unet(
        dim=64,
        input_channels=1,
        hf_channels=1,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
    )

    for name, param in model.named_parameters():
        if 'downs' in name or 'mid_' in name or 'time_mlp' in name or 'init_conv' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = Adam(trainable_params, lr=initial_lr)

    if use_lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=500000, eta_min=1e-5)
    else:
        scheduler = None

    high_freq_model = FBPConvNet()
    high_freq_model.load_state_dict(torch.load(high_freq_model_path))
    high_freq_model.eval()

    trainer = Trainer(
        model=model,
        diffusion_model=GaussianDiffusion(
            sqrt_etas=get_named_eta_schedule(
                'exponential',
                timesteps,
                min_noise_level,
                etas_end=etas_end,
                kappa=kappa,
                kwargs={'power': power}
            ),
            kappa=kappa,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            sf=1,
            scale_factor=None,
            normalize_input=True,
            latent_flag=True
        ),
        train_dataset=IMADataset(data_folder),
        test_dataset=IMADataset(test_data_folder),
        train_batch_size=1,
        test_batch_size=1,
        train_lr=initial_lr,
        train_num_steps=500000,
        save_and_sample_every=1000,
        results_folder=results_folder,
        high_freq_model_path=None
    )

    trainer.opt = optimizer
    trainer.model_high = high_freq_model.to(trainer.device)
    trainer.scheduler = scheduler
    trainer.ema_model_high = copy.deepcopy(high_freq_model).to(trainer.device)

    if resume_training:
        print(f"Loading checkpoint {resume_checkpoint}...")
        trainer.load(resume_checkpoint)

    trainer.train()
