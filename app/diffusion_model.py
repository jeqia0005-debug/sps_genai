"""Denoising Diffusion Model (UNet) for CIFAR-10 image generation.

Reference: Module 7 Practical 2 - Diffusion Methods.
Adapted from 64x64 Flowers102 to 32x32 CIFAR-10 (3 channels).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.image_utils import tensor_to_png

IMAGE_SIZE = 32
NUM_CHANNELS = 3


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Maps diffusion times in [0, 1] to (noise_rates, signal_rates)."""
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(
        torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device)
    )
    end_angle = torch.acos(
        torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device)
    )

    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)
    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    """Encodes a scalar (the noise variance) as a vector of sines and cosines."""

    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)"""
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if self.needs_projection
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks followed by average pooling (halves the resolution)."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)  # keep for the skip connections
        return self.pool(x)


class UpBlock(nn.Module):
    """Upsampling followed by residual blocks fed with the skip connections."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise that was added to a noisy image, given the noise variance."""

    def __init__(self, image_size=IMAGE_SIZE, num_channels=NUM_CHANNELS, embedding_dim=32):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim

        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)   # 32 -> 16
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)   # 16 -> 8
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)   # 8  -> 4

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)      # 4  -> 8
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)       # 8  -> 16
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)       # 16 -> 32

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)

        # Embed the noise variance and broadcast it over the whole image grid,
        # then concatenate it with the image features.
        noise_emb = self.embedding(noise_variances)              # (B, 1, 1, 32)
        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    """Wraps the UNet with the forward (corruption) and reverse (generation) processes."""

    def __init__(self, model, schedule_fn, normalizer_mean=0.5, normalizer_std=0.5):
        super().__init__()
        self.network = model
        self.schedule_fn = schedule_fn
        self.normalizer_mean = normalizer_mean
        self.normalizer_std = normalizer_std

    def normalize(self, x):
        return (x - self.normalizer_mean) / self.normalizer_std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        self.network.train(training)
        pred_noises = self.network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones(
                (initial_noise.shape[0], 1, 1, 1), device=initial_noise.device
            ) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )

            next_t = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_t)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=IMAGE_SIZE, initial_noise=None):
        device = next(self.parameters()).device
        if initial_noise is None:
            initial_noise = torch.randn(
                (num_images, self.network.num_channels, image_size, image_size),
                device=device,
            )
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = self.normalize(images)
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(self, images, loss_fn):
        images = self.normalize(images)
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            loss = loss_fn(pred_noises, noises)
        return loss.item()


def generate_diffusion_image(
    weights_path: str = "cifar10_diffusion.pth",
    num_images: int = 1,
    diffusion_steps: int = 20,
    device: str = "cpu",
):
    """Load the trained UNet and generate images by reverse diffusion."""
    unet = UNet(image_size=IMAGE_SIZE, num_channels=NUM_CHANNELS)
    unet.load_state_dict(torch.load(weights_path, map_location=device))

    model = DiffusionModel(unet, offset_cosine_diffusion_schedule).to(device)
    model.eval()

    images = model.generate(
        num_images=num_images, diffusion_steps=diffusion_steps, image_size=IMAGE_SIZE
    )
    return tensor_to_png(images)


if __name__ == "__main__":
    # Shape check
    net = UNet()
    dummy_images = torch.randn(4, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    dummy_variances = torch.rand(4, 1, 1, 1)
    print("UNet output shape:", net(dummy_images, dummy_variances).shape)  # [4, 3, 32, 32]

    diffusion = DiffusionModel(net, offset_cosine_diffusion_schedule)
    print("Generated shape:", diffusion.generate(2, diffusion_steps=3).shape)  # [2, 3, 32, 32]