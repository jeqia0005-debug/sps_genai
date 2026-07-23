"""Energy-Based Model (EBM) for CIFAR-10 image generation.

Reference: Module 7 Practical 1 - Energy Based Methods.
Adapted from 1-channel MNIST to 3-channel CIFAR-10 (both 32x32).
Samples are drawn with Langevin dynamics: gradient descent on the INPUT image,
not on the network weights.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from app.image_utils import tensor_to_png

IMAGE_SIZE = 32
NUM_CHANNELS = 3


def swish(x):
    """Swish activation: smoother than ReLU, keeps gradients alive for x < 0."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Maps an image to a single scalar energy value."""

    def __init__(self, num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)            # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)            # 8  -> 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)            # 4  -> 2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """Langevin dynamics: iteratively move the INPUT images towards lower energy."""
    nn_energy_model.eval()

    for _ in range(steps):
        # Add noise without tracking gradients: the gradient is only needed for
        # the energy computation below, not for the noise injection itself.
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        energy = nn_energy_model(inp_imgs)

        # Explicitly ask for the gradient with respect to the input images.
        # grad_outputs is needed because `energy` holds one value per image.
        grads, = torch.autograd.grad(
            energy, inp_imgs, grad_outputs=torch.ones_like(energy)
        )

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)  # clipping stabilises the sampling
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


class Buffer:
    """Replay buffer of generated samples, so sampling does not restart from
    pure noise every step (avoids paying the Langevin mixing time each time)."""

    def __init__(self, model, device, batch_size: int = 128):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.examples = [
            torch.rand((1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device) * 2 - 1
            for _ in range(batch_size)
        ]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch_size, 0.05)

        new_rand_imgs = (
            torch.rand((n_new, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=self.device) * 2 - 1
        )
        old_imgs = torch.cat(
            random.choices(self.examples, k=self.batch_size - n_new), dim=0
        )
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)

        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class Metric:
    """Running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def update(self, value):
        self.total += float(value.detach()) if torch.is_tensor(value) else float(value)
        self.count += 1

    def result(self):
        return self.total / max(self.count, 1)

    def reset(self):
        self.total = 0.0
        self.count = 0


class EBM(nn.Module):
    """Training wrapper implementing the contrastive divergence loss."""

    def __init__(self, model, alpha, steps, step_size, noise, device, batch_size=128):
        super().__init__()
        self.model = model
        self.device = device
        self.buffer = Buffer(model, device=device, batch_size=batch_size)

        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for m in [
            self.loss_metric,
            self.reg_loss_metric,
            self.cdiv_loss_metric,
            self.real_out_metric,
            self.fake_out_metric,
        ]:
            m.reset()

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = self.buffer.sample_new_exmps(
            steps=self.steps, step_size=self.step_size, noise=self.noise
        )

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)

        self.model.train()
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(
            out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0
        )

        # Contrastive divergence: push energy DOWN on real data, UP on samples.
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())
        return self.metrics()


def generate_energy_image(
    weights_path: str = "cifar10_energy.pth",
    num_images: int = 1,
    steps: int = 256,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    device: str = "cpu",
):
    """Load the trained energy model and sample images with Langevin dynamics."""
    model = EnergyModel(num_channels=NUM_CHANNELS)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    start = torch.rand(
        (num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device
    ) * 2 - 1
    samples = generate_samples(model, start, steps, step_size, noise_std)

    samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    return tensor_to_png(samples)


if __name__ == "__main__":
    # Shape check
    net = EnergyModel()
    dummy = torch.randn(8, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print("Energy output shape:", net(dummy).shape)  # expected [8, 1]
    sampled = generate_samples(net, dummy, steps=2, step_size=10.0, noise_std=0.005)
    print("Sample shape:", sampled.shape)  # expected [8, 3, 32, 32]