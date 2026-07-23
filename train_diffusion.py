"""Train the diffusion model (UNet) on CIFAR-10 and save the weights."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.diffusion_model import (
    IMAGE_SIZE,
    NUM_CHANNELS,
    DiffusionModel,
    UNet,
    offset_cosine_diffusion_schedule,
)

SUBSET_SIZE = 8000      # subset of CIFAR-10 to keep CPU training time reasonable
BATCH_SIZE = 64
EPOCHS = 10
WEIGHTS_PATH = "cifar10_diffusion.pth"


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Keep images in [0, 1]; the model normalizes internally to mean 0 / std 1.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_set = Subset(train_set, range(SUBSET_SIZE))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_set = Subset(test_set, range(1000))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    unet = UNet(image_size=IMAGE_SIZE, num_channels=NUM_CHANNELS)
    model = DiffusionModel(unet, offset_cosine_diffusion_schedule).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for index, (images, _) in enumerate(train_loader):
            loss = model.train_step(images.to(device), optimizer, loss_fn)
            running_loss += loss
            if (index + 1) % 20 == 0:
                print(
                    f"  epoch {epoch + 1} batch {index + 1}/{len(train_loader)} - "
                    f"train loss: {running_loss / (index + 1):.4f}",
                    flush=True,
                )

        val_loss = sum(
            model.test_step(images.to(device), loss_fn) for images, _ in test_loader
        ) / len(test_loader)

        # Save every epoch so training can be stopped at any point.
        torch.save(unet.state_dict(), WEIGHTS_PATH)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - train loss: {running_loss / len(train_loader):.4f}, "
            f"val loss: {val_loss:.4f}, weights saved to {WEIGHTS_PATH}",
            flush=True,
        )


if __name__ == "__main__":
    main()