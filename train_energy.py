"""Train the Energy-Based Model on CIFAR-10 and save the weights."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.energy_model import EBM, EnergyModel

SUBSET_SIZE = 5000      # subset of CIFAR-10 to keep CPU training time reasonable
BATCH_SIZE = 128
EPOCHS = 5
LANGEVIN_STEPS = 40
WEIGHTS_PATH = "cifar10_energy.pth"


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Normalize to [-1, 1] to match the clamping range used in Langevin sampling.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset = Subset(dataset, range(SUBSET_SIZE))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = EnergyModel(num_channels=3).to(device)
    ebm = EBM(
        model,
        alpha=0.1,
        steps=LANGEVIN_STEPS,
        step_size=10.0,
        noise=0.005,
        device=device,
        batch_size=BATCH_SIZE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    for epoch in range(EPOCHS):
        ebm.reset_metrics()
        for index, (real_imgs, _) in enumerate(loader):
            metrics = ebm.train_step(real_imgs.to(device), optimizer)
            if (index + 1) % 5 == 0:
                print(
                    f"  epoch {epoch + 1} batch {index + 1}/{len(loader)} - "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
                    flush=True,
                )

        # Save every epoch so training can be stopped at any point.
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Epoch {epoch + 1}/{EPOCHS} done, weights saved to {WEIGHTS_PATH}", flush=True)


if __name__ == "__main__":
    main()