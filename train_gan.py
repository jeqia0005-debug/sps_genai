import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator

# ----- Config -----
EPOCHS = 10
BATCH_SIZE = 128
Z_DIM = 100
LR = 2e-4
DEVICE = torch.device("cpu")

# ----- Data: MNIST normalized to [-1, 1] to match the Generator's Tanh output -----
transform = transforms.Compose([
    transforms.ToTensor(),                 # -> [0, 1]
    transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Models, loss, optimizers -----
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# A fixed noise batch so we can watch the same digits improve across epochs.
fixed_noise = torch.randn(64, Z_DIM, device=DEVICE)
os.makedirs("samples", exist_ok=True)

# ----- Training loop -----
for epoch in range(1, EPOCHS + 1):
    for real_images, _ in dataloader:
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # --- Train Discriminator: real -> 1, fake -> 0 ---
        noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
        fake_images = generator(noise)

        d_real = discriminator(real_images)
        d_fake = discriminator(fake_images.detach())
        loss_d = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # --- Train Generator: try to make D output "real" (1) on fakes ---
        d_fake_for_g = discriminator(fake_images)
        loss_g = criterion(d_fake_for_g, real_labels)

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print(f"Epoch [{epoch}/{EPOCHS}]  loss_D: {loss_d.item():.4f}  loss_G: {loss_g.item():.4f}")

    # Save a grid of sample digits from the fixed noise after each epoch.
    with torch.no_grad():
        samples = generator(fixed_noise)
        samples = (samples + 1) / 2  # back to [0, 1] for viewing
        save_image(samples, f"samples/epoch_{epoch:02d}.png", nrow=8)

# ----- Save the trained generator weights -----
torch.save(generator.state_dict(), "mnist_gan_generator.pth")
print("Saved generator weights to mnist_gan_generator.pth")