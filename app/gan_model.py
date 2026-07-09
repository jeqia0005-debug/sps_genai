import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Input noise: (B, 100). z_dim = size of the noise vector.
        self.z_dim = 100

        # Fully connected layer: 100 -> 7*7*128 = 6272 numbers.
        self.fc = nn.Linear(self.z_dim, 7 * 7 * 128)

        # ConvTranspose2D: 128 -> 64, upsamples 7x7 -> 14x14.
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)

        # ConvTranspose2D: 64 -> 1, upsamples 14x14 -> 28x28.
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x comes in as (B, 100).
        x = self.fc(x)                  # (B, 6272)
        x = x.view(-1, 128, 7, 7)       # reshape to (B, 128, 7, 7)

        x = self.deconv1(x)             # (B, 64, 14, 14)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)             # (B, 1, 28, 28)
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv2D: 1 -> 64, downsamples 28x28 -> 14x14.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # Conv2D: 64 -> 128, downsamples 14x14 -> 7x7.
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Flatten and map to a single real/fake output.
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        # x comes in as (B, 1, 28, 28).
        x = self.conv1(x)               # (B, 64, 14, 14)
        x = self.act1(x)

        x = self.conv2(x)               # (B, 128, 7, 7)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.flatten(x)             # (B, 128*7*7)
        x = self.fc(x)                  # (B, 1)
        return x
    
import io
from PIL import Image
from torchvision.utils import make_grid


def generate_digit_image(weights_path="mnist_gan_generator.pth", num_images=1):
    # Load the trained generator.
    generator = Generator()
    generator.load_state_dict(torch.load(weights_path, map_location="cpu"))
    generator.eval()

    # Generate images from random noise.
    with torch.no_grad():
        noise = torch.randn(num_images, 100)
        output = generator(noise)          # (num_images, 1, 28, 28), values in [-1, 1]

    # Convert to a viewable image.
    output = (output + 1) / 2              # scale to [0, 1]
    grid = make_grid(output, nrow=num_images)  # arrange into one image
    grid = grid.mul(255).clamp(0, 255).byte()   # to 0-255 pixel values
    ndarr = grid.permute(1, 2, 0).numpy()       # (H, W, C) for PIL

    image = Image.fromarray(ndarr)

    # Save into an in-memory PNG buffer to send over the API.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer
