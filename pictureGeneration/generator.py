import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import os

print(os.path.abspath("pictureGeneration/data"))
print(os.path.exists("pictureGeneration/data"))
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, feature_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, feature_g * 16, 4, 1, 0),    # (1 → 4)
            self._block(feature_g * 16, feature_g * 8, 4, 2, 1),  # 4 → 8
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),   # 8 → 16
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),   # 16 → 32
            self._block(feature_g * 2, feature_g, 4, 2, 1),       # 32 → 64
            self._block(feature_g, feature_g // 2, 4, 2, 1),      # 64 → 112
            nn.ConvTranspose2d(feature_g // 2, img_channels, 4, 2, 1),  # 112 → 224
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),  # 224 → 112
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d * 2, 4, 2, 1),    # 112 → 56
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),  # 56 → 28
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),  # 28 → 14
            self._block(feature_d * 8, feature_d * 16, 4, 2, 1), # 14 → 7
            nn.Conv2d(feature_d * 16, 1, 7, 1, 0),  # 7×7 → 1×1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# Hyperparameters
z_dim = 100
#image_size = 224
channels_img = 1  # 3 for RGB datasets
features_d = 64
features_g = 64
batch_size = 64
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Models
gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

#resizing the image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder(root="pictureGeneration/data", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator
        #disc.trainable = True
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        #disc.trainable = False
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {loss_disc.item():.4f} | Loss G: {loss_gen.item():.4f}")
    if (epoch + 1) % 10 == 0:
        save_image(gen(fixed_noise), f"generated_epoch_{epoch+1}.png", normalize=True)
