import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def init(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def len(self):
        return len(self.image_files)

    def getitem(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image 




def generate(session_id, pic_amount, epoch_amount):

    print("Hello from generator.py")

    print("Session ID:", session_id)
    print("Picture Amount:", pic_amount)
    print("Epoch Amount:", epoch_amount)

    print(torch.cuda.is_available())

    class Generator(nn.Module):
        def __init__(self, z_dim, img_channels, feature_g):
            super(Generator, self).__init__()
            self.gen = nn.Sequential(
                self._block(z_dim, feature_g * 8, 4, 1, 0),  # (batch_size, f_g*8, 4, 4)
                self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
                self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
                self._block(feature_g * 2, feature_g, 4, 2, 1),
                nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),
                nn.Tanh(),  # Normalize output to [-1, 1]
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
                nn.Conv2d(img_channels, feature_d, 4, 2, 1),
                nn.LeakyReLU(0.2),
                self._block(feature_d, feature_d * 2, 4, 2, 1),
                self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
                self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
                nn.Conv2d(feature_d * 8, 1, 4, 1, 0),
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
    image_size = 64
    channels_img = 3  # 3 for RGB datasets like CIFAR10
    features_d = 64
    features_g = 64
    batch_size = 128
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

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = CustomImageDataset(image_dir='uploads/'+session_id+'/', transform=transform)
    #dataset = datasets.ImageFolder(root="uploads", transform=transform)
    #dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = epoch_amount
    fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator
            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {loss_disc.item():.4f} | Loss G: {loss_gen.item():.4f}")
        output_dir = os.path.join("download", str(session_id))
        os.makedirs(output_dir, exist_ok=True)
        save_image(gen(fixed_noise), os.path.join(output_dir, f"generated_epoch_{epoch+1}.png"), normalize=True)
        
    return 0