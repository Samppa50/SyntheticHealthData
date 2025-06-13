import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
import os
from PIL import Image
from gifmaker import make_gif

progress_bar = 0
stop_prosessing = False

def get_stop():
    global stop_prosessing
    return stop_prosessing

def set_stop(value):
    global stop_prosessing
    stop_prosessing = value
    print(f"Stop processing set to: {value}")



def get_progress():
    return progress_bar

def set_progress(progress):
    global progress_bar
    progress_bar = progress
    print(f"Progress set to: {progress}")


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image




def generate(session_id, pic_amount, epoch_amount, generation_type):

    print("Hello from generator.py")

    print("Session ID:", session_id)
    print("Picture Amount:", pic_amount)
    print("Epoch Amount:", epoch_amount)
    print("Generation Type:", generation_type)
    # Generation type 0 = start training new model
    # Generation type 1 = continue training existing hand model
    # new generation types can be added later

    if generation_type != 0:
        epoch_amount = 0
        print(f'Changed epoch amount to {epoch_amount} for generation type {generation_type}')


    print(torch.cuda.is_available())

    model_path = os.path.join("models", "hand_model.pth")
    os.makedirs("models", exist_ok=True)


    class Generator(nn.Module):
        def __init__(self, nz, ngf, nc):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)

# Discriminator (no sigmoid)
    class Discriminator(nn.Module):
        def __init__(self, nc, ndf):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            )

        def forward(self, input):
            return self.main(input).view(-1)

    # Weight Initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Gradient Penalty
    def compute_gradient_penalty(D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones_like(d_interpolates, device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Loss functions
    def discriminator_loss(real_output, fake_output):
        return fake_output.mean() - real_output.mean()

    def generator_loss(fake_output):
        return -fake_output.mean()

    # Hyperparameters
    image_size = 128
    nz = 100
    ngf = 128
    ndf = 128
    nc = 3
    lambda_gp = 10
    n_critic = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Models
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    if generation_type == 1:
        print(f"Loading existing generator weights from {model_path}")
        netG.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Starting with a new generator model.")

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.0, 0.9))

    # Data
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data_location = os.path.join('uploads', session_id)
    if generation_type == 0:
        dataset = CustomImageDataset(image_dir=data_location, transform=transform)
        if len(dataset) == 0:
            print("No images found for training. Aborting.")
            set_progress(0)
            return -1
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        num_epochs = epoch_amount
        for epoch in range(num_epochs):
            if get_stop():
                print("User stopped processing.")
                set_progress(0)
                return -1
            for i, data in enumerate(dataloader):
                real_data = data.to(device)
                batch_size = real_data.size(0)

                # Train Discriminator
                for _ in range(n_critic):
                    netD.zero_grad()
                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_data = netG(noise).detach()
                    real_output = netD(real_data)
                    fake_output = netD(fake_data)
                    gp = compute_gradient_penalty(netD, real_data.data, fake_data.data)
                    d_loss = discriminator_loss(real_output, fake_output) + lambda_gp * gp
                    d_loss.backward()
                    optimizerD.step()

                # Train Generator
                netG.zero_grad()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_data = netG(noise)
                fake_output = netD(fake_data)
                g_loss = generator_loss(fake_output)
                g_loss.backward()
                optimizerG.step()

                if i % 50 == 0:
                    print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                        f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} "
                        f"D(real): {real_output.mean().item():.4f} D(fake): {fake_output.mean().item():.4f}")

            # Save samples
            if epoch % 100 == 0:
                with torch.no_grad():
                    fake_images = netG(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
                    output_dir_epoch = os.path.join("gifs/epochs/")
                    os.makedirs(output_dir_epoch, exist_ok=True)
                    save_image(fake_images, os.path.join(output_dir_epoch, f"generated_epoch_{epoch+1}.png"), normalize=True)

            set_progress(epoch / num_epochs * 100)
    else:
        print("Skipping training, using pretrained model to generate images.")

    output_dir = os.path.join("download", str(session_id))
    os.makedirs(output_dir, exist_ok=True)
    netG.eval()
    with torch.no_grad():
        for idx in range(1, pic_amount + 1):
            noise = torch.randn(1, nz, 1, 1, device=device)
            fake_img = netG(noise).detach().cpu()
            save_image(fake_img, os.path.join(output_dir, f"generated_{idx}.png"), normalize=True)

    os.makedirs("gifs/generated/", exist_ok=True)
    if make_gif("gifs/epochs/", "gifs/generated/animated.gif", duration=1000):
        print("GIF created successfully.")
    else:
        print("Failed to create GIF. No images found.")
    set_progress(0)
    print("Generation completed successfully.")

    return 0