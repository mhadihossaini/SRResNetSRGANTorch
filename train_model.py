import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_dataset import LocalFlowersDataset
from model_definitions import SRResNet, Discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the models
generator = SRResNet().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

# Define loss functions
criterion_GAN = nn.BCEWithLogitsLoss().to(device)
criterion_content = nn.MSELoss().to(device)

def train(generator, discriminator, loader, optimizer_G, optimizer_D, epochs=5):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            valid = torch.ones((hr_imgs.size(0),), dtype=torch.float, device=device)
            fake = torch.zeros((hr_imgs.size(0),), dtype=torch.float, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = criterion_GAN(discriminator(hr_imgs), valid)
            fake_imgs = generator(lr_imgs)
            fake_loss = criterion_GAN(discriminator(fake_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Adversarial loss to fool the discriminator
            g_loss = criterion_GAN(discriminator(fake_imgs), valid)

            # Content loss
            content_loss = criterion_content(fake_imgs, hr_imgs)
            # Total loss
            total_loss = content_loss + 0.001 * g_loss

            total_loss.backward()
            optimizer_G.step()

            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, D Loss: {d_loss.item()}, G Loss: {total_loss.item()}")

            # Optionally, add code to save models and log detailed progress

# Dataset Directory
train_dir = 'train_dataset'
test_dir = 'test_dataset'

# Creating dataset instances
train_dataset = LocalFlowersDataset(train_dir)
test_dataset = LocalFlowersDataset(test_dir)

# Creating DataLoader instances

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

train(generator, discriminator, train_loader, optimizer_G, optimizer_D, epochs=5)

# Define paths for saving
generator_path = 'generator_model.pth'
discriminator_path = 'discriminator_model.pth'

# Save models
torch.save(generator.state_dict(), generator_path)
torch.save(discriminator.state_dict(), discriminator_path)
