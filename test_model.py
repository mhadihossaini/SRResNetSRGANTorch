import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import os
from torchvision.transforms import Resize, ToPILImage, ToTensor
import matplotlib.pyplot as plt
from model_definitions import SRResNet, Discriminator
from load_dataset import LocalFlowersDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = SRResNet().to(device)  # Assuming SRResNet can also represent your SRGAN generator for simplicity
generator.load_state_dict(torch.load('generator_model.pth', map_location=device))
generator.eval()

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load('discriminator_model.pth', map_location=device))
discriminator.eval()

def show_images(lr_img, hr_img, generator, discriminator):
    # Convert tensor images to PIL for processing and visualization
    lr_pil = ToPILImage()(lr_img)
    hr_pil = ToPILImage()(hr_img)
    
    # Bicubic upscaling
    bicubic_img = lr_pil.resize(hr_pil.size, Image.BICUBIC)
    
    # Generate with SRResNet/SRGAN
    with torch.no_grad():
        sr_img = generator(lr_img.to(device).unsqueeze(0)).squeeze(0).cpu()
        sr_pil = ToPILImage()(sr_img)
    
    # Displaying the images
    imgs = [bicubic_img, sr_pil, hr_pil]
    titles = ['Bicubic Upscaling', 'SRResNet/SRGAN', 'Original High-Resolution']
    
    plt.figure(figsize=(20, 10))
    for i, img in enumerate(imgs):
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(titles[i])
        #plt.axis('off')
    
    # Discriminator Evaluation (assuming real=1, fake=0)
    real_score = discriminator(hr_img.unsqueeze(0).to(device)).item()
    fake_score = discriminator(sr_img.unsqueeze(0).to(device)).item()
    
    # Since there's no image to show for discriminator scores, just print them out
    print(f'Discriminator Scores - Real: {real_score:.2f} | Fake: {fake_score:.2f}')
    
    plt.show()

# Dataset Directory
train_dir = 'train_dataset'
test_dir = 'test_dataset'

# Creating dataset instances
train_dataset = LocalFlowersDataset(train_dir)
test_dataset = LocalFlowersDataset(test_dir)

# Creating DataLoader instances

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

lr_img, hr_img = next(iter(test_loader))
lr_img, hr_img = lr_img[2], hr_img[2]  # Adjust index to 0 for simplicity
show_images(lr_img, hr_img, generator, discriminator)
