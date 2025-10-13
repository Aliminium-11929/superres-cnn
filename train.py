import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.srcnn import SRCNN
from utils import upscale_image, show_images, psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: grayscale + resize for LR-HR pair
transform_hr = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Using a simple public dataset (e.g., CIFAR10) for quick tests
dataset = datasets.CIFAR10(root="./data", download=True, transform=transform_hr)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 3  # short run for baseline demo
for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch, (hr, _) in enumerate(loader):
        hr = hr.to(device)
        lr = F.interpolate(hr, scale_factor=0.5, mode='bicubic', recompute_scale_factor=True)
        lr_upscaled = F.interpolate(lr, scale_factor=2, mode='bicubic', recompute_scale_factor=True)

        optimizer.zero_grad()
        output = model(lr_upscaled)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

# Evaluate sample
sample_hr = hr[0].unsqueeze(0)
sample_lr = F.interpolate(sample_hr, scale_factor=0.5, mode='bicubic')
sample_lr_up = F.interpolate(sample_lr, scale_factor=2, mode='bicubic')
sample_out = model(sample_lr_up)

p = psnr(sample_hr, sample_out)
print(f"Baseline PSNR: {p:.2f} dB")

show_images(sample_lr_up, sample_out, sample_hr)
torch.save(model.state_dict(), "results/srcnn_baseline.pth")
