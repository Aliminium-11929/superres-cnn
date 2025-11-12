import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from math import log10

def psnr(hr, sr):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher is better.
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(hr, sr, window_size=11):
    """
    Calculate Structural Similarity Index (SSIM).
    Range: [-1, 1], higher is better.
    Simplified version for batch processing.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                          for x in range(window_size)])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(hr.device)
    
    # Calculate means
    mu1 = F.conv2d(hr, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(sr, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(hr * hr, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(sr * sr, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(hr * sr, window, padding=window_size//2, groups=1) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def show_images(lr, sr, hr, save_path=None):
    """
    Display LR input, SR output, and HR target side by side.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy for display
    imgs = [lr, sr, hr]
    titles = ["Input (LR)", "Output (SR)", "Target (HR)"]
    
    for i in range(3):
        img = imgs[i].detach().cpu().squeeze().numpy()
        axs[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axs[i].set_title(titles[i], fontsize=14)
        axs[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename} (Epoch {epoch}, Loss: {loss:.6f})")
    return epoch, loss


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs, decay_factor=0.5):
    """
    Decay learning rate by a factor every decay_epochs.
    """
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr