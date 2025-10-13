import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def psnr(hr, sr):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def show_images(lr_up, sr, hr):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    imgs = [lr_up, sr, hr]
    titles = ["Input (Bicubic)", "Output (SRCNN)", "Target (HR)"]
    for i in range(3):
        axs[i].imshow(imgs[i].detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()
def upscale_image(img_tensor, scale_factor=2):
    """Performs simple bicubic upscaling for a given image tensor."""
    return F.interpolate(img_tensor, scale_factor=scale_factor, mode='bicubic', recompute_scale_factor=True)
