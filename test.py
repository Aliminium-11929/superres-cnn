import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.srcnn_improved import SRCNN, EDSR_Lite
from utils import psnr, ssim


def load_image(path, device):
    """Load and preprocess image."""
    img = Image.open(path).convert('L')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return img_tensor, img


def save_image(tensor, path):
    """Save tensor as image."""
    img = tensor.squeeze().cpu().clamp(0, 1).numpy()
    img = (img * 255).astype('uint8')
    Image.fromarray(img, mode='L').save(path)


def test_image(model, lr_path, hr_path, device, save_dir):
    """Test model on a single image."""
    model.eval()
    
    # Load images
    lr_tensor, lr_img = load_image(lr_path, device)
    
    if hr_path:
        hr_tensor, hr_img = load_image(hr_path, device)
    else:
        hr_tensor = None
    
    # Inference
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # Calculate metrics if HR is available
    if hr_tensor is not None:
        psnr_val = psnr(hr_tensor, sr_tensor).item()
        ssim_val = ssim(hr_tensor, sr_tensor).item()
        print(f"PSNR: {psnr_val:.2f}dB")
        print(f"SSIM: {ssim_val:.4f}")
    
    # Save SR image
    filename = os.path.basename(lr_path).replace('.png', '_SR.png')
    save_path = os.path.join(save_dir, filename)
    save_image(sr_tensor, save_path)
    print(f"Saved SR image to: {save_path}")
    
    # Visualize
    if hr_tensor is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_tensor.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title('LR Input')
        axes[0].axis('off')
        
        axes[1].imshow(sr_tensor.squeeze().cpu().clamp(0, 1).numpy(), cmap='gray')
        axes[1].set_title(f'SR Output\nPSNR: {psnr_val:.2f}dB')
        axes[1].axis('off')
        
        axes[2].imshow(hr_tensor.squeeze().cpu().numpy(), cmap='gray')
        axes[2].set_title('HR Target')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename.replace('_SR.png', '_comparison.png')), dpi=150)
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(lr_tensor.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title('LR Input')
        axes[0].axis('off')
        
        axes[1].imshow(sr_tensor.squeeze().cpu().clamp(0, 1).numpy(), cmap='gray')
        axes[1].set_title('SR Output')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename.replace('_SR.png', '_comparison.png')), dpi=150)
        plt.show()


def test_folder(model, lr_dir, hr_dir, device, save_dir):
    """Test model on all images in a folder."""
    model.eval()
    
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
    
    avg_psnr = 0
    avg_ssim = 0
    count = 0
    
    for lr_file in lr_files:
        lr_path = os.path.join(lr_dir, lr_file)
        hr_path = os.path.join(hr_dir, lr_file) if hr_dir else None
        
        if hr_path and not os.path.exists(hr_path):
            print(f"Warning: HR image not found for {lr_file}")
            hr_path = None
        
        print(f"\nProcessing: {lr_file}")
        
        # Load images
        lr_tensor, _ = load_image(lr_path, device)
        
        if hr_path:
            hr_tensor, _ = load_image(hr_path, device)
        else:
            hr_tensor = None
        
        # Inference
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        # Calculate metrics
        if hr_tensor is not None:
            psnr_val = psnr(hr_tensor, sr_tensor).item()
            ssim_val = ssim(hr_tensor, sr_tensor).item()
            print(f"PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}")
            
            avg_psnr += psnr_val
            avg_ssim += ssim_val
            count += 1
        
        # Save SR image
        save_path = os.path.join(save_dir, lr_file.replace('.png', '_SR.png'))
        save_image(sr_tensor, save_path)
    
    if count > 0:
        print(f"\nAverage PSNR: {avg_psnr/count:.2f}dB")
        print(f"Average SSIM: {avg_ssim/count:.4f}")


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if args.model == 'srcnn':
        model = SRCNN().to(device)
    elif args.model == 'edsr':
        model = EDSR_Lite(num_blocks=args.num_blocks).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Run inference
    if args.lr_image:
        # Single image
        test_image(model, args.lr_image, args.hr_image, device, args.output_dir)
    elif args.lr_dir:
        # Folder
        test_folder(model, args.lr_dir, args.hr_dir, device, args.output_dir)
    else:
        print("Please specify either --lr_image or --lr_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Super-Resolution Model')
    
    # Model settings
    parser.add_argument('--model', type=str, default='edsr', choices=['srcnn', 'edsr'],
                        help='Model architecture')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='Number of residual blocks (for EDSR)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input settings
    parser.add_argument('--lr_image', type=str, default='',
                        help='Path to single LR image')
    parser.add_argument('--hr_image', type=str, default='',
                        help='Path to single HR image (optional, for metrics)')
    parser.add_argument('--lr_dir', type=str, default='',
                        help='Path to LR images folder')
    parser.add_argument('--hr_dir', type=str, default='',
                        help='Path to HR images folder (optional, for metrics)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save SR images')
    
    args = parser.parse_args()
    main(args)