import torch
import argparse
import os
from PIL import Image
from tqdm import tqdm

from dataset import DIV2KValidationDataset
from models.srcnn_improved import SRCNN, EDSR_Lite
from utils import psnr, ssim, show_images


def save_all_predictions(model, dataset, device, save_dir, save_comparisons=True):
    """
    Generate and save super-resolution predictions for entire dataset.
    
    Args:
        model: Trained model
        dataset: Validation dataset
        device: Device to run on
        save_dir: Directory to save predictions
        save_comparisons: If True, also save LR-SR-HR comparison images
    """
    model.eval()
    
    # Create subdirectories
    sr_dir = os.path.join(save_dir, 'SR_images')
    os.makedirs(sr_dir, exist_ok=True)
    
    if save_comparisons:
        comparison_dir = os.path.join(save_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
    
    # Metrics storage
    metrics = []
    
    print(f"\nGenerating super-resolution predictions...")
    print(f"Output directory: {save_dir}")
    print(f"Processing {len(dataset)} images...\n")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Load image
            lr_tensor, hr_tensor, filename = dataset[idx]
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)
            
            # Generate SR image
            sr_tensor = model(lr_tensor)
            
            # Calculate metrics
            psnr_val = psnr(hr_tensor, sr_tensor).item()
            ssim_val = ssim(hr_tensor, sr_tensor).item()
            metrics.append({
                'filename': filename,
                'psnr': psnr_val,
                'ssim': ssim_val
            })
            
            # Save SR image
            sr_img = sr_tensor.squeeze().cpu().clamp(0, 1).numpy()
            sr_img = (sr_img * 255).astype('uint8')
            sr_filename = filename.replace('.png', '_SR.png')
            Image.fromarray(sr_img, mode='L').save(
                os.path.join(sr_dir, sr_filename)
            )
            
            # Save comparison if requested
            if save_comparisons:
                show_images(
                    lr_tensor[0],
                    sr_tensor[0],
                    hr_tensor[0],
                    save_path=os.path.join(comparison_dir, filename.replace('.png', '_comparison.png'))
                )
    
    # Save metrics summary
    avg_psnr = sum(m['psnr'] for m in metrics) / len(metrics)
    avg_ssim = sum(m['ssim'] for m in metrics) / len(metrics)
    
    summary_path = os.path.join(save_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Super-Resolution Evaluation Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total Images: {len(metrics)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
        f.write(f"Per-Image Metrics:\n")
        f.write(f"-" * 50 + "\n")
        
        for m in metrics:
            f.write(f"{m['filename']:<40} PSNR: {m['psnr']:>6.2f} dB  SSIM: {m['ssim']:>6.4f}\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved successfully!")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  SR Images:        {sr_dir}")
    if save_comparisons:
        print(f"  Comparisons:      {comparison_dir}")
    print(f"  Metrics Summary:  {summary_path}")
    print(f"\nAverage Metrics:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation dataset
    print(f"\nLoading validation dataset...")
    val_dataset = DIV2KValidationDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir
    )
    print(f"Found {len(val_dataset)} validation images")
    
    # Load model
    if args.model == 'srcnn':
        model = SRCNN().to(device)
    elif args.model == 'edsr':
        model = EDSR_Lite(num_blocks=args.num_blocks).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    
    # Generate predictions
    metrics = save_all_predictions(
        model=model,
        dataset=val_dataset,
        device=device,
        save_dir=args.output_dir,
        save_comparisons=args.save_comparisons
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate and save super-resolution predictions for entire validation set'
    )
    
    # Model settings
    parser.add_argument('--model', type=str, default='edsr', 
                        choices=['srcnn', 'edsr'],
                        help='Model architecture')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='Number of residual blocks (for EDSR)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data settings
    parser.add_argument('--hr_dir', type=str, 
                        default='data/DIV2K_valid_HR_gray',
                        help='Path to HR validation images')
    parser.add_argument('--lr_dir', type=str, 
                        default='data/DIV2K_valid_LR_gray',
                        help='Path to LR validation images')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, 
                        default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--save_comparisons', action='store_true',
                        help='Save side-by-side comparison images')
    
    args = parser.parse_args()
    main(args)