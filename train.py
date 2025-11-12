import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from dataset import DIV2KDataset, DIV2KValidationDataset
from models.srcnn_improved import SRCNN, EDSR_Lite
from utils import psnr, ssim, show_images, save_checkpoint, load_checkpoint, AverageMeter, adjust_learning_rate


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for lr, hr in pbar:
        lr, hr = lr.to(device), hr.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        sr = model(lr)
        
        # Calculate loss
        loss = criterion(sr, hr)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            batch_psnr = psnr(hr, sr)
        
        losses.update(loss.item(), lr.size(0))
        psnr_meter.update(batch_psnr.item(), lr.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.6f}',
            'PSNR': f'{psnr_meter.avg:.2f}dB'
        })
    
    return losses.avg, psnr_meter.avg


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        for lr, hr, _ in tqdm(val_loader, desc="Validating"):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            sr = model(lr)
            
            # Calculate metrics
            loss = criterion(sr, hr)
            batch_psnr = psnr(hr, sr)
            batch_ssim = ssim(hr, sr)
            
            losses.update(loss.item(), lr.size(0))
            psnr_meter.update(batch_psnr.item(), lr.size(0))
            ssim_meter.update(batch_ssim.item(), lr.size(0))
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg


def save_sample_images(model, val_dataset, device, save_dir, epoch, num_samples=5):
    """
    Save super-resolution predictions for sample images.
    """
    model.eval()
    sample_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            lr_sample, hr_sample, filename = val_dataset[i]
            lr_sample = lr_sample.unsqueeze(0).to(device)
            hr_sample = hr_sample.unsqueeze(0).to(device)
            sr_sample = model(lr_sample)
            
            # Calculate metrics
            psnr_val = psnr(hr_sample, sr_sample).item()
            ssim_val = ssim(hr_sample, sr_sample).item()
            
            # Save comparison
            show_images(
                lr_sample[0],
                sr_sample[0],
                hr_sample[0],
                save_path=os.path.join(sample_dir, f'{i+1}_{filename.replace(".png", "")}_comparison.png')
            )
            
            # Save SR image separately
            sr_img = sr_sample.squeeze().cpu().clamp(0, 1).numpy()
            sr_img = (sr_img * 255).astype('uint8')
            from PIL import Image
            Image.fromarray(sr_img, mode='L').save(
                os.path.join(sample_dir, f'{i+1}_{filename.replace(".png", "")}_SR.png')
            )
    
    print(f"Saved {min(num_samples, len(val_dataset))} sample predictions to {sample_dir}")


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'predictions'), exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DIV2KDataset(
        hr_dir=args.hr_train_dir,
        lr_dir=args.lr_train_dir,
        patch_size=args.patch_size,
        augment=True
    )
    
    val_dataset = DIV2KValidationDataset(
        hr_dir=args.hr_val_dir,
        lr_dir=args.lr_val_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    if args.model == 'srcnn':
        model = SRCNN().to(device)
    elif args.model == 'edsr':
        model = EDSR_Lite(num_blocks=args.num_blocks).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
            start_epoch += 1
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs)
        print(f"\nEpoch {epoch+1}/{args.epochs} - LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
            print(f"Validation - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model}_best.pth')
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    best_checkpoint_path
                )
                print(f"★ New best model! PSNR: {val_psnr:.2f}dB")
                
                # Auto-export best weights for inference
                if args.auto_export:
                    export_dir = os.path.join(args.checkpoint_dir, 'best_weights_export')
                    print(f"Auto-exporting best weights to {export_dir}...")
                    from export_model import export_model_weights
                    export_model_weights(
                        checkpoint_path=best_checkpoint_path,
                        output_dir=export_dir,
                        model_type=args.model,
                        num_blocks=args.num_blocks if args.model == 'edsr' else None
                    )
            
            # Save sample predictions
            save_sample_images(
                model, 
                val_dataset, 
                device, 
                os.path.join(args.results_dir, 'predictions'),
                epoch + 1,
                num_samples=args.num_samples
            )
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(args.checkpoint_dir, f'{args.model}_epoch_{epoch+1}.pth')
            )
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}dB")
    
    # Final summary with best model path
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Best Validation PSNR: {best_psnr:.2f}dB")
    print(f"Best Model Saved: checkpoints/{args.model}_best.pth")
    if args.auto_export:
        print(f"Best Weights Exported: checkpoints/best_weights_export/")
        print(f"\nTo use the model for inference:")
        print(f"  python inference.py --weights checkpoints/best_weights_export/model_weights.pth --input your_image.png")
    else:
        print(f"\nTo export weights for inference:")
        print(f"  python export_model.py --checkpoint checkpoints/{args.model}_best.pth --output_dir exported_model")
        print(f"  python inference.py --weights exported_model/model_weights.pth --input your_image.png")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Super-Resolution Model on DIV2K')
    
    # Data paths
    parser.add_argument('--hr_train_dir', type=str, default='data/DIV2K_train_HR_gray',
                        help='Path to HR training images')
    parser.add_argument('--lr_train_dir', type=str, default='data/DIV2K_train_LR_gray',
                        help='Path to LR training images')
    parser.add_argument('--hr_val_dir', type=str, default='data/DIV2K_valid_HR_gray',
                        help='Path to HR validation images')
    parser.add_argument('--lr_val_dir', type=str, default='data/DIV2K_valid_LR_gray',
                        help='Path to LR validation images')
    
    # Model settings
    parser.add_argument('--model', type=str, default='edsr', choices=['srcnn', 'edsr'],
                        help='Model architecture')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='Number of residual blocks (for EDSR)')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--patch_size', type=int, default=96,
                        help='Training patch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, default=30,
                        help='Decay LR every N epochs')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'mse'],
                        help='Loss function')
    
    # Other settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample images to save each validation epoch')
    parser.add_argument('--auto_export', action='store_true',
                        help='Automatically export best weights for inference')
    
    args = parser.parse_args()
    main(args)