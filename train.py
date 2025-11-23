import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from dataset import DIV2KDataset, DIV2KValidationDataset
from models.srcnn_improved import SRCNN, EDSR_Lite
from utils import psnr, ssim, show_images, save_checkpoint, load_checkpoint, AverageMeter, adjust_learning_rate


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for lr, hr in pbar:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_psnr = psnr(hr, sr)
        
        losses.update(loss.item(), lr.size(0))
        psnr_meter.update(batch_psnr.item(), lr.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.6f}',
            'PSNR': f'{psnr_meter.avg:.2f}dB'
        })
    
    return losses.avg, psnr_meter.avg


def validate(model, val_loader, criterion, device, epoch=None, save_dir=None):
    """Validate the model and optionally save SR images."""
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    if save_dir and epoch is not None:
        epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_save_dir, exist_ok=True)
    else:
        epoch_save_dir = None
    
    with torch.no_grad():
        for idx, (lr, hr, filenames) in enumerate(tqdm(val_loader, desc="Validating")):
            lr, hr = lr.to(device), hr.to(device)
            
            try:
                sr = model(lr)
                
                # Ensure dimensions match exactly
                if sr.shape != hr.shape:
                    # Crop to minimum dimensions
                    min_h = min(sr.shape[2], hr.shape[2])
                    min_w = min(sr.shape[3], hr.shape[3])
                    sr = sr[:, :, :min_h, :min_w]
                    hr = hr[:, :, :min_h, :min_w]
                
                loss = criterion(sr, hr)
                batch_psnr = psnr(hr, sr)
                batch_ssim = ssim(hr, sr)
                
                losses.update(loss.item(), lr.size(0))
                psnr_meter.update(batch_psnr.item(), lr.size(0))
                ssim_meter.update(batch_ssim.item(), lr.size(0))
                
                if epoch_save_dir:
                    for i in range(sr.size(0)):
                        sr_img = sr[i].squeeze().cpu().clamp(0, 1).numpy()
                        sr_img = (sr_img * 255).astype('uint8')
                        
                        from PIL import Image
                        filename = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
                        save_path = os.path.join(epoch_save_dir, filename.replace('.png', '_SR.png'))
                        Image.fromarray(sr_img, mode='L').save(save_path)
            
            except RuntimeError as e:
                print(f"\nWarning: Skipping validation image {idx} due to error: {e}")
                continue
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg


def plot_training_curves(train_loss, val_loss, val_epochs, save_dir, loss_type='L1'):
    """Generate and save training loss plot."""
    plt.figure(figsize=(14, 10))
    
    all_epochs = list(range(1, len(train_loss) + 1))
    plt.plot(all_epochs, train_loss, linewidth=3.5, 
             label='Training Loss', color='#4696FF', alpha=0.85)
    
    if len(val_loss) > 0:
        plt.plot(val_epochs, val_loss, linewidth=3.5,
                label='Validation Loss', color='#FF6B6B', alpha=0.85, 
                marker='o', markersize=8)
    
    plt.xlabel('Epoch', fontsize=26, fontweight='bold')
    plt.ylabel(f'{loss_type} Loss', fontsize=26, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=30, fontweight='bold', pad=25)
    plt.legend(fontsize=22, loc='upper right', framealpha=0.95)
    plt.grid(True, alpha=0.35, linestyle='--', linewidth=2)
    plt.xlim(0, len(train_loss))
    
    if len(train_loss) > 0:
        y_max = max(max(train_loss), max(val_loss) if val_loss else max(train_loss))
        y_min = min(min(train_loss), min(val_loss) if val_loss else min(train_loss))
        plt.ylim(y_min * 0.95, y_max * 1.05)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if len(val_loss) > 0:
        final_val = val_loss[-1]
        final_epoch = val_epochs[-1]
        plt.annotate(f'Final: {final_val:.4f}', 
                    xy=(final_epoch, final_val),
                    xytext=(final_epoch - len(train_loss)*0.15, final_val + y_max*0.08),
                    fontsize=18,
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FF6B6B', 
                             alpha=0.75, edgecolor='none'),
                    color='white',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2.5))
    
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return loss_plot_path


def plot_psnr_curves(train_psnr, val_psnr, val_epochs, save_dir):
    """Generate and save PSNR plot."""
    plt.figure(figsize=(14, 10))
    
    all_epochs = list(range(1, len(train_psnr) + 1))
    plt.plot(all_epochs, train_psnr, linewidth=3.5,
             label='Training PSNR', color='#4696FF', alpha=0.85)
    
    if len(val_psnr) > 0:
        plt.plot(val_epochs, val_psnr, linewidth=3.5,
                label='Validation PSNR', color='#FF6B6B', alpha=0.85, 
                marker='o', markersize=8)
    
    plt.xlabel('Epoch', fontsize=26, fontweight='bold')
    plt.ylabel('PSNR (dB)', fontsize=26, fontweight='bold')
    plt.title('Training and Validation PSNR', fontsize=30, fontweight='bold', pad=25)
    plt.legend(fontsize=22, loc='lower right', framealpha=0.95)
    plt.grid(True, alpha=0.35, linestyle='--', linewidth=2)
    plt.xlim(0, len(train_psnr))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    psnr_plot_path = os.path.join(save_dir, 'training_psnr.png')
    plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return psnr_plot_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN auto-tuner for better performance on RTX GPUs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"Using device: {device}")
    
    # Create model-specific directories
    model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
    model_results_dir = os.path.join(args.results_dir, args.model)
    
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    os.makedirs(model_results_dir, exist_ok=True)
    
    sr_images_dir = os.path.join(model_results_dir, 'sr_images')
    os.makedirs(sr_images_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  MODEL: {args.model.upper()}")
    print(f"{'='*70}")
    print(f"  Checkpoints → {model_checkpoint_dir}")
    print(f"  Results     → {model_results_dir}")
    print(f"{'='*70}\n")
    
    # Training history
    train_loss_history = []
    val_loss_history = []
    train_psnr_history = []
    val_psnr_history = []
    epoch_numbers = []
    
    # Create datasets
    print("Loading datasets...")
    
    # Training set: first 600 images (reduced from 700)
    train_dataset = DIV2KDataset(
        hr_dir=args.hr_train_dir,
        lr_dir=args.lr_train_dir,
        patch_size=args.patch_size,
        augment=True,
        max_samples=600
    )
    
    # Test set: next 100 images (601-700)
    all_hr_files = sorted([f for f in os.listdir(args.hr_train_dir) if f.endswith('.png')])
    all_lr_files = sorted([f for f in os.listdir(args.lr_train_dir) if f.endswith('.png')])
    
    # Create temporary test directories with symlinks or subset
    test_hr_files = all_hr_files[600:700]
    test_lr_files = all_lr_files[600:700]
    
    # For simplicity, use validation dataset class for test set (no augmentation)
    class TestDataset(DIV2KValidationDataset):
        def __init__(self, hr_dir, lr_dir, file_list_hr, file_list_lr):
            self.hr_dir = hr_dir
            self.lr_dir = lr_dir
            self.hr_files = file_list_hr
            self.lr_files = file_list_lr
    
    test_dataset = TestDataset(args.hr_train_dir, args.lr_train_dir, test_hr_files, test_lr_files)
    
    # Validation set: separate validation directory
    val_dataset = DIV2KValidationDataset(
        hr_dir=args.hr_val_dir,
        lr_dir=args.lr_val_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Use 0 workers for validation to avoid multiprocessing issues
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
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
        current_lr = adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs)
        print(f"\nEpoch {epoch+1}/{args.epochs} - LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        train_loss_history.append(train_loss)
        train_psnr_history.append(train_psnr)
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            save_sr_dir = sr_images_dir if args.save_sr_images else None
            
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, criterion, device, 
                epoch=epoch+1, save_dir=save_sr_dir
            )
            print(f"Validation - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f}")
            
            val_loss_history.append(val_loss)
            val_psnr_history.append(val_psnr)
            epoch_numbers.append(epoch + 1)
            
            if args.save_sr_images:
                print(f"SR images → {os.path.join(sr_images_dir, f'epoch_{epoch+1}')}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(model_checkpoint_dir, f'{args.model}_best.pth')
                )
                print(f"★ New best model! PSNR: {val_psnr:.2f}dB")
            
            # Visualize sample
            if args.visualize:
                model.eval()
                with torch.no_grad():
                    lr_sample, hr_sample, filename = val_dataset[0]
                    lr_sample = lr_sample.unsqueeze(0).to(device)
                    hr_sample = hr_sample.unsqueeze(0).to(device)
                    sr_sample = model(lr_sample)
                    
                    show_images(
                        lr_sample[0],
                        sr_sample[0],
                        hr_sample[0],
                        save_path=os.path.join(model_results_dir, f'sample_epoch_{epoch+1}.png')
                    )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(model_checkpoint_dir, f'{args.model}_epoch_{epoch+1}.pth')
            )
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE - {args.model.upper()}")
    print(f"{'='*70}")
    print(f"  Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"{'='*70}\n")
    
    # Test on test set
    print("\n" + "="*70)
    print("  TESTING ON TEST SET")
    print("="*70)
    
    test_loss, test_psnr, test_ssim = validate(
        model, test_loader, criterion, device,
        epoch=None, save_dir=None
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  PSNR: {test_psnr:.2f} dB")
    print(f"  SSIM: {test_ssim:.4f}")
    print("="*70 + "\n")
    
    # Save training history
    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_psnr': train_psnr_history,
        'val_psnr': val_psnr_history,
        'val_epochs': epoch_numbers,
        'best_psnr': best_psnr,
        'test_loss': test_loss,
        'test_psnr': test_psnr,
        'test_ssim': test_ssim
    }
    
    history_path = os.path.join(model_results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history: {history_path}")
    
    # Generate plots
    print("\nGenerating training plots...")
    loss_type = 'L1' if args.loss == 'l1' else 'MSE'
    loss_plot = plot_training_curves(train_loss_history, val_loss_history, 
                                     epoch_numbers, model_results_dir, loss_type)
    print(f"✓ Loss plot: {loss_plot}")
    
    psnr_plot = plot_psnr_curves(train_psnr_history, val_psnr_history, 
                                 epoch_numbers, model_results_dir)
    print(f"✓ PSNR plot: {psnr_plot}")
    
    print(f"\n{'='*70}")
    print(f"  FILES SAVED TO: {model_results_dir}")
    print(f"{'='*70}")
    print(f"\nFinal Results Summary:")
    print(f"  Training:   {len(train_dataset)} images")
    print(f"  Test:       {len(test_dataset)} images - PSNR: {test_psnr:.2f} dB")
    print(f"  Validation: {len(val_dataset)} images - Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*70}\n")


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
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'edsr'],
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
    parser.add_argument('--loss', type=str, default='mse', choices=['l1', 'mse'],
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
    parser.add_argument('--visualize', action='store_true',
                        help='Save sample visualizations')
    parser.add_argument('--save_sr_images', action='store_true',
                        help='Save all SR validation images for each epoch')
    
    args = parser.parse_args()
    main(args)