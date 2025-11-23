import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path).convert('L')
    return np.array(img)

def compare_epochs(sr_dir, image_name, epochs_to_compare, hr_dir=None):
    """
    Compare SR images from different epochs side by side.
    
    Args:
        sr_dir: Directory containing epoch folders (e.g., results/sr_images/)
        image_name: Name of the image to compare
        epochs_to_compare: List of epoch numbers to compare
        hr_dir: Optional directory containing HR ground truth images
    """
    n_epochs = len(epochs_to_compare)
    has_hr = hr_dir is not None and os.path.exists(hr_dir)
    
    # Determine subplot layout
    n_cols = n_epochs + (1 if has_hr else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    # Load and display SR images from each epoch
    for idx, epoch in enumerate(epochs_to_compare):
        epoch_dir = os.path.join(sr_dir, f'epoch_{epoch}')
        sr_path = os.path.join(epoch_dir, image_name.replace('.png', '_SR.png'))
        
        if os.path.exists(sr_path):
            sr_img = load_image(sr_path)
            axes[idx].imshow(sr_img, cmap='gray', vmin=0, vmax=255)
            axes[idx].set_title(f'Epoch {epoch}', fontsize=14)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'Not found\n{sr_path}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'Epoch {epoch}', fontsize=14)
            axes[idx].axis('off')
    
    # Display HR ground truth if available
    if has_hr:
        hr_path = os.path.join(hr_dir, image_name)
        if os.path.exists(hr_path):
            hr_img = load_image(hr_path)
            axes[-1].imshow(hr_img, cmap='gray', vmin=0, vmax=255)
            axes[-1].set_title('Ground Truth (HR)', fontsize=14)
            axes[-1].axis('off')
    
    plt.suptitle(f'Super-Resolution Progression: {image_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save comparison
    output_dir = os.path.join(sr_dir, 'comparisons')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{image_name.replace(".png", "")}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    plt.show()

def create_grid_comparison(sr_dir, num_images=4, epochs_to_compare=None, hr_dir=None):
    """
    Create a grid showing multiple images across epochs.
    
    Args:
        sr_dir: Directory containing epoch folders
        num_images: Number of different images to show
        epochs_to_compare: List of epoch numbers (if None, auto-detect)
        hr_dir: Optional HR directory
    """
    # Auto-detect epochs if not specified
    if epochs_to_compare is None:
        epoch_dirs = [d for d in os.listdir(sr_dir) if d.startswith('epoch_')]
        epochs_to_compare = sorted([int(d.split('_')[1]) for d in epoch_dirs])
    
    if not epochs_to_compare:
        print("No epochs found!")
        return
    
    # Get list of available images
    first_epoch_dir = os.path.join(sr_dir, f'epoch_{epochs_to_compare[0]}')
    available_images = [f for f in os.listdir(first_epoch_dir) if f.endswith('_SR.png')]
    
    if not available_images:
        print("No images found!")
        return
    
    # Select images to display
    step = max(1, len(available_images) // num_images)
    selected_images = available_images[::step][:num_images]
    
    # Create grid
    n_rows = len(selected_images)
    n_cols = len(epochs_to_compare) + (1 if hr_dir else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, img_name in enumerate(selected_images):
        # Original filename without _SR suffix
        original_name = img_name.replace('_SR.png', '.png')
        
        # Display SR images from each epoch
        for col_idx, epoch in enumerate(epochs_to_compare):
            epoch_dir = os.path.join(sr_dir, f'epoch_{epoch}')
            sr_path = os.path.join(epoch_dir, img_name)
            
            if os.path.exists(sr_path):
                sr_img = load_image(sr_path)
                axes[row_idx, col_idx].imshow(sr_img, cmap='gray', vmin=0, vmax=255)
            
            axes[row_idx, col_idx].axis('off')
            
            # Add title to top row
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'Epoch {epoch}', fontsize=12)
        
        # Display HR if available
        if hr_dir:
            hr_path = os.path.join(hr_dir, original_name)
            if os.path.exists(hr_path):
                hr_img = load_image(hr_path)
                axes[row_idx, -1].imshow(hr_img, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, -1].axis('off')
            
            if row_idx == 0:
                axes[row_idx, -1].set_title('HR Target', fontsize=12)
        
        # Add image name to the left
        axes[row_idx, 0].text(-0.1, 0.5, original_name, 
                             transform=axes[row_idx, 0].transAxes,
                             rotation=90, ha='right', va='center', fontsize=10)
    
    plt.suptitle('Training Progress: SR Quality Across Epochs', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save grid
    output_dir = os.path.join(sr_dir, 'comparisons')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_progress_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid comparison to: {output_path}")
    
    plt.show()

def main(args):
    if args.mode == 'single':
        if not args.image_name:
            print("Please specify --image_name for single image comparison")
            return
        
        compare_epochs(
            args.sr_dir,
            args.image_name,
            args.epochs,
            args.hr_dir
        )
    
    elif args.mode == 'grid':
        create_grid_comparison(
            args.sr_dir,
            num_images=args.num_images,
            epochs_to_compare=args.epochs if args.epochs else None,
            hr_dir=args.hr_dir
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare SR images across training epochs')
    
    parser.add_argument('--mode', type=str, default='grid', choices=['single', 'grid'],
                        help='Comparison mode: single image or grid of multiple images')
    parser.add_argument('--sr_dir', type=str, default='results/sr_images',
                        help='Directory containing epoch folders')
    parser.add_argument('--hr_dir', type=str, default='data/DIV2K_valid_HR_gray',
                        help='Directory containing HR ground truth images')
    parser.add_argument('--image_name', type=str, default='',
                        help='Image filename to compare (for single mode)')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='List of epoch numbers to compare (e.g., --epochs 5 10 20 50)')
    parser.add_argument('--num_images', type=int, default=4,
                        help='Number of images to show in grid mode')
    
    args = parser.parse_args()
    main(args)