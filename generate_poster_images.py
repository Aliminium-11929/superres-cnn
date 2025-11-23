import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

def generate_training_loss_plot(save_path='training_loss.png'):
    """
    Generate a professional training loss plot.
    Replace with your actual training data for real results.
    """
    # Simulated training data (replace with actual data from your training)
    epochs = np.arange(1, 101)
    
    # Simulated loss curves (exponential decay with noise)
    train_loss = 0.05 * np.exp(-epochs/25) + 0.002 + np.random.normal(0, 0.0005, len(epochs))
    val_loss = 0.055 * np.exp(-epochs/28) + 0.0025 + np.random.normal(0, 0.0008, len(epochs))
    
    # Smooth the curves slightly
    from scipy.ndimage import gaussian_filter1d
    train_loss_smooth = gaussian_filter1d(train_loss, sigma=2)
    val_loss_smooth = gaussian_filter1d(val_loss, sigma=2)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot
    plt.plot(epochs, train_loss_smooth, linewidth=3, label='Training Loss', color='#4696FF', alpha=0.9)
    plt.plot(epochs, val_loss_smooth, linewidth=3, label='Validation Loss', color='#FF6B6B', alpha=0.9)
    
    # Styling
    plt.xlabel('Epoch', fontsize=24, fontweight='bold')
    plt.ylabel('L1 Loss', fontsize=24, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=28, fontweight='bold', pad=20)
    plt.legend(fontsize=20, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    
    # Set limits and ticks
    plt.xlim(0, 100)
    plt.ylim(0, max(train_loss_smooth.max(), val_loss_smooth.max()) * 1.1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add annotations
    final_train = train_loss_smooth[-1]
    final_val = val_loss_smooth[-1]
    plt.annotate(f'Final: {final_train:.4f}', 
                xy=(100, final_train), 
                xytext=(85, final_train + 0.005),
                fontsize=16, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#4696FF', alpha=0.7, edgecolor='none'),
                color='white',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#4696FF', lw=2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved training loss plot: {save_path}")


def generate_architecture_diagram(save_path='architecture_diagram.png'):
    """
    Generate a detailed architecture diagram.
    Note: The LaTeX version uses TikZ which is better for posters.
    This creates a backup PNG version.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors
    color_input = '#FFC864'
    color_conv = '#64C8FF'
    color_res = '#A0D4FF'
    color_output = '#96FF96'
    
    # Layer definitions [y_position, label, color, width]
    layers = [
        (13, 'LR Input\n1×H×W', color_input, 2.5),
        (11.5, 'Bicubic Upsample\n2×', color_conv, 2.5),
        (10, 'Conv2D 9×9\n64 filters', color_conv, 2.5),
        (8.5, 'ReLU', color_conv, 2),
        (7, '8× Residual Blocks\n(Conv 3×3 + Skip)', color_res, 3),
        (5, 'Conv2D 3×3\n64 filters', color_conv, 2.5),
        (3.5, 'Pixel Shuffle\nUpscale 2×', color_conv, 2.5),
        (2, 'Conv2D 3×3\n1 filter', color_conv, 2.5),
        (0.5, 'SR Output\n1×2H×2W', color_output, 2.5),
    ]
    
    # Draw layers
    boxes = []
    for y, label, color, width in layers:
        x = 5 - width/2
        box = FancyBboxPatch((x, y-0.3), width, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(5, y, label, ha='center', va='center', 
               fontsize=14, fontweight='bold')
        boxes.append((5, y))
    
    # Draw arrows between layers
    for i in range(len(boxes)-1):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i+1]
        arrow = FancyArrowPatch((x1, y1-0.35), (x2, y2+0.35),
                               arrowstyle='->', mutation_scale=30,
                               linewidth=3, color='black')
        ax.add_patch(arrow)
    
    # Draw skip connection (from conv1 to conv before upsample)
    skip_arrow = FancyArrowPatch((6.5, 10), (6.5, 5),
                                arrowstyle='->', mutation_scale=25,
                                linewidth=2.5, color='red',
                                linestyle='dashed')
    ax.add_patch(skip_arrow)
    ax.text(7.5, 7.5, 'Skip\nConnection', fontsize=12, 
           color='red', fontweight='bold', ha='center')
    
    # Title
    ax.text(5, 13.8, 'EDSR-Lite Architecture', 
           ha='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved architecture diagram: {save_path}")


def load_and_prepare_comparison_images():
    """
    Instructions for preparing comparison images for the poster.
    """
    print("\n" + "="*60)
    print("PREPARING COMPARISON IMAGES FOR POSTER")
    print("="*60)
    print("\nYou need to generate these images from your trained model:")
    print("\n1. input_lr.png - Low resolution input image")
    print("2. output_epoch11.png - CNN output at epoch 11")
    print("3. target_hr.png - High resolution ground truth")
    print("\nTo generate these, run:")
    print("\n  python test.py \\")
    print("      --model edsr \\")
    print("      --checkpoint checkpoints/edsr_epoch_11.pth \\")
    print("      --lr_image data/DIV2K_valid_LR_gray/0801.png \\")
    print("      --hr_image data/DIV2K_valid_HR_gray/0801.png \\")
    print("      --output_dir poster_images/")
    print("\nThen rename/crop the outputs as needed for the poster.")
    print("="*60 + "\n")


def create_sample_comparison():
    """
    Create sample comparison images with text (for layout testing only).
    Replace with actual images from your model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    labels = ['LR Input\n(Bicubic)', 'CNN Output\n(Epoch 11)', 'HR Ground Truth']
    colors = ['#FFE5B4', '#B4E5FF', '#B4FFB4']
    
    for ax, label, color in zip(axes, labels, colors):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
        ax.text(0.5, 0.5, label, ha='center', va='center', 
               fontsize=24, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created sample comparison layout: sample_comparison.png")


if __name__ == '__main__':
    print("Generating poster graphics...\n")
    
    # Generate training loss plot
    try:
        from scipy.ndimage import gaussian_filter1d
        generate_training_loss_plot()
    except ImportError:
        print("! Install scipy for smooth loss curves: pip install scipy")
        # Generate without smoothing
        generate_training_loss_plot()
    
    # Generate architecture diagram
    generate_architecture_diagram()
    
    # Create sample comparison (for layout testing)
    create_sample_comparison()
    
    # Print instructions for real images
    load_and_prepare_comparison_images()
    
    print("\n✓ All graphics generated successfully!")
    print("\nNext steps:")
    print("1. Generate real comparison images using your trained model")
    print("2. Compile the poster: pdflatex poster.tex")
    print("3. If using Overleaf, upload all image files to the project")