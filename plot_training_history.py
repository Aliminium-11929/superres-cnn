"""
Generate training loss plot from training history JSON file.
If you don't have a history file, you can create one manually or use dummy data.
"""

import matplotlib.pyplot as plt
import json
import argparse
import os
import numpy as np


def plot_training_loss(history_path, output_path='training_loss.png'):
    """
    Generate training loss plot from history JSON file.
    """
    # Load training history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        train_loss = history['train_loss']
        val_loss = history.get('val_loss', [])
        val_epochs = history.get('val_epochs', [])
        
        print(f"Loaded training history from: {history_path}")
        print(f"Total epochs: {len(train_loss)}")
        print(f"Validation points: {len(val_loss)}")
    else:
        print(f"History file not found: {history_path}")
        print("Generating sample data for demonstration...")
        
        # Generate sample data
        num_epochs = 100
        train_loss = generate_sample_loss_curve(num_epochs)
        val_loss = generate_sample_loss_curve(num_epochs // 5, offset=0.002)
        val_epochs = list(range(5, num_epochs + 1, 5))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    all_epochs = list(range(1, len(train_loss) + 1))
    plt.plot(all_epochs, train_loss, linewidth=2.5, 
             label='Training Loss', color='#4696FF', alpha=0.8)
    
    # Plot validation loss
    if len(val_loss) > 0:
        plt.plot(val_epochs, val_loss, linewidth=2.5,
                label='Validation Loss', color='#FF6B6B', alpha=0.8, 
                marker='o', markersize=6)
    
    # Styling
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('L1 Loss', fontsize=22, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=26, fontweight='bold', pad=20)
    plt.legend(fontsize=18, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlim(0, len(train_loss))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add final loss annotation
    if len(val_loss) > 0:
        final_val = val_loss[-1]
        final_epoch = val_epochs[-1]
        plt.annotate(f'Final: {final_val:.4f}', 
                    xy=(final_epoch, final_val),
                    xytext=(final_epoch - 15, final_val + max(val_loss) * 0.1),
                    fontsize=16,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF6B6B', 
                             alpha=0.7, edgecolor='none'),
                    color='white',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Training loss plot saved: {output_path}")


def generate_sample_loss_curve(num_points, start=0.05, end=0.003, offset=0):
    """
    Generate realistic-looking loss curve with exponential decay and noise.
    """
    # Exponential decay
    epochs = np.arange(num_points)
    loss = start * np.exp(-epochs / (num_points / 3)) + end + offset
    
    # Add realistic noise
    noise = np.random.normal(0, 0.0005, num_points)
    loss = loss + noise
    
    # Smooth slightly
    window = 3
    loss_smooth = np.convolve(loss, np.ones(window)/window, mode='same')
    
    return loss_smooth.tolist()


def create_manual_history_file(output_path='training_history.json', num_epochs=100):
    """
    Create a sample training history file with realistic data.
    Modify the values to match your actual training results.
    """
    print(f"\nCreating sample training history file: {output_path}")
    
    train_loss = generate_sample_loss_curve(num_epochs, start=0.05, end=0.002)
    val_loss = generate_sample_loss_curve(num_epochs // 5, start=0.055, end=0.0025)
    val_epochs = list(range(5, num_epochs + 1, 5))
    
    # Generate PSNR values (increasing over time)
    train_psnr = [25 + 8 * (1 - np.exp(-i / 30)) + np.random.normal(0, 0.3) 
                  for i in range(num_epochs)]
    val_psnr = [25 + 8 * (1 - np.exp(-i / 30)) + np.random.normal(0, 0.5) 
                for i in val_epochs]
    
    history = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_psnr': train_psnr,
        'val_psnr': val_psnr,
        'val_epochs': val_epochs
    }
    
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Sample history file created: {output_path}")
    print("\nIMPORTANT: Replace this with your actual training data!")
    print("You can edit the JSON file directly or re-train with the updated train.py")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate training loss plot for poster'
    )
    parser.add_argument(
        '--history', 
        type=str, 
        default='results/training_history.json',
        help='Path to training history JSON file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='training_loss.png',
        help='Output path for loss plot'
    )
    parser.add_argument(
        '--create-sample', 
        action='store_true',
        help='Create a sample history file with dummy data'
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        history_path = create_manual_history_file()
        plot_training_loss(history_path, args.output)
    else:
        plot_training_loss(args.history, args.output)


if __name__ == '__main__':
    main()