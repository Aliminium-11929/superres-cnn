import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_srcnn_architecture(output_path='architecture.png'):
    """
    Create Enhanced SRCNN architecture diagram for poster.
    """
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Colors
    color_input = '#FFC864'      # Orange
    color_bicubic = '#B4D4FF'    # Light blue
    color_conv = '#64C8FF'       # Blue
    color_output = '#96FF96'     # Green
    
    # Layer definitions: [y_position, label, color, width, height]
    layers = [
        (15, 'LR Input\n1×H×W', color_input, 2.5, 0.7),
        (13.5, 'Bicubic Upsample\n2×', color_bicubic, 2.5, 0.7),
        (11.5, 'Conv2D 9×9\n64 filters\nReLU', color_conv, 3, 1.2),
        (9.5, 'Conv2D 5×5\n64 filters\nReLU', color_conv, 3, 1.2),
        (7.5, 'Conv2D 5×5\n32 filters\nReLU', color_conv, 3, 1.2),
        (5.5, 'Conv2D 5×5\n1 filter', color_conv, 3, 1.2),
        (3.5, 'SR Output\n1×2H×2W', color_output, 2.5, 0.7),
    ]
    
    # Draw layers
    boxes_centers = []
    for y, label, color, width, height in layers:
        x = 5 - width/2
        box = FancyBboxPatch((x, y - height/2), width, height,
                            boxstyle="round,pad=0.05",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2.5)
        ax.add_patch(box)
        ax.text(5, y, label, ha='center', va='center', 
               fontsize=15, fontweight='bold', multialignment='center')
        boxes_centers.append((5, y, height/2))
    
    # Draw arrows between layers
    for i in range(len(boxes_centers)-1):
        x1, y1, h1 = boxes_centers[i]
        x2, y2, h2 = boxes_centers[i+1]
        arrow = FancyArrowPatch((x1, y1 - h1 - 0.05), (x2, y2 + h2 + 0.05),
                               arrowstyle='->', mutation_scale=35,
                               linewidth=4, color='black')
        ax.add_patch(arrow)
    
    # Add stage labels on the right
    ax.text(8.2, 11.5, 'Feature\nExtraction', fontsize=13, 
           fontweight='bold', ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.6))
    
    ax.text(8.2, 8.5, 'Non-linear\nMapping', fontsize=13, 
           fontweight='bold', ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.6))
    
    ax.text(8.2, 5.5, 'Reconstruction', fontsize=13, 
           fontweight='bold', ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.6))
    
    # Title
    ax.text(5, 15.8, 'Enhanced SRCNN Architecture', 
           ha='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Architecture diagram saved: {output_path}")


if __name__ == '__main__':
    create_srcnn_architecture('architecture.png')
    print("\nArchitecture diagram ready for poster!")