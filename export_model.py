import torch
import json
import argparse
import os
from collections import OrderedDict

from models.srcnn_improved import SRCNN, EDSR_Lite


def export_model_weights(checkpoint_path, output_dir, model_type='edsr', num_blocks=8):
    """
    Export trained model weights in multiple formats:
    1. Clean state dict (weights only, no optimizer)
    2. Human-readable text format
    3. Model configuration JSON
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
    else:
        state_dict = checkpoint
        epoch = 'unknown'
        loss = 'unknown'
    
    print(f"Checkpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss}")
    print(f"  Model Type: {model_type}")
    
    # 1. Save clean weights (for inference)
    clean_weights_path = os.path.join(output_dir, 'model_weights.pth')
    torch.save({
        'model_state_dict': state_dict,
        'model_type': model_type,
        'num_blocks': num_blocks if model_type == 'edsr' else None,
        'epoch': epoch,
        'loss': loss
    }, clean_weights_path)
    print(f"\n✓ Saved clean weights: {clean_weights_path}")
    
    # 2. Save human-readable weight information
    weights_txt_path = os.path.join(output_dir, 'weights_info.txt')
    with open(weights_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL WEIGHTS INFORMATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model Type: {model_type.upper()}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Loss: {loss}\n")
        if model_type == 'edsr':
            f.write(f"Number of Blocks: {num_blocks}\n")
        f.write(f"\n{'='*80}\n\n")
        
        f.write("LAYER STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        total_params = 0
        for name, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            param_shape = tuple(param.shape)
            param_mean = param.mean().item()
            param_std = param.std().item()
            param_min = param.min().item()
            param_max = param.max().item()
            
            f.write(f"Layer: {name}\n")
            f.write(f"  Shape: {param_shape}\n")
            f.write(f"  Parameters: {param_count:,}\n")
            f.write(f"  Mean: {param_mean:.6f}\n")
            f.write(f"  Std Dev: {param_std:.6f}\n")
            f.write(f"  Min: {param_min:.6f}\n")
            f.write(f"  Max: {param_max:.6f}\n")
            f.write("\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"TOTAL PARAMETERS: {total_params:,}\n")
        f.write(f"{'='*80}\n")
    
    print(f"✓ Saved weights info: {weights_txt_path}")
    
    # 3. Save model configuration
    config = {
        'model_type': model_type,
        'num_blocks': num_blocks if model_type == 'edsr' else None,
        'num_channels': 1,
        'scale_factor': 2,
        'epoch': int(epoch) if isinstance(epoch, int) else None,
        'loss': float(loss) if isinstance(loss, (int, float)) else None,
        'total_parameters': sum(p.numel() for p in state_dict.values()),
        'layers': {name: list(param.shape) for name, param in state_dict.items()}
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved model config: {config_path}")
    
    # 4. Print weight summary to console
    print(f"\n{'='*80}")
    print("WEIGHT SUMMARY")
    print(f"{'='*80}")
    print(f"Total Parameters: {sum(p.numel() for p in state_dict.values()):,}")
    print(f"Number of Layers: {len(state_dict)}")
    print(f"\nKey Layers:")
    for i, (name, param) in enumerate(list(state_dict.items())[:5]):
        print(f"  {name}: {tuple(param.shape)}")
    if len(state_dict) > 5:
        print(f"  ... ({len(state_dict) - 5} more layers)")
    print(f"{'='*80}\n")
    
    return clean_weights_path, config_path


def print_weight_values(checkpoint_path, output_file=None):
    """
    Print actual weight values in a readable format.
    Warning: This can generate very large files for big models.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    output = []
    output.append("="*80)
    output.append("DETAILED WEIGHT VALUES")
    output.append("="*80 + "\n")
    
    for name, param in state_dict.items():
        output.append(f"\n{'='*80}")
        output.append(f"Layer: {name}")
        output.append(f"Shape: {tuple(param.shape)}")
        output.append(f"{'='*80}\n")
        
        # Flatten and show first/last few values
        flat = param.flatten()
        if len(flat) <= 20:
            # Show all values for small layers
            output.append(str(flat.numpy()))
        else:
            # Show first and last 10 values for large layers
            output.append("First 10 values:")
            output.append(str(flat[:10].numpy()))
            output.append("\n...")
            output.append(f"\n(... {len(flat) - 20} values omitted ...)")
            output.append("\n...\n")
            output.append("Last 10 values:")
            output.append(str(flat[-10:].numpy()))
        output.append("\n")
    
    output_text = "\n".join(output)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"✓ Detailed weights saved to: {output_file}")
    else:
        print(output_text)


def main(args):
    print("\n" + "="*80)
    print("MODEL WEIGHT EXPORTER")
    print("="*80 + "\n")
    
    # Export model weights
    clean_weights_path, config_path = export_model_weights(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_type=args.model,
        num_blocks=args.num_blocks
    )
    
    # Optionally save detailed weight values
    if args.save_detailed:
        print("\nExporting detailed weight values...")
        detailed_path = os.path.join(args.output_dir, 'detailed_weights.txt')
        print_weight_values(args.checkpoint, detailed_path)
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"\nExported files in: {args.output_dir}/")
    print(f"  - model_weights.pth      (Clean weights for inference)")
    print(f"  - model_config.json      (Model configuration)")
    print(f"  - weights_info.txt       (Weight statistics)")
    if args.save_detailed:
        print(f"  - detailed_weights.txt   (Detailed weight values)")
    print(f"\nTo use these weights for inference, run:")
    print(f"  python inference.py --weights {clean_weights_path} --input your_image.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export trained model weights for inference'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default='edsr',
                        choices=['srcnn', 'edsr'],
                        help='Model architecture')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='Number of residual blocks (for EDSR)')
    parser.add_argument('--output_dir', type=str, default='exported_model',
                        help='Directory to save exported weights')
    parser.add_argument('--save_detailed', action='store_true',
                        help='Save detailed weight values (creates large file)')
    
    args = parser.parse_args()
    main(args)