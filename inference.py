#!/usr/bin/env python3
"""
Ready-to-use Super-Resolution Inference Script
Simply loads pre-trained weights and applies super-resolution to any image.
"""

import torch
import argparse
import os
import json
from PIL import Image
import torchvision.transforms as transforms
import time

from models.srcnn_improved import SRCNN, EDSR_Lite


class SuperResolutionModel:
    """
    Ready-to-use super-resolution model.
    Loads pre-trained weights and provides a simple interface.
    """
    def __init__(self, weights_path, device='auto'):
        """
        Initialize the super-resolution model.
        
        Args:
            weights_path: Path to model_weights.pth file
            device: 'auto', 'cuda', or 'cpu'
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load weights and config
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Extract model info
        self.model_type = checkpoint.get('model_type', 'edsr')
        self.num_blocks = checkpoint.get('num_blocks', 8)
        self.state_dict = checkpoint['model_state_dict']
        
        # Load configuration if available
        config_path = os.path.join(os.path.dirname(weights_path), 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded model config from: {config_path}")
        else:
            self.config = None
        
        # Create model
        if self.model_type == 'srcnn':
            self.model = SRCNN()
        elif self.model_type == 'edsr':
            self.model = EDSR_Lite(num_blocks=self.num_blocks)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel Information:")
        print(f"  Type: {self.model_type.upper()}")
        if self.model_type == 'edsr':
            print(f"  Residual Blocks: {self.num_blocks}")
        print(f"  Total Parameters: {total_params:,}")
        if self.config and self.config.get('epoch'):
            print(f"  Trained Epochs: {self.config['epoch']}")
        print()
    
    def upscale(self, input_image, output_path=None):
        """
        Upscale a single image.
        
        Args:
            input_image: PIL Image or path to image file
            output_path: Optional path to save the result
        
        Returns:
            PIL Image of the super-resolved result
        """
        # Load image if path is provided
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Image not found: {input_image}")
            input_image = Image.open(input_image)
        
        # Convert to grayscale
        input_image = input_image.convert('L')
        
        # Convert to tensor
        img_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            sr_tensor = self.model(img_tensor)
        inference_time = time.time() - start_time
        
        # Convert back to PIL Image
        sr_array = sr_tensor.squeeze().cpu().clamp(0, 1).numpy()
        sr_array = (sr_array * 255).astype('uint8')
        sr_image = Image.fromarray(sr_array, mode='L')
        
        # Save if output path provided
        if output_path:
            sr_image.save(output_path)
            print(f"Saved super-resolution image to: {output_path}")
        
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Input size: {input_image.size}")
        print(f"Output size: {sr_image.size}")
        
        return sr_image
    
    def upscale_batch(self, input_dir, output_dir, extensions=['.png', '.jpg', '.jpeg']):
        """
        Upscale all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            extensions: List of file extensions to process
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images from {input_dir}")
        print(f"Saving to {output_dir}\n")
        
        total_time = 0
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('.', '_SR.')
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"[{i}/{len(image_files)}] Processing: {filename}")
            
            start_time = time.time()
            self.upscale(input_path, output_path)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print()
        
        print(f"{'='*60}")
        print(f"Batch processing complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(image_files):.4f} seconds")
        print(f"{'='*60}\n")


def main(args):
    print("\n" + "="*60)
    print("SUPER-RESOLUTION INFERENCE")
    print("="*60 + "\n")
    
    # Initialize model
    sr_model = SuperResolutionModel(
        weights_path=args.weights,
        device=args.device
    )
    
    # Process single image or batch
    if args.input:
        # Single image
        output_path = args.output if args.output else args.input.replace('.', '_SR.')
        sr_model.upscale(args.input, output_path)
        
    elif args.input_dir:
        # Batch processing
        output_dir = args.output_dir if args.output_dir else args.input_dir + '_SR'
        sr_model.upscale_batch(args.input_dir, output_dir)
        
    else:
        print("Error: Please specify either --input or --input_dir")
        return
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Super-Resolution Inference - Apply trained model to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --weights exported_model/model_weights.pth --input image.png
  
  # Custom output path
  python inference.py --weights exported_model/model_weights.pth --input image.png --output result.png
  
  # Batch processing
  python inference.py --weights exported_model/model_weights.pth --input_dir test_images/
  
  # Use CPU only
  python inference.py --weights exported_model/model_weights.pth --input image.png --device cpu
        """
    )
    
    # Model settings
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model_weights.pth file')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    
    # Input/Output
    parser.add_argument('--input', type=str, default='',
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='',
                        help='Path to output image (default: input_SR.ext)')
    parser.add_argument('--input_dir', type=str, default='',
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save output images (default: input_dir_SR)')
    
    args = parser.parse_args()
    main(args)