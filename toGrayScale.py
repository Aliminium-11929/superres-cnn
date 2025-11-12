from PIL import Image
import os
from tqdm import tqdm

def convert_to_grayscale(src_dir, dst_dir):
    """
    Convert all PNG images in src_dir to grayscale and save to dst_dir.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    
    print(f"Converting {len(files)} images from {src_dir} to {dst_dir}")
    
    for fname in tqdm(files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        
        try:
            img = Image.open(src_path).convert('L')  # 'L' mode = grayscale
            img.save(dst_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == '__main__':
    # Define paths
    base_dir = 'data'
    
    # Training images
    hr_train_rgb = os.path.join(base_dir, 'DIV2K_train_HR')
    lr_train_rgb = os.path.join(base_dir, 'DIV2K_train_LR_bicubic', 'X2')
    
    hr_train_gray = os.path.join(base_dir, 'DIV2K_train_HR_gray')
    lr_train_gray = os.path.join(base_dir, 'DIV2K_train_LR_gray')
    
    # Validation images
    hr_val_rgb = os.path.join(base_dir, 'DIV2K_valid_HR')
    lr_val_rgb = os.path.join(base_dir, 'DIV2K_valid_LR_bicubic', 'X2')
    
    hr_val_gray = os.path.join(base_dir, 'DIV2K_valid_HR_gray')
    lr_val_gray = os.path.join(base_dir, 'DIV2K_valid_LR_gray')
    
    # Convert all datasets
    conversions = [
        (hr_train_rgb, hr_train_gray, "HR Training"),
        (lr_train_rgb, lr_train_gray, "LR Training"),
        (hr_val_rgb, hr_val_gray, "HR Validation"),
        (lr_val_rgb, lr_val_gray, "LR Validation")
    ]
    
    for src, dst, name in conversions:
        if os.path.exists(src):
            print(f"\n{name}:")
            convert_to_grayscale(src, dst)
        else:
            print(f"\nWarning: {name} directory not found: {src}")
    
    print("\n✓ Grayscale conversion complete!")
    print("\nGrayscale images saved to:")
    for _, dst, name in conversions:
        if os.path.exists(dst):
            count = len([f for f in os.listdir(dst) if f.endswith('.png')])
            print(f"  {dst} ({count} images)")