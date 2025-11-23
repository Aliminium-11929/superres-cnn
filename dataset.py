import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    """
    DIV2K dataset loader for super-resolution training.
    Handles grayscale LR-HR image pairs.
    """
    def __init__(self, hr_dir, lr_dir, patch_size=96, augment=True, max_samples=None):
        """
        Args:
            hr_dir: Path to high-resolution grayscale images
            lr_dir: Path to low-resolution grayscale images
            patch_size: Size of random crops for training
            augment: Whether to apply data augmentation
            max_samples: Maximum number of samples to use (for train/test split)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.augment = augment
        
        # Get all image filenames
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
        assert len(self.hr_files) == len(self.lr_files), "Mismatch in HR and LR image counts"
        
        # Limit samples if specified
        if max_samples is not None:
            self.hr_files = self.hr_files[:max_samples]
            self.lr_files = self.lr_files[:max_samples]
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load images
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        
        hr_img = Image.open(hr_path).convert('L')  # Grayscale
        lr_img = Image.open(lr_path).convert('L')
        
        # Convert to tensors
        hr_tensor = transforms.ToTensor()(hr_img)
        lr_tensor = transforms.ToTensor()(lr_img)
        
        # Random crop (ensure LR and HR patches align)
        if self.patch_size > 0:
            hr_h, hr_w = hr_tensor.shape[1], hr_tensor.shape[2]
            
            # Calculate valid crop region
            top = torch.randint(0, hr_h - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, hr_w - self.patch_size + 1, (1,)).item()
            
            hr_tensor = hr_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            
            # Corresponding LR patch (2x downscaled coordinates)
            lr_top, lr_left = top // 2, left // 2
            lr_size = self.patch_size // 2
            lr_tensor = lr_tensor[:, lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                hr_tensor = torch.flip(hr_tensor, [2])
                lr_tensor = torch.flip(lr_tensor, [2])
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                hr_tensor = torch.flip(hr_tensor, [1])
                lr_tensor = torch.flip(lr_tensor, [1])
            
            # Random rotation (90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                hr_tensor = torch.rot90(hr_tensor, k, [1, 2])
                lr_tensor = torch.rot90(lr_tensor, k, [1, 2])
        
        return lr_tensor, hr_tensor


class DIV2KValidationDataset(Dataset):
    """
    DIV2K validation dataset - no augmentation, full images.
    """
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
        # Ensure matching files
        hr_names = set([f.split('.')[0].split('x')[0] for f in self.hr_files])
        lr_names = set([f.split('.')[0].split('x')[0] for f in self.lr_files])
        common_names = sorted(hr_names.intersection(lr_names))
        
        # Filter to only common files
        self.hr_files = [f for f in self.hr_files if f.split('.')[0].split('x')[0] in common_names]
        self.lr_files = [f for f in self.lr_files if f.split('.')[0].split('x')[0] in common_names]
        
        # Double check they match
        assert len(self.hr_files) == len(self.lr_files), \
            f"Mismatch: {len(self.hr_files)} HR files vs {len(self.lr_files)} LR files"
        
        print(f"Validation dataset: {len(self.hr_files)} image pairs")
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.hr_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.hr_files)}")
            
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        
        hr_img = Image.open(hr_path).convert('L')
        lr_img = Image.open(lr_path).convert('L')
        
        hr_tensor = transforms.ToTensor()(hr_img)
        lr_tensor = transforms.ToTensor()(lr_img)
        
        return lr_tensor, hr_tensor, self.hr_files[idx]