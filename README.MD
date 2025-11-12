# Super-Resolution CNN for DIV2K Dataset

A PyTorch implementation of deep learning models for **2× grayscale image super-resolution** trained on the DIV2K dataset.

---

## 🚀 Features

- **Multiple Architectures**: 
  - SRCNN: Simple baseline model
  - EDSR-Lite: Enhanced residual network with better performance
- **Optimized for DIV2K**: Custom dataset loader with augmentation
- **Training Features**:
  - Random cropping and data augmentation
  - Learning rate scheduling
  - Checkpoint saving and resuming
  - Validation with PSNR and SSIM metrics
- **Inference**: Test on single images or batches

---

## 📁 Project Structure

```
superres-cnn/
├── dataset.py              # DIV2K dataset loader
├── models/
│   └── srcnn_improved.py   # Model architectures (SRCNN, EDSR-Lite)
├── utils.py                # Utility functions (metrics, visualization)
├── train.py                # Training script
├── test.py                 # Inference script
├── save_predictions.py     # Batch prediction script
├── export_model.py         # Export trained weights
├── inference.py            # Ready-to-use inference (production)
├── toGrayScale.py          # Convert RGB images to grayscale
├── checkpoints/            # Saved model checkpoints
│   └── best_weights_export/ # Exported weights (if auto_export enabled)
├── results/
│   └── predictions/        # Sample predictions from training
│       └── epoch_N/        # Predictions for epoch N
└── data/
    ├── DIV2K_train_HR_gray/
    ├── DIV2K_train_LR_gray/
    ├── DIV2K_valid_HR_gray/
    └── DIV2K_valid_LR_gray/
```

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/superres-cnn.git
cd superres-cnn
```

### 2. Install Dependencies
```bash
pip install torch torchvision pillow numpy matplotlib tqdm
```

### 3. Download DIV2K Dataset
Download from: https://data.vision.ee.ethz.ch/cvl/DIV2K/

Required files:
- `DIV2K_train_HR.zip` (High-res training images)
- `DIV2K_train_LR_bicubic_X2.zip` (Low-res training images)
- `DIV2K_valid_HR.zip` (High-res validation images)
- `DIV2K_valid_LR_bicubic_X2.zip` (Low-res validation images)

Extract to:
```
data/
├── DIV2K_train_HR/
├── DIV2K_train_LR_bicubic/X2/
├── DIV2K_valid_HR/
└── DIV2K_valid_LR_bicubic/X2/
```

### 4. Convert to Grayscale
```bash
python toGrayScale.py
```

This will create grayscale versions in:
- `data/DIV2K_train_HR_gray/`
- `data/DIV2K_train_LR_gray/`
- `data/DIV2K_valid_HR_gray/`
- `data/DIV2K_valid_LR_gray/`

---

## 🏋️ Training

### Basic Training (EDSR-Lite) with Auto-Export
```bash
python train.py \
    --model edsr \
    --epochs 100 \
    --batch_size 16 \
    --patch_size 96 \
    --lr 1e-4 \
    --num_samples 5 \
    --auto_export
```

**Note**: 
- The training script automatically saves sample super-resolution predictions during validation to `results/predictions/epoch_N/`.
- With `--auto_export`, best weights are automatically exported to `checkpoints/best_weights_export/` for immediate use.

### Training SRCNN
```bash
python train.py \
    --model srcnn \
    --epochs 50 \
    --batch_size 32 \
    --patch_size 64 \
    --num_samples 10 \
    --auto_export
```

### Resume Training
```bash
python train.py \
    --model edsr \
    --resume checkpoints/edsr_epoch_50.pth
```

### Key Arguments
- `--model`: Model architecture (`srcnn` or `edsr`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--patch_size`: Training patch size (HR dimension)
- `--lr`: Learning rate
- `--lr_decay_epochs`: Decay LR every N epochs
- `--loss`: Loss function (`l1` or `mse`)
- `--num_blocks`: Number of residual blocks (EDSR only)
- `--num_samples`: Number of sample images to save each validation epoch
- `--auto_export`: Automatically export best weights for easy inference
- `--resume`: Resume from checkpoint

**Training Outputs**:
- Checkpoints saved to `checkpoints/`
- Sample predictions saved to `results/predictions/epoch_N/`
  - `N_filename_SR.png`: Super-resolution output
  - `N_filename_comparison.png`: Side-by-side LR-SR-HR comparison
- Best weights exported to `checkpoints/best_weights_export/` (if `--auto_export` used)

---

## 🎯 Using Trained Model for Production

After training, you have multiple options to use your model:

### Option 1: Auto-Exported Weights (Recommended)
If you trained with `--auto_export`, weights are ready to use:

```bash
# Single image
python inference.py \
    --weights checkpoints/best_weights_export/model_weights.pth \
    --input my_image.png

# Batch processing
python inference.py \
    --weights checkpoints/best_weights_export/model_weights.pth \
    --input_dir my_images/ \
    --output_dir upscaled_images/
```

### Option 2: Manual Export
Export weights from any checkpoint:

```bash
# Export best model
python export_model.py \
    --checkpoint checkpoints/edsr_best.pth \
    --model edsr \
    --output_dir exported_model

# Use exported weights
python inference.py \
    --weights exported_model/model_weights.pth \
    --input my_image.png
```

### Option 3: Export with Detailed Information
```bash
# Export with detailed weight values and statistics
python export_model.py \
    --checkpoint checkpoints/edsr_best.pth \
    --model edsr \
    --output_dir exported_model \
    --save_detailed
```

This creates:
- `model_weights.pth` - Clean weights for inference
- `model_config.json` - Model configuration
- `weights_info.txt` - Layer statistics and weight information
- `detailed_weights.txt` - Full weight values (if `--save_detailed`)

---

## 🚀 Production Inference Examples

### Single Image Super-Resolution
```bash
python inference.py \
    --weights checkpoints/best_weights_export/model_weights.pth \
    --input low_res_photo.png \
    --output high_res_photo.png
```

### Batch Processing
```bash
# Process entire folder
python inference.py \
    --weights checkpoints/best_weights_export/model_weights.pth \
    --input_dir low_res_images/ \
    --output_dir super_res_images/
```

### Using in Python Script
```python
from inference import SuperResolutionModel

# Initialize model
sr_model = SuperResolutionModel(
    weights_path='checkpoints/best_weights_export/model_weights.pth'
)

# Upscale single image
sr_image = sr_model.upscale('input.png', 'output_SR.png')

# Batch upscale
sr_model.upscale_batch('input_folder/', 'output_folder/')
```

---

## 🖼️ Saving Predictions

### During Training
Sample predictions are automatically saved every validation epoch to `results/predictions/epoch_N/`.

### Save All Validation Predictions
Generate super-resolution images for the entire validation set:

```bash
python save_predictions.py \
    --model edsr \
    --checkpoint checkpoints/edsr_best.pth \
    --save_comparisons
```

This will create:
- `predictions/SR_images/`: All super-resolution outputs
- `predictions/comparisons/`: Side-by-side comparison images (if `--save_comparisons` used)
- `predictions/metrics_summary.txt`: Per-image PSNR/SSIM metrics

---

## 🧪 Testing

### Test Single Image
```bash
python test.py \
    --model edsr \
    --checkpoint checkpoints/edsr_best.pth \
    --lr_image path/to/lr_image.png \
    --hr_image path/to/hr_image.png \
    --output_dir output/
```

### Test Folder
```bash
python test.py \
    --model edsr \
    --checkpoint checkpoints/edsr_best.pth \
    --lr_dir data/DIV2K_valid_LR_gray/ \
    --hr_dir data/DIV2K_valid_HR_gray/ \
    --output_dir output/
```

### Test Without Ground Truth
```bash
python test.py \
    --model edsr \
    --checkpoint checkpoints/edsr_best.pth \
    --lr_image path/to/lr_image.png \
    --output_dir output/
```

---

## 📊 Model Comparison

| Model | Parameters | PSNR (avg) | Training Time |
|-------|-----------|------------|---------------|
| SRCNN | ~8K | ~28-30 dB | Fast |
| EDSR-Lite (8 blocks) | ~300K | ~30-32 dB | Moderate |

*PSNR values are approximate and depend on training duration*

---

## 🎯 Expected Results

After training EDSR-Lite for 100 epochs:
- **PSNR**: 30-32 dB on DIV2K validation set
- **SSIM**: 0.85-0.90
- Visual quality: Sharp edges, reduced artifacts

---

## 🔧 Customization

### Adding New Models
Edit `models/srcnn_improved.py` and add your architecture:

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Your architecture here
    
    def forward(self, x):
        # Forward pass
        return output
```

### Custom Loss Functions
Modify `train.py`:
```python
# Example: Perceptual loss
criterion = PerceptualLoss()
```

### Data Augmentation
Edit `dataset.py` to add more augmentation techniques.

---

## 📈 Monitoring Training

Training outputs:
```
Epoch 1/100 - LR: 0.000100
Epoch 1: 100%|████████| 50/50 [00:45<00:00, loss=0.012345, PSNR=28.50dB]
Validation - Loss: 0.011234, PSNR: 29.15dB, SSIM: 0.8234
```

Visualizations saved in `results/` folder.

---

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--patch_size` (try 64)
- Use SRCNN instead of EDSR

### Slow Training
- Increase `--num_workers` for faster data loading
- Use smaller validation frequency: `--val_freq 10`
- Train on GPU (CUDA)

### Low PSNR
- Train for more epochs
- Use L1 loss instead of MSE: `--loss l1`
- Increase model capacity: `--num_blocks 16`

---

## 📚 References

- [SRCNN Paper](https://arxiv.org/abs/1501.00092)
- [EDSR Paper](https://arxiv.org/abs/1707.02921)
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

## 📝 License

MIT License - Feel free to use for research and educational purposes.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Color image super-resolution (RGB channels)
- Different upscaling factors (4×, 8×)
- Advanced architectures (RCAN, SwinIR)
- Perceptual and adversarial losses

---

## ✅ TODO

- [x] Baseline SRCNN implementation
- [x] EDSR-Lite architecture
- [x] DIV2K dataset integration
- [x] Training pipeline with checkpoints
- [x] PSNR and SSIM metrics
- [x] Weight export system
- [x] Production-ready inference script
- [ ] Color image super-resolution
- [ ] 4× upscaling support
- [ ] Perceptual loss implementation
- [ ] Web demo interface
- [ ] Model quantization for faster inference

---

## 📖 Complete Workflow Example

```bash
# 1. Prepare data
python toGrayScale.py

# 2. Train model with auto-export
python train.py --model edsr --epochs 100 --auto_export

# 3. Model is automatically ready for inference!
python inference.py \
    --weights checkpoints/best_weights_export/model_weights.pth \
    --input your_image.png

# 4. (Optional) Export weights manually for specific checkpoint
python export_model.py \
    --checkpoint checkpoints/edsr_epoch_50.pth \
    --model edsr \
    --output_dir my_exported_model

# 5. Use exported weights
python inference.py \
    --weights my_exported_model/model_weights.pth \
    --input_dir test_images/
```

---

**Happy Training! 🚀**