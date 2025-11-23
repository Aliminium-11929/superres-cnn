# Poster Quick Start Guide

## 🎯 What You Need (5 Files)

Place these files in the **same directory** as `poster.tex`:

1. ✅ `input_lr.png` - LR input image (you have)
2. ✅ `output_cnn.png` - CNN output (you have)
3. ✅ `target_hr.png` - HR ground truth (you have)
4. ⚠️ `training_loss.png` - **Generate this** (instructions below)
5. ⚠️ `architecture.png` - **Generate this** (instructions below)

---

## 🚀 Generate Missing Images

### Step 1: Generate Architecture Diagram
```bash
python generate_architecture.py
```
✓ Creates `architecture.png`

### Step 2: Generate Training Loss Plot

**Option A - Use your actual training results:**
```bash
# For SRCNN
cp results/srcnn/training_loss.png training_loss.png

# For EDSR
cp results/edsr/training_loss.png training_loss.png
```

**Option B - Generate from history file:**
```bash
python plot_training_history.py \
    --history results/srcnn/training_history.json \
    --output training_loss.png
```

**Option C - Create sample plot (for testing):**
```bash
python plot_training_history.py --create-sample --output training_loss.png
```

---

## 📝 Compile Poster

### Verify Files First:
```bash
ls -la poster.tex *.png
```

You should see:
- `poster.tex`
- `input_lr.png`
- `output_cnn.png`
- `target_hr.png`
- `training_loss.png`
- `architecture.png`

### Compile:
```bash
pdflatex poster.tex
```

If you get errors, run it twice:
```bash
pdflatex poster.tex
pdflatex poster.tex
```

### Output:
Opens `poster.pdf` - your finished A0 poster!

---

## 📁 New File Organization

Training now creates separate folders for each model:

```
checkpoints/
├── srcnn/
│   ├── srcnn_best.pth
│   ├── srcnn_epoch_10.pth
│   └── srcnn_epoch_20.pth
└── edsr/
    ├── edsr_best.pth
    └── edsr_epoch_10.pth

results/
├── srcnn/
│   ├── training_loss.png      ← Use for poster
│   ├── training_psnr.png
│   ├── training_history.json
│   ├── sample_epoch_5.png
│   └── sr_images/
│       ├── epoch_5/
│       └── epoch_10/
└── edsr/
    ├── training_loss.png      ← Use for poster
    ├── training_psnr.png
    └── training_history.json
```

---

## 🎓 Training Your Models

### Train SRCNN:
```bash
python train.py \
    --model srcnn \
    --epochs 100 \
    --batch_size 16 \
    --loss mse \
    --visualize
```

**Outputs:**
- Checkpoints → `checkpoints/srcnn/`
- Results → `results/srcnn/`
- `training_loss.png` automatically created!

### Train EDSR:
```bash
python train.py \
    --model edsr \
    --epochs 100 \
    --batch_size 16 \
    --loss l1 \
    --visualize
```

**Outputs:**
- Checkpoints → `checkpoints/edsr/`
- Results → `results/edsr/`

---

## ✅ Complete Poster Workflow

```bash
# 1. Generate architecture diagram
python generate_architecture.py

# 2. Train your model (if not done already)
python train.py --model srcnn --epochs 100 --visualize

# 3. Copy training loss to poster directory
cp results/srcnn/training_loss.png .

# 4. Verify all images exist
ls -la poster.tex *.png

# 5. Compile poster
pdflatex poster.tex

# 6. Open result
open poster.pdf  # macOS
# or
evince poster.pdf  # Linux
```

---

## 🔧 Troubleshooting

### "File not found: architecture.png"
```bash
python generate_architecture.py
```

### "File not found: training_loss.png"
```bash
# Use sample data
python plot_training_history.py --create-sample --output training_loss.png

# Or use your actual data
cp results/srcnn/training_loss.png .
```

### LaTeX errors
1. Make sure all 5 PNG files exist in the same directory as `poster.tex`
2. Run `pdflatex poster.tex` twice
3. Check filenames are exact (case-sensitive!)

### Images don't appear in PDF
- Verify image files aren't corrupted: `file *.png`
- Regenerate any problematic images
- Make sure images aren't 0 bytes: `ls -lh *.png`

### Poster text is cut off
This is designed for A0 landscape paper. To print smaller:
- Use PDF viewer's "Fit to page" option
- Or scale the PDF: `pdfposter -mA3 poster.pdf poster_a3.pdf`

---

## 📊 Using Your Own Training Data

If you have your own loss values but no JSON file:

1. **Create history file manually:**
```json
{
  "train_loss": [0.045, 0.038, 0.032, ...],
  "val_loss": [0.048, 0.042, 0.035, ...],
  "val_epochs": [5, 10, 15, 20, ...]
}
```

2. **Save as** `my_history.json`

3. **Generate plot:**
```bash
python plot_training_history.py \
    --history my_history.json \
    --output training_loss.png
```

---

## 🎨 What's in the Poster

**Top:** Title and author info

**Three Columns:**
1. **Left:** Motivation, Method, Training Setup
2. **Center:** Model Architecture diagram
3. **Right:** Training/Validation Loss graph (large), Results, Future Work, GitHub

**Bottom:** Full-width image comparison (LR | CNN Output | HR Ground Truth)

---

## 📦 File Checklist

Before compiling, verify:

- [ ] `poster.tex` exists
- [ ] `input_lr.png` exists (your image)
- [ ] `output_cnn.png` exists (your image)
- [ ] `target_hr.png` exists (your image)
- [ ] `training_loss.png` exists (generated)
- [ ] `architecture.png` exists (generated)
- [ ] All images are in same directory as `poster.tex`

Then:
```bash
pdflatex poster.tex
```

Done! 🎉

---

## 🆘 Quick Commands Reference

```bash
# Generate everything needed for poster
python generate_architecture.py
python plot_training_history.py --create-sample --output training_loss.png

# Or use real data
cp results/srcnn/training_loss.png .

# Compile poster
pdflatex poster.tex

# View result
open poster.pdf
```

That's it! Your poster should compile cleanly now. The LaTeX code is simplified and error-free.