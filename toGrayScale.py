from PIL import Image
import os

hr_rgb = '/content/data/DIV2K_train_HR/'
lr_rgb = '/content/data/DIV2K_train_LR_bicubic/X2/'

hr_gray = '/content/data/DIV2K_train_HR_gray/'
lr_gray = '/content/data/DIV2K_train_LR_gray/'

os.makedirs(hr_gray, exist_ok=True)
os.makedirs(lr_gray, exist_ok=True)

for folder_src, folder_dst in [(hr_rgb, hr_gray), (lr_rgb, lr_gray)]:
    for fname in os.listdir(folder_src):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(folder_src, fname)).convert('L')  # 'L' = grayscale
            img.save(os.path.join(folder_dst, fname))
