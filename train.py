# train.py
import os, sys, glob
from multiprocessing import freeze_support
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Configuración ──────────────────────────────────────────────────────────
IMG_SIZE   = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_LOADER_WORKERS = 2 if os.name == "nt" else 4 # Windows suele tener problemas con >0 workers, Linux/Mac pueden usar más
PIN_MEMORY = DEVICE.type == "cuda"

# ── Rutas — ajusta si tu estructura es diferente ───────────────────────────
PATH_IMAGES = "./data/ISIC2018_Task1-2_Training_Input"
PATH_MASKS  = "./data/ISIC2018_Task1_Training_GroundTruth"

def get_image_for_mask(mask_path, image_list):
    mask_id = os.path.basename(mask_path).replace('_segmentation.png', '')
    for img in image_list:
        if mask_id in img:
            return img
    return None

# ── Dataset ────────────────────────────────────────────────────────────────
class ISICDataset(Dataset):
    def __init__(self, df, img_size=256, augment=False):
        self.df        = df.reset_index(drop=True)
        self.img_size  = img_size
        self.transform = self._build_transforms(augment)

    def _build_transforms(self, augment):
        if augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size,
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05),
                         rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.2),
                A.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size,
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = cv2.cvtColor(cv2.imread(row['image_path']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        t    = self.transform(image=img, mask=mask)
        return t['image'], t['mask'].float().unsqueeze(0)

# ── Modelo y Loss ──────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pw = torch.tensor([pos_weight])

    def forward(self, pred, target):
        p   = torch.sigmoid(pred)
        inter = (p.view(-1) * target.view(-1)).sum()
        dice  = 1 - (2*inter + 1e-6) / (p.view(-1).sum() + target.view(-1).sum() + 1e-6)
        bce   = F.binary_cross_entropy_with_logits(pred, target,
                    pos_weight=self.pw.to(pred.device))
        return dice + bce

def dice_score(pred, target, thr=0.5):
    pb    = (torch.sigmoid(pred) > thr).float()
    inter = (pb * target).sum()
    return ((2*inter + 1e-6) / (pb.sum() + target.sum() + 1e-6)).item()


def build_dataframes():
    images = sorted(glob.glob(f"{PATH_IMAGES}/**/*.jpg", recursive=True))
    masks  = sorted(glob.glob(f"{PATH_MASKS}/**/*_segmentation.png", recursive=True))

    print(f"Imágenes : {len(images)}")
    print(f"Máscaras : {len(masks)}")

    records = []
    for mask_path in masks:
        img_path = get_image_for_mask(mask_path, images)
        if img_path:
            records.append({'image_path': img_path, 'mask_path': mask_path})

    import pandas as pd
    df = pd.DataFrame(records)
    print(f"Pares válidos: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df

def main():
    print(f"Dispositivo: {DEVICE}")
    train_df, val_df = build_dataframes()

    train_loader = DataLoader(ISICDataset(train_df, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=DATA_LOADER_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(ISICDataset(val_df, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=DATA_LOADER_WORKERS, pin_memory=PIN_MEMORY)

    model     = smp.Unet("resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation=None).to(DEVICE)
    criterion = CombinedLoss(pos_weight=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', patience=3, factor=0.5)

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    best_dice = 0.0
    history   = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]", leave=False)

        for imgs, masks in bar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds      = model(imgs)
                val_loss  += criterion(preds, masks).item()
                val_dice  += dice_score(preds, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step(val_dice)
        tag = " ← mejor" if val_dice == best_dice else ""
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}{tag}")
        sys.stdout.flush()

    # ── Curvas ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train", color='royalblue')
    axes[0].plot(history["val_loss"],   label="Val",   color='tomato')
    axes[0].set_title("Loss por epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_dice"], color='green', label="Val Dice")
    axes[1].axhline(best_dice, color='darkgreen', linestyle='--',
                    label=f"Mejor: {best_dice:.4f}")
    axes[1].set_title("Dice Score"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("curvas_entrenamiento.png", dpi=110)
    plt.show()
    print(f"\nMejor Dice Score: {best_dice:.4f}")
    print("Modelo guardado en best_model.pth")


if __name__ == "__main__":
    freeze_support()
    main()