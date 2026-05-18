# train_multitask.py
import os, sys, glob
from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

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
NUM_EPOCHS = 40
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pesos de la loss multitask
W_SEG = 1.0   # peso segmentación
W_CLS = 0.5   # peso clasificación (menos porque es tarea secundaria)

CLASES3 = ['BENIGNO', 'MALIGNO', 'UNKNOWN']

print(f"Dispositivo: {DEVICE}")

# ── Dataset multitask ──────────────────────────────────────────────────────
class ISICMultitaskDataset(Dataset):
    def __init__(self, df, img_size=256, augment=False):
        self.df        = df.reset_index(drop=True)
        self.img_size  = img_size
        self.transform = self._build_transforms(augment)

    def __len__(self):
        return len(self.df)

    def _build_transforms(self, augment):
        if augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size,
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(scale=(0.9,1.1), translate_percent=(0.05,0.05),
                         rotate=(-15,15), p=0.5),
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Imagen ─────────────────────────────────────────────────────────
        img  = cv2.cvtColor(cv2.imread(row['image_path']), cv2.COLOR_BGR2RGB)

        # ── Máscara ────────────────────────────────────────────────────────
        if row['has_mask'] and pd.notna(row['mask_path']):
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # ── Transformaciones ───────────────────────────────────────────────
        t       = self.transform(image=img, mask=mask)
        image_t = t['image']
        mask_t  = t['mask'].float().unsqueeze(0)

        # ── Etiqueta de clasificación ──────────────────────────────────────
        if row['has_label'] and row['clase3'] in CLASES3:
            label = CLASES3.index(row['clase3'])
        else:
            label = -1  # sin etiqueta

        return image_t, mask_t, torch.tensor(label, dtype=torch.long)


# ── Modelo multitask ───────────────────────────────────────────────────────
class MultitaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder + Decoder para segmentación (U-Net completa)
        self.unet = smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )

        # Cabeza de clasificación sobre el encoder compartido
        # ResNet34 produce 512 canales en la última capa del encoder
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, 512, H, W) → (B, 512, 1, 1)
            nn.Flatten(),              # (B, 512)
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Obtener features del encoder
        features = self.unet.encoder(x)
        # features es una lista — el último elemento son las features más profundas
        deep_features = features[-1]  # (B, 512, H/32, W/32)

        # Segmentación: pasar por decoder
        seg_out = self.unet.decoder(features)
        seg_out = self.unet.segmentation_head(seg_out)  # (B, 1, H, W)

        # Clasificación: pasar features profundas por cabeza cls
        cls_out = self.cls_head(deep_features)  # (B, num_classes)

        return seg_out, cls_out


# ── Loss multitask ─────────────────────────────────────────────────────────
class MultitaskLoss(nn.Module):
    def __init__(self, w_seg=1.0, w_cls=0.5, pos_weight=2.0):
        super().__init__()
        self.w_seg = w_seg
        self.w_cls = w_cls
        self.pw    = torch.tensor([pos_weight])

    def dice_loss(self, pred, target):
        p     = torch.sigmoid(pred)
        inter = (p.view(-1) * target.view(-1)).sum()
        return 1 - (2*inter + 1e-6) / (p.view(-1).sum() + target.view(-1).sum() + 1e-6)

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        # ── Loss de segmentación (solo muestras con máscara real) ──────────
        has_mask = (seg_target.sum(dim=(1,2,3)) >= 0).all()  # todas tienen máscara tensor
        seg_bce  = F.binary_cross_entropy_with_logits(
                       seg_pred, seg_target,
                       pos_weight=self.pw.to(seg_pred.device))
        seg_dice = self.dice_loss(seg_pred, seg_target)
        loss_seg = seg_bce + seg_dice

        # ── Loss de clasificación (solo muestras con etiqueta) ─────────────
        valid_cls = cls_target >= 0   # máscara de muestras con etiqueta
        if valid_cls.sum() > 0:
            loss_cls = F.cross_entropy(
                cls_pred[valid_cls],
                cls_target[valid_cls]
            )
        else:
            loss_cls = torch.tensor(0.0, device=seg_pred.device)

        return self.w_seg * loss_seg + self.w_cls * loss_cls, loss_seg, loss_cls


# ── Métricas ───────────────────────────────────────────────────────────────
def dice_score(pred, target, thr=0.5):
    pb    = (torch.sigmoid(pred) > thr).float()
    inter = (pb * target).sum()
    return ((2*inter + 1e-6) / (pb.sum() + target.sum() + 1e-6)).item()

def cls_accuracy(pred, target):
    valid = target >= 0
    if valid.sum() == 0:
        return 0.0
    correct = (pred[valid].argmax(dim=1) == target[valid]).float().sum()
    return (correct / valid.sum()).item()


# ── Entrenamiento ──────────────────────────────────────────────────────────
def main():
    # Cargar splits
    train_df = pd.read_csv("./data/train_multitask.csv")
    val_df   = pd.read_csv("./data/val_multitask.csv")

    # Convertir columnas booleanas
    train_df['has_mask']  = train_df['has_mask'].astype(bool)
    train_df['has_label'] = train_df['has_label'].astype(bool)
    val_df['has_mask']    = val_df['has_mask'].astype(bool)
    val_df['has_label']   = val_df['has_label'].astype(bool)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        ISICMultitaskDataset(train_df, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        ISICMultitaskDataset(val_df, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model     = MultitaskModel(num_classes=3).to(DEVICE)
    criterion = MultitaskLoss(w_seg=W_SEG, w_cls=W_CLS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', patience=4, factor=0.5)

    best_dice  = 0.0
    history    = {"train_loss":[], "val_loss":[],
                  "val_dice":[], "val_acc":[]}

    for epoch in range(NUM_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader,
                   desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]",
                   leave=False)

        for imgs, masks, labels in bar:
            imgs   = imgs.to(DEVICE)
            masks  = masks.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            seg_out, cls_out = model(imgs)
            loss, l_seg, l_cls = criterion(seg_out, masks, cls_out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}",
                            seg=f"{l_seg.item():.3f}",
                            cls=f"{l_cls.item():.3f}")

        train_loss /= len(train_loader)

        # ── Validación ─────────────────────────────────────────────────────
        model.eval()
        val_loss, val_dice, val_acc = 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs   = imgs.to(DEVICE)
                masks  = masks.to(DEVICE)
                labels = labels.to(DEVICE)

                seg_out, cls_out     = model(imgs)
                loss, _, _           = criterion(seg_out, masks, cls_out, labels)
                val_loss            += loss.item()
                val_dice            += dice_score(seg_out, masks)
                val_acc             += cls_accuracy(cls_out, labels)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc  /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_multitask_model.pth")

        scheduler.step(val_dice)

        tag = " ← mejor" if val_dice == best_dice else ""
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | "
              f"Acc: {val_acc:.4f}{tag}")
        sys.stdout.flush()

    # ── Curvas ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train", color='royalblue')
    axes[0].plot(history["val_loss"],   label="Val",   color='tomato')
    axes[0].set_title("Loss total"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_dice"], color='green', label="Val Dice")
    axes[1].axhline(best_dice, color='darkgreen', linestyle='--',
                    label=f"Mejor: {best_dice:.4f}")
    axes[1].set_title("Dice Score (segmentación)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_acc"], color='purple', label="Val Accuracy")
    axes[2].set_title("Accuracy (clasificación 3 clases)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("curvas_multitask.png", dpi=110)
    plt.show()
    print(f"\nMejor Dice     : {best_dice:.4f}")
    print(f"Última Accuracy: {val_acc:.4f}")
    print("Modelo guardado en best_multitask_model.pth")


if __name__ == "__main__":
    freeze_support()
    main()