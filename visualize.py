# visualize.py
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import freeze_support

# ── Configuración ──────────────────────────────────────────────────────────
IMG_SIZE = 256
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES3  = ['BENIGNO', 'MALIGNO', 'UNKNOWN']
COLORES  = {'BENIGNO': 'green', 'MALIGNO': 'red', 'UNKNOWN': 'gray'}

# ── Modelo (mismo que train_multitask.py) ──────────────────────────────────
class MultitaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features      = self.unet.encoder(x)
        deep_features = features[-1]
        decoder_out   = self.unet.decoder(features)
        seg_out       = self.unet.segmentation_head(decoder_out)
        cls_out       = self.cls_head(deep_features)
        return seg_out, cls_out


def preprocess(img_path):
    """Preprocesar imagen para el modelo"""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    t   = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])(image=img)
    return img, t['image'].unsqueeze(0).to(DEVICE)


def predict(model, img_tensor):
    """Obtener predicciones del modelo"""
    with torch.no_grad():
        seg_out, cls_out = model(img_tensor)
    mask_pred  = (torch.sigmoid(seg_out) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    cls_probs  = F.softmax(cls_out, dim=1).squeeze().cpu().numpy()
    cls_idx    = cls_probs.argmax()
    return mask_pred, cls_probs, cls_idx


def dice_single(pred, target):
    inter = (pred * target).sum()
    return (2*inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def main():
    # ── Cargar modelo ──────────────────────────────────────────────────────
    model = MultitaskModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("best_multitask_model.pth", map_location=DEVICE))
    model.eval()
    print("✅ Modelo cargado")

    # ── Cargar val con máscara Y etiqueta ─────────────────────────────────
    val_df = pd.read_csv("./data/val_multitask.csv")
    val_df['has_mask']  = val_df['has_mask'].astype(bool)
    val_df['has_label'] = val_df['has_label'].astype(bool)

    # Muestras con máscara para evaluar segmentación
    df_seg = val_df[val_df['has_mask']].reset_index(drop=True)
    # Muestras con etiqueta para evaluar clasificación
    df_cls = val_df[val_df['has_label'] & (val_df['clase3'] != 'UNKNOWN')].reset_index(drop=True)

    print(f"Muestras con máscara : {len(df_seg)}")
    print(f"Muestras con etiqueta: {len(df_cls)}")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 1: Segmentación — predicción vs ground truth
    # ══════════════════════════════════════════════════════════════════════
    sample_seg = df_seg.sample(8, random_state=42).reset_index(drop=True)
    dice_scores = []

    fig, axes = plt.subplots(8, 4, figsize=(16, 26))
    fig.suptitle("Segmentación — Predicción vs Ground Truth\n"
                 "🟡 Ambos correcto  🟢 Solo GT  🔴 Solo predicción",
                 fontsize=13, y=1.005)

    for i, row in sample_seg.iterrows():
        img_orig, img_tensor = preprocess(row['image_path'])
        mask_pred, cls_probs, cls_idx = predict(model, img_tensor)

        # Ground truth
        mask_gt = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask_gt = cv2.resize(mask_gt, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_NEAREST)
        mask_gt = (mask_gt > 127).astype(np.uint8)

        img_vis = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))

        # Dice por imagen
        d = dice_single(mask_pred, mask_gt)
        dice_scores.append(d)

        # Overlay comparativo
        overlay   = img_vis.copy()
        only_gt   = (mask_gt == 1) & (mask_pred == 0)
        only_pred = (mask_pred == 1) & (mask_gt == 0)
        both      = (mask_gt == 1) & (mask_pred == 1)
        overlay[only_gt]   = (overlay[only_gt]   * 0.4 + np.array([0,220,0])   * 0.6).astype(np.uint8)
        overlay[only_pred] = (overlay[only_pred] * 0.4 + np.array([220,50,50]) * 0.6).astype(np.uint8)
        overlay[both]      = (overlay[both]      * 0.4 + np.array([255,220,0]) * 0.6).astype(np.uint8)

        # Clasificación predicha
        cls_nombre = CLASES3[cls_idx]
        cls_color  = COLORES[cls_nombre]

        axes[i,0].imshow(img_vis)
        axes[i,0].set_title(f"Original\nClasif: {cls_nombre} ({cls_probs[cls_idx]*100:.0f}%)",
                             fontsize=8, color=cls_color, fontweight='bold')
        axes[i,0].axis('off')

        axes[i,1].imshow(mask_gt, cmap='hot')
        axes[i,1].set_title("Ground Truth", fontsize=8)
        axes[i,1].axis('off')

        axes[i,2].imshow(mask_pred, cmap='hot')
        axes[i,2].set_title(f"Predicción (Dice: {d:.3f})", fontsize=8)
        axes[i,2].axis('off')

        axes[i,3].imshow(overlay)
        axes[i,3].set_title("Comparativo", fontsize=8)
        axes[i,3].axis('off')

    plt.tight_layout()
    plt.savefig("viz_segmentacion.png", dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\nDice promedio (8 muestras): {np.mean(dice_scores):.4f}")
    print(f"Mejor  : {max(dice_scores):.4f}")
    print(f"Peor   : {min(dice_scores):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 2: Clasificación — ejemplos con confianza
    # ══════════════════════════════════════════════════════════════════════
    # 4 malignos + 4 benignos
    mal = df_cls[df_cls['clase3']=='MALIGNO'].sample(4, random_state=42)
    ben = df_cls[df_cls['clase3']=='BENIGNO'].sample(4, random_state=42)
    sample_cls = pd.concat([mal, ben]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle("Clasificación — Predicción del modelo (MALIGNO / BENIGNO)",
                 fontsize=13)

    for i, (_, row) in enumerate(sample_cls.iterrows()):
        r_idx = i // 4
        c_idx = i % 4

        img_orig, img_tensor = preprocess(row['image_path'])
        _, cls_probs, cls_idx = predict(model, img_tensor)

        img_vis    = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))
        pred_clase = CLASES3[cls_idx]
        real_clase = row['clase3']
        correcto   = pred_clase == real_clase

        # Borde de color según acierto
        borde_color = [0,200,0] if correcto else [220,50,50]
        img_borde   = cv2.copyMakeBorder(img_vis, 8,8,8,8,
                                          cv2.BORDER_CONSTANT, value=borde_color)

        axes[r_idx, c_idx].imshow(img_borde)
        titulo = (f"Real: {real_clase}\n"
                  f"Pred: {pred_clase} ({cls_probs[cls_idx]*100:.0f}%)\n"
                  f"{'✅ Correcto' if correcto else '❌ Error'}")
        axes[r_idx, c_idx].set_title(titulo, fontsize=9,
                                      color='green' if correcto else 'red')
        axes[r_idx, c_idx].axis('off')

        # Mini barra de probabilidades
        ax_ins = axes[r_idx, c_idx].inset_axes([0, -0.35, 1, 0.28])
        bars   = ax_ins.bar(CLASES3, cls_probs,
                             color=['green','red','gray'], alpha=0.7)
        ax_ins.set_ylim(0,1)
        ax_ins.tick_params(labelsize=7)
        ax_ins.set_title("Probabilidades", fontsize=7)

    plt.tight_layout()
    plt.savefig("viz_clasificacion.png", dpi=120, bbox_inches='tight')
    plt.show()

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 3: Matriz de confusión
    # ══════════════════════════════════════════════════════════════════════
    print("\nCalculando matriz de confusión en val completo...")

    y_true, y_pred = [], []
    for _, row in df_cls.iterrows():
        _, img_tensor    = preprocess(row['image_path'])
        _, cls_probs, ci = predict(model, img_tensor)
        y_true.append(CLASES3.index(row['clase3']))
        y_pred.append(int(ci))

    # Matriz manual
    n_cls   = 2  # solo BENIGNO y MALIGNO
    classes = ['BENIGNO', 'MALIGNO']
    cm      = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t < 2 and p < 2:  # ignorar UNKNOWN
            cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusión — Clasificación", fontsize=13)

    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, str(cm[i,j]),
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

    # Métricas
    tp = cm[1,1]; tn = cm[0,0]
    fp = cm[0,1]; fn = cm[1,0]
    sensibilidad = tp/(tp+fn+1e-6)
    especificidad = tn/(tn+fp+1e-6)
    accuracy = (tp+tn)/(cm.sum()+1e-6)

    stats = (f"Accuracy     : {accuracy:.3f}\n"
             f"Sensibilidad : {sensibilidad:.3f}  (detecta malignos)\n"
             f"Especificidad: {especificidad:.3f}  (descarta benignos)")
    ax.text(1.05, 0.5, stats, transform=ax.transAxes,
            fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("viz_confusion.png", dpi=120, bbox_inches='tight')
    plt.show()

    print(stats)
    print("\n✅ 3 visualizaciones guardadas:")
    print("  viz_segmentacion.png")
    print("  viz_clasificacion.png")
    print("  viz_confusion.png")


if __name__ == "__main__":
    freeze_support()
    main()