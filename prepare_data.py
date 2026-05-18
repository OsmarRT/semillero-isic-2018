# prepare_data.py — corregir split para que val tenga máscaras
import glob, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_MASKS   = "./data/ISIC2018_Task1_Training_GroundTruth"
PATH_T1_IMGS = "./data/ISIC2018_Task1-2_Training_Input"
PATH_T3_IMGS = "./data/ISIC2018_Task3_Training_Input"
PATH_T3_CSV  = "./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"

# ── Etiquetas ──────────────────────────────────────────────────────────────
df_labels = pd.read_csv(PATH_T3_CSV)
CLASES  = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
MALIGNO = {'MEL','BCC','AKIEC'}

def get_diagnostico(row):
    for c in CLASES:
        if row[c] == 1.0:
            return c
    return 'UNKNOWN'

def get_clase3(diag):
    if diag in {'MEL','BCC','AKIEC'}:  return 'MALIGNO'
    if diag in {'NV','BKL','DF','VASC'}: return 'BENIGNO'
    return 'UNKNOWN'

df_labels['diagnostico'] = df_labels.apply(get_diagnostico, axis=1)
df_labels['clase3']      = df_labels['diagnostico'].apply(get_clase3)
df_labels['label_idx']   = df_labels['diagnostico'].apply(
                               lambda x: CLASES.index(x) if x in CLASES else -1)
df_labels['es_maligno']  = df_labels['diagnostico'].isin(MALIGNO).astype(int)

# ── Indexar archivos ───────────────────────────────────────────────────────
masks        = sorted(glob.glob(f"{PATH_MASKS}/**/*_segmentation.png", recursive=True))
t1_mask_idx  = {os.path.basename(m).replace('_segmentation.png',''): m for m in masks}
t1_imgs      = sorted(glob.glob(f"{PATH_T1_IMGS}/**/*.jpg", recursive=True))
t1_img_idx   = {os.path.basename(p).replace('.jpg',''): p for p in t1_imgs}
t3_imgs      = sorted(glob.glob(f"{PATH_T3_IMGS}/**/*.jpg", recursive=True))
t3_img_idx   = {os.path.basename(p).replace('.jpg',''): p for p in t3_imgs}

# ── Construir registros ────────────────────────────────────────────────────
records = []

# Task 1: imágenes con máscara (sin etiqueta)
for img_id, mask_path in t1_mask_idx.items():
    img_path = t1_img_idx.get(img_id)
    records.append({
        'image_id':    img_id,
        'image_path':  img_path,
        'mask_path':   mask_path,
        'has_mask':    True,
        'has_label':   False,
        'diagnostico': 'UNKNOWN',
        'clase3':      'UNKNOWN',
        'label_idx':   -1,
        'es_maligno':  -1,
    })

# Task 3: imágenes con etiqueta (sin máscara)
for _, row in df_labels.iterrows():
    img_id   = row['image']
    img_path = t3_img_idx.get(img_id)
    if img_path is None:
        continue
    records.append({
        'image_id':    img_id,
        'image_path':  img_path,
        'mask_path':   None,
        'has_mask':    False,
        'has_label':   True,
        'diagnostico': row['diagnostico'],
        'clase3':      row['clase3'],
        'label_idx':   int(row['label_idx']),
        'es_maligno':  int(row['es_maligno']),
    })

df = pd.DataFrame(records)

# ── Split CORRECTO: ambos sets tienen máscaras Y etiquetas ─────────────────
# Separar por tipo
df_mask_only  = df[df['has_mask'] & ~df['has_label']].copy()   # 2594 solo máscara
df_label_only = df[~df['has_mask'] & df['has_label']].copy()   # 10015 solo etiqueta

# Split de los que tienen máscara → 85% train, 15% val
train_mask, val_mask = train_test_split(
    df_mask_only, test_size=0.15, random_state=42
)

# Split de los que tienen etiqueta → 85% train, 15% val estratificado
train_label, val_label = train_test_split(
    df_label_only, test_size=0.15,
    stratify=df_label_only['diagnostico'],
    random_state=42
)

# Combinar
train_df = pd.concat([train_mask, train_label]).reset_index(drop=True)
val_df   = pd.concat([val_mask,   val_label  ]).reset_index(drop=True)

print(f"=== SPLIT CORREGIDO ===")
print(f"Train: {len(train_df)}")
print(f"  con máscara  : {train_df['has_mask'].sum()}")
print(f"  con etiqueta : {train_df['has_label'].sum()}")
print(f"\nVal  : {len(val_df)}")
print(f"  con máscara  : {val_df['has_mask'].sum()}")
print(f"  con etiqueta : {val_df['has_label'].sum()}")
print(f"\nDistribución diagnóstico en val:")
print(val_df[val_df['has_label']]['diagnostico'].value_counts())
print(f"\nClase3 en val:")
print(val_df[val_df['has_label']]['clase3'].value_counts())

train_df.to_csv("./data/train_multitask.csv", index=False)
val_df.to_csv("./data/val_multitask.csv",     index=False)
print("\n✅ CSVs guardados")