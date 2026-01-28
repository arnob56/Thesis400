# =============================================================
# Multimodal Ventilator Days Prediction
# Clinical + Chest X-ray (DICOM)
# =============================================================

import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# =============================================================
# CONFIG
# =============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", DEVICE)

IMG_DIR = r"D:\project\data\xrays"
CSV_PATH = r"D:\project\data\labels.csv"

TARGET = "invasive_vent_days"
PATIENT_ID = "to_patient_id"

BATCH_SIZE = 12
EPOCHS = 80
LR = 3e-5
WEIGHT_DECAY = 1e-4

# =============================================================
# DICOM PROCESSING
# =============================================================

def load_dicom(path):
    dcm = pydicom.dcmread(path)

    img = dcm.pixel_array.astype(np.float32)

    # ---------------- Windowing ----------------
    if "WindowCenter" in dcm and "WindowWidth" in dcm:
        wc = np.mean(dcm.WindowCenter)
        ww = np.mean(dcm.WindowWidth)
        low = wc - ww / 2
        high = wc + ww / 2
        img = np.clip(img, low, high)

    # ---------------- Normalize ----------------
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # ---------------- Invert MONOCHROME1 ----------------
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 1.0 - img

    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (224, 224))

    img = np.stack([img, img, img], axis=-1)

    return Image.fromarray(img)

# =============================================================
# LOAD CSV
# =============================================================

df = pd.read_csv(CSV_PATH)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].notna()]
df = df[df[TARGET] >= 0]

df["target_log"] = np.log1p(df[TARGET])
df = df.reset_index(drop=True)

print("✅ Samples:", len(df))

# =============================================================
# CLINICAL FEATURES
# =============================================================

y = df["target_log"].values.astype(np.float32)

X = df.drop(columns=[TARGET, "target_log", PATIENT_ID])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("missing")

X = pd.get_dummies(X, columns=cat_cols)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# =============================================================
# IMAGE TRANSFORM
# =============================================================

img_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================================================
# DATASET
# =============================================================

class MultimodalDICOMDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        pid = str(self.ids[idx]).strip()
        path = os.path.join(IMG_DIR, f"{pid}.dcm")

        image = load_dicom(path)
        image = img_tf(image)

        return image, self.X[idx], self.y[idx]

# =============================================================
# MODEL
# =============================================================

class MultiModalAttentionDenseNet(nn.Module):
    def __init__(self, clin_dim):
        super().__init__()

        base = models.densenet121(weights="DEFAULT")

        for name, p in base.named_parameters():
            if "denseblock1" in name or "denseblock2" in name:
                p.requires_grad = False

        self.cnn = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.img_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.clin_fc = nn.Sequential(
            nn.Linear(clin_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        self.attn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, clin):
        img_feat = self.pool(self.cnn(img)).flatten(1)
        img_feat = self.img_fc(img_feat)

        clin_feat = self.clin_fc(clin)

        concat = torch.cat([img_feat, clin_feat], dim=1)
        w = self.attn(concat)

        fused = (
            w[:, 0:1] * img_feat +
            w[:, 1:2] * clin_feat
        )

        return self.regressor(fused).squeeze(1)

# =============================================================
# TRAIN / TEST
# =============================================================

X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    X, y, df[PATIENT_ID].values,
    test_size=0.2,
    random_state=42
)

train_ds = MultimodalDICOMDataset(X_tr, y_tr, id_tr)
test_ds  = MultimodalDICOMDataset(X_te, y_te, id_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =============================================================
# TRAINING
# =============================================================

model = MultiModalAttentionDenseNet(X.shape[1]).to(DEVICE)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=6
)

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for img, clin, y in train_loader:
        img, clin, y = img.to(DEVICE), clin.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(img, clin)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total += loss.item()

    scheduler.step(total)

    print(f"Epoch {epoch+1:03d} | Loss {total/len(train_loader):.4f}")

# =============================================================
# FINAL RESULTS
# =============================================================

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for img, clin, y in loader:
            img, clin = img.to(DEVICE), clin.to(DEVICE)
            p = model(img, clin)

            preds.append(p.cpu().numpy())
            trues.append(y.numpy())

    preds = np.expm1(np.concatenate(preds))
    trues = np.expm1(np.concatenate(trues))

    return {
        "MAE": mean_absolute_error(trues, preds),
        "RMSE": np.sqrt(mean_squared_error(trues, preds)),
        "R2": r2_score(trues, preds),
        "PCC": pearsonr(trues, preds)[0],
        "SCC": spearmanr(trues, preds)[0]
    }

results = evaluate(model, test_loader)

print("\n================ FINAL RESULTS ================\n")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
