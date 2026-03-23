# 🖼️ Image Rating Prediction
### VGG16 · ViT · DINO → Random Forest · XGBoost · Regression

Predict human-judged image ratings by extracting deep visual features with pretrained CNN/Transformer backbones and feeding them into classical ML models.

**Pipeline overview:**
```
Images → [VGG16 | ViT | DINO] (feature extractor)
       → feature vectors (NumPy / Pandas)
       → train/test split
       → [Linear Regression | Ridge | Random Forest | XGBoost]
       → predicted rating score
```

---

## ⚙️ Setup with Conda

### Step 1 — Create environment

```bash
conda create -n rating-env python=3.11 -y
conda activate rating-env
```

### Step 2 — Install PyTorch

**GPU (CUDA 12.1):**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU only / Apple Silicon:**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### Step 3 — Install remaining packages

```bash
pip install -r requirements.txt
```

### Step 4 — Verify

```bash
python -c "
import torch, torchvision, timm, sklearn, xgboost
print('torch    :', torch.__version__)
print('timm     :', timm.__version__)
print('sklearn  :', sklearn.__version__)
print('xgboost  :', xgboost.__version__)
print('CUDA     :', torch.cuda.is_available())
"
```

---

## 🔧 Feature Extraction

### VGG16 (TorchVision)

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Build extractor — remove final classifier, keep up to pool layer
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
extractor = torch.nn.Sequential(*list(vgg16.features), vgg16.avgpool)
extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def extract_vgg16(image_path: str) -> np.ndarray:
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = extractor(img)           # (1, 512, 7, 7)
    return feat.flatten().numpy()       # → 25088-dim vector
```

---

### ViT-B/16 (timm)

```python
import timm, torch
import numpy as np

vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
vit.eval()

def extract_vit(image_path: str) -> np.ndarray:
    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = vit(img)                 # (1, 768) — CLS token
    return feat.squeeze().numpy()       # → 768-dim vector
```

---

### DINO ViT-S/16 (timm, self-supervised)

```python
import timm, torch
import numpy as np

dino = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)
dino.eval()

def extract_dino(image_path: str) -> np.ndarray:
    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = dino(img)                # (1, 384) — CLS token
    return feat.squeeze().numpy()       # → 384-dim vector
```

---

## 🏗️ Build Feature Matrix

```python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Assume: image_paths (list of file paths), ratings (list of float scores)
image_paths = [...]   # e.g. ["data/img001.jpg", "data/img002.jpg", ...]
ratings     = [...]   # e.g. [7.2, 4.5, 8.1, ...]

# Choose your extractor (swap freely)
extract_fn = extract_dino    # or extract_vgg16 / extract_vit

features = np.array([
    extract_fn(p) for p in tqdm(image_paths, desc="Extracting features")
])

df = pd.DataFrame(features)
df["rating"] = ratings
df.to_csv("features.csv", index=False)
print(f"Feature matrix: {df.shape}")  # (n_images, n_features + 1)
```

---

## 🤖 Train & Evaluate ML Models

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load features
df = pd.read_csv("features.csv")
X = df.drop(columns=["rating"]).values
y = df["rating"].values

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for regression models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":           XGBRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=6, random_state=42),
}

# Train and evaluate all
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse  = mean_squared_error(y_test, preds, squared=False)
    r2    = r2_score(y_test, preds)
    results.append({"Model": name, "RMSE": round(rmse, 4), "R²": round(r2, 4)})
    print(f"{name:<22}  RMSE={rmse:.4f}  R²={r2:.4f}")

results_df = pd.DataFrame(results)
```

---

## 📊 Plots

### Model Comparison Bar Chart

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[0], palette="Blues_d")
axes[0].set_title("RMSE (lower is better)")
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(data=results_df, x="Model", y="R²", ax=axes[1], palette="Greens_d")
axes[1].set_title("R² Score (higher is better)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
```

### Predicted vs Actual Scatter

```python
best_model = models["XGBoost"]
preds = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.5, edgecolors="k", linewidths=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect fit")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("XGBoost — Predicted vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150)
plt.show()
```

### Residual Distribution

```python
residuals = y_test - preds

plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True, bins=30, color="steelblue")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residual (Actual − Predicted)")
plt.title("Residual Distribution — XGBoost")
plt.tight_layout()
plt.savefig("residuals.png", dpi=150)
plt.show()
```

---

## 🗂️ Project Structure

```
project/
├── data/
│   ├── images/            # Raw image files
│   └── ratings.csv        # image_path, rating columns
├── features.csv           # Extracted feature matrix (generated)
├── notebooks/
│   └── exploration.ipynb
├── extract_features.py    # Feature extraction script
├── train.py               # Model training script
├── requirements.txt
└── README.md
```

---

## 🧹 Conda Cheatsheet

```bash
conda activate rating-env          # activate
conda deactivate                   # deactivate
conda env export > environment.yml # snapshot for reproducibility
conda env create -f environment.yml# restore from snapshot
conda remove -n rating-env --all   # delete environment
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| CUDA not available | Check `nvidia-smi`, reinstall PyTorch with matching CUDA version |
| `timm` DINO model not found | `pip install timm --upgrade` |
| Memory error during extraction | Process images in batches or use CPU |
| XGBoost slow on CPU | Add `tree_method="hist"` to `XGBRegressor(...)` |
| Poor R² score | Try feature scaling, PCA to reduce dimensions, or tune hyperparameters with `GridSearchCV` |
