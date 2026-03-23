#Image Rating Prediction

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

##Setup with Conda

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



## Conda Cheatsheet

```bash
conda activate rating-env          # activate
conda deactivate                   # deactivate
conda env export > environment.yml # snapshot for reproducibility
conda env create -f environment.yml# restore from snapshot
conda remove -n rating-env --all   # delete environment
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA not available | Check `nvidia-smi`, reinstall PyTorch with matching CUDA version |
| `timm` DINO model not found | `pip install timm --upgrade` |
| Memory error during extraction | Process images in batches or use CPU |
| XGBoost slow on CPU | Add `tree_method="hist"` to `XGBRegressor(...)` |
| Poor R² score | Try feature scaling, PCA to reduce dimensions, or tune hyperparameters with `GridSearchCV` |
