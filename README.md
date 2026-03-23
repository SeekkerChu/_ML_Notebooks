#Predicting Expert Aesthetic Ratings of Conservation Photography from Neural Network Activation Geometry
AestheticGeometry



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


Summary:

Determining the "beauty" of an image is a subjective and uniquely human task. There is significant value in building a model to study the human subjective thought process. The CCNY psychology department aims to build a model that predicts the mean score from judges for photos from the “Big Picture Competition.”


Steps: 
I. Feature Extraction & "Uniqueness":
   1.  **VGG16:** Extract activations from the last and middle convolutional layers. We use the middle layers to prevent data loss associated with the heavy downsampling in later max-pooling stages.
   2.  **Manual Features (Backup):** Color contrast, foreground/background detection, and segmentation counts to determine if "uniqueness" is actually the best indicator of a high score.
   3.  **DINO (Backup):** Unlike VGG16, which is largely translation-invariant, we hypothesize that object positioning matters for "uniqueness." DINO will be used to capture these spatial nuances.

II. Machine Learning Models for predicting the new features
   1. **Linear Regression:** To identify which features offer the best linear correlation to the mean judge score.
   2. **Classification:** Convert the continuous mean score into three categorical buckets: **Low (0)**, **Medium (1)**, and **High (2)**. We will use **Random Forest** to test predictability. We'll use a Random Forest classifier to see if our model is good at broad (but not precise) predictions.
   3. **Gradient Boosting:** Use **XGBoost** to handle the complex, non-linear relationships between visual activations and ratings by building a sequence of decision trees that iteratively correct prediction errors to estimate the precise continuous mean judge scores.
