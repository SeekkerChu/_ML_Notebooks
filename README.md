# 🧠 Deep Learning Environment Setup
### CNN · DINO · ViT · Data Science Stack

A fully reproducible environment for training and experimenting with convolutional networks (CNN), self-supervised vision transformers (DINO), and standard ViT models — alongside a complete data science stack (NumPy, Pandas, Matplotlib, Seaborn, Plotly, and more).

---

## 📦 What's Included

| Category | Libraries |
|---|---|
| **Core Computing** | NumPy, Pandas, SciPy |
| **Deep Learning** | PyTorch, TorchVision |
| **CNN** | TorchVision (ResNet, VGG, EfficientNet via `timm`) |
| **ViT** | `transformers` (ViT-B/16, ViT-L/32), `timm` |
| **DINO** | `timm` (DINO-pretrained ViT), `transformers` |
| **Plotting** | Matplotlib, Seaborn, Plotly, UMAP |
| **Training Utils** | Accelerate, TorchMetrics, TensorBoard, W&B |
| **Data Loading** | Hugging Face Datasets, Pillow, OpenCV, Albumentations |
| **Notebooks** | JupyterLab, ipywidgets |

---

## 🚀 Setup with Conda (Recommended)

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed
- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration

---

### Step 1 — Create a new Conda environment

```bash
conda create -n dl-env python=3.11 -y
```

> **Why Python 3.11?** It offers the best compatibility with PyTorch 2.x and Hugging Face libraries as of 2024–2025.

---

### Step 2 — Activate the environment

```bash
conda activate dl-env
```

You should see `(dl-env)` at the start of your terminal prompt.

---

### Step 3 — Install PyTorch (choose your hardware)

**GPU (CUDA 12.1) — recommended for training:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**GPU (CUDA 11.8) — for older NVIDIA drivers:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CPU only — for Mac or machines without a GPU:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

> ⚠️ Always install PyTorch via `conda` **before** installing the rest of the requirements. This ensures the correct CUDA binaries are linked.

---

### Step 4 — Install all remaining dependencies

```bash
pip install -r requirements.txt
```

> `pip` is used here because some packages (e.g., `timm`, `transformers`, `albumentations`) are best installed from PyPI.

---

### Step 5 — Verify the installation

Run the following quick check to confirm everything is working:

```python
# verify.py
import torch
import torchvision
import transformers
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PyTorch:        {torch.__version__}")
print(f"TorchVision:    {torchvision.__version__}")
print(f"Transformers:   {transformers.__version__}")
print(f"timm:           {timm.__version__}")
print(f"NumPy:          {np.__version__}")
print(f"Pandas:         {pd.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
```

```bash
python verify.py
```

---

## 🔬 Quick Usage Examples

### Load a DINO ViT (pretrained, self-supervised)

```python
import timm
import torch

# Load DINO ViT-S/16 pretrained on ImageNet
model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
model.eval()

# Dummy input: batch of 4 images, 3 channels, 224x224
x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    features = model.forward_features(x)   # (4, 197, 384) — patch tokens
    cls_token = features[:, 0, :]          # (4, 384) — CLS embedding
print("DINO CLS token shape:", cls_token.shape)
```

---

### Load a ViT from Hugging Face

```python
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import requests

# Load model and processor
processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Load a sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print("Last hidden state:", outputs.last_hidden_state.shape)  # (1, 197, 768)
```

---

### Load a CNN (ResNet-50 via timm)

```python
import timm
import torch

model = timm.create_model("resnet50", pretrained=True, num_classes=10)
model.eval()

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    logits = model(x)
print("CNN output shape:", logits.shape)  # (2, 10)
```

---

### Plotting with Matplotlib & Seaborn

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample training history
epochs = np.arange(1, 21)
loss = np.exp(-0.1 * epochs) + np.random.normal(0, 0.01, 20)
acc = 1 - np.exp(-0.15 * epochs) + np.random.normal(0, 0.01, 20)

df = pd.DataFrame({"epoch": epochs, "loss": loss, "accuracy": acc})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=df, x="epoch", y="loss", ax=axes[0], marker="o")
axes[0].set_title("Training Loss")

sns.lineplot(data=df, x="epoch", y="accuracy", ax=axes[1], marker="o", color="green")
axes[1].set_title("Training Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
```

---

### UMAP Feature Visualization

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

# Simulated features from a ViT CLS token
features = np.random.randn(500, 384)
labels = np.random.randint(0, 10, 500)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10)
plt.colorbar(scatter, label="Class")
plt.title("UMAP of ViT Features")
plt.savefig("umap_features.png", dpi=150)
plt.show()
```

---

## 🗂️ Recommended Project Structure

```
project/
├── data/                  # Raw and processed datasets
├── models/                # Model definitions (CNN, ViT, DINO heads)
├── notebooks/             # EDA and experiment notebooks
├── scripts/               # Training and evaluation scripts
├── configs/               # Hydra YAML config files
├── outputs/               # Checkpoints, logs, plots
├── requirements.txt       # This file
└── README.md
```

---

## 🧹 Managing the Conda Environment

```bash
# Activate
conda activate dl-env

# Deactivate
conda deactivate

# List all environments
conda env list

# Export environment (for exact reproducibility)
conda env export > environment.yml

# Recreate from exported file
conda env create -f environment.yml

# Remove environment
conda remove -n dl-env --all -y
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA not available` | Check driver version with `nvidia-smi`. Reinstall PyTorch with matching CUDA version. |
| `ImportError: timm` | Run `pip install timm --upgrade` |
| `ModuleNotFoundError: transformers` | Run `pip install transformers --upgrade` |
| Slow downloads on Hugging Face | Set `HF_HUB_OFFLINE=1` after caching models, or use `TRANSFORMERS_CACHE=/your/path` |
| Out of memory (OOM) | Reduce batch size, use `torch.cuda.empty_cache()`, enable mixed precision with `torch.autocast` |
| `umap-learn` install fails | Install `wheel` first: `pip install wheel` then retry |

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers — ViT](https://huggingface.co/docs/transformers/model_doc/vit)
- [timm — PyTorch Image Models](https://huggingface.co/docs/timm/index)
- [DINO: Self-supervised Vision Transformers (paper)](https://arxiv.org/abs/2104.14294)
- [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
