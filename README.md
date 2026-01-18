

![header](https://capsule-render.vercel.app/api?type=waving&height=150&color=gradient&text=LitePath:-nl-%20&reversal=false&fontAlign=13&desc=A%20Scalable%20Foundation%20Model%20Framework%20for%20Efficient%20Computational%20Pathology&fontSize=45&descSize=20&fontAlignY=35&descAlignY=50&descAlign=45&textBg=false)

We present **LitePath**, a framework designed to mitigate model over-parameterization and patch‑level redundancy. LitePath integrates **LiteFM**, a compact model distilled from three large PFMs (Virchow2, H-Optimus-1 and UNI2) using 190 million patches, and the **Adaptive Patch Selector (APS)**, a lightweight modular component for task-specific patch selection. 



## Features

- ⚡ **High Efficiency in Computational Pathology**

  **28x smaller** and **105x faster** compared to Virchow2.

- 🎯 **State-of-the-Art Accuracy**

  Deliver performance on par with leading pathology foundation models, maintaining a **99.71% average AUC retention** compared to Virchow2.

- 🌍 **Friendly for Edge Deployment**

  Easily deployable on various edge devices, including **NVIDIA Jetson Orin Nano Super**, **personal computers**, and **mobile phones**.



<p align="center"><img width=100% src="./figs/deployment.png"></p>


<p align="center"><img width=100% src="./figs/rank_all_horizon.svg"></p>

## Project Structure

```
LitePath/
├── distillation/          # Knowledge distillation training of LiteFM
├── diagnosis_prediction/  # Downstream task training and evaluation
├── inference/            # Deployment and inference
└── README.md
```



## Requirements

- Python==3.10
- PyTorch==2.3.1
- CUDA==12.9



## Quick Start

**Navigate to the `inference` directory:**

```bash
cd inference/
```

### Lightweight Patch Feature Extractor

Example Python code to extract patch features:

```python
from models import get_model
import torch

device = torch.device('cuda')
x = torch.randn(1, 3, 224, 224).to(device)
model = get_model('LiteFM', device)
feat = model(x)
print(feat.shape)
```

### Efficient PFM framework

To learn how to use the LitePath framework in a practical case, refer to the provided Jupyter Notebook: [example.ipynb](./inference/example.ipynb).



## Usage

### Stage 1: Distillation Pretraining of LiteFM

**Navigate to the `distillation` directory:**

```bash
cd distillation/
```

**1. Extract teacher features and store them to disk**

```bash
# Use DDP to speed up feature extraction:
python extract_features_ddp.py
# Merge extracted features into a single file:
python merge_features.py

### Alternative: Running on SLURM
sbatch slurm/feature_preparation/extract.slurm
```

**2.  Train LiteFM using the configuration file**

```bash
python train_lite_fast.py --config configs/litefm.yaml

### Alternative: Running on SLURM
sbatch slurm/model_training/litefm.slurm
```

- `configs/litefm.yaml` contains hyperparameters and training configuration. Modify it as needed to fit your setup.



### Stage 2: ABMIL Training and Evaluation

**Navigate to the `diagnosis_prediction` directory:**

```bash
cd diagnosis_prediction/
```

**1. Use [PrePath](https://github.com/birkhoffkiki/PrePATH) to extract the `LiteFM` features required for training and evaluation. (Our model has been integrated into PrePath.)**

**2. Train and evaluate ABMIL**

```bash
bash scripts/classification.sh
```

**3. Optional: Ablation on inference with partial patches (topk & uniformk)**

```bash
bash scripts/topk.sh
```



### Stage 3: APS Training and Evaluation

**Navigate to the `diagnosis_prediction` directory:**

```bash
cd diagnosis_prediction/
```

**1. Use [PrePath](https://github.com/birkhoffkiki/PrePATH) to extract the `LiteFM-block0` features. Here, `LiteFM-block0` refers to the intermediate shallow features from the first Transformer block**

**2. Train the APS Scoring Network**

```bash
bash scripts/aps.sh
```

**3. Grid search the optimal selection number for each task**

```bash
bash scripts/aps_gridsearch.sh
```

**4. Update the optimal selection number in `datasets/selection.json`.**

**5. Evaluate with the configuration in `selection.json`**

```bash
bash scripts/aps.sh
```

(The code will automatically scan the well-trained `ckpt` and `json` files)



## Model Variants

| **Model Name**           | **Backbone** | **#Params.** | **Teachers**                |
| ------------------------ | ------------ | ------------ | --------------------------- |
| **LiteFM (Recommended)** | ViT-S        | 22.06M       | Virchow2, H-Optimus-1, UNI2 |
| LiteFM-L                 | ViT-B        | 86.59M       | Virchow2, H-Optimus-1, UNI2 |
| LiteVirchow2             | ViT-S        | 22.06M       | Virchow2                    |
| LiteFM-S                 | ViT-Ti       | 5.72M        | Virchow2, H-Optimus-1, UNI2 |





## Acknowledgements

The work was built on top of repositories including: [PrePath](https://github.com/birkhoffkiki/PrePATH), [mSTAR](https://github.com/Innse/mSTAR), [GPFM](https://github.com/birkhoffkiki/GPFM). We thank the original authors for their excellent work!



If any questions, feel free to email [Yu Cai](mailto:yu.cai@connect.ust.hk)

