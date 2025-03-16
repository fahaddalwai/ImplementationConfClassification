# Conformal Prediction for Deep Learning Models

This repository provides an implementation of **conformal prediction** for deep learning models. Conformal prediction offers uncertainty estimates by outputting **sets of predictions** instead of single class labels, ensuring a desired coverage probability \(1 - α\).

## 📌 Features
- **Conformal Prediction Wrapper**: `ConformalModel` class modifies neural networks to output **prediction sets**.
- **Platt Scaling**: Implements **temperature scaling** to calibrate model confidence scores.
- **Adaptive Regularization**: Automatically tunes parameters \( k_{reg} \) and \( \lambda \) for optimal prediction set sizes.
- **ImageNet & ImageNet-V2 Support**: Precomputed logits enable efficient evaluation on large-scale datasets.
- **GPU Acceleration**: Compatible with CUDA-enabled GPUs for high-speed inference.
- **Multiple Experiments**: Evaluate coverage, set size distribution, and adaptiveness across datasets.
ha
---

## 📁 Directory Structure
```bash
📦 project_root
 ┣ 📂 utilities
 ┃ ┣ 📜 conformal.py         # Core implementation of conformal prediction
 ┃ ┣ 📜 gpu_check.py         # Checks for GPU availability
 ┃ ┗ 📜 utils.py             # Utility functions for model evaluation, accuracy, etc.
 ┣ 📜 run_experiments.py     # Runs predefined experiments
 ┣ 📜 README.md              # Project documentation
 ┗ 📂 outputs                # Stores experimental results (e.g., plots, tables)
```

---

## 🚀 Quick Start
### 1️⃣ Setup Environment
```bash
pip install -r requirements.txt
```

### 2️⃣ Check GPU Availability
```bash
python utilities/gpu_check.py
```

### 3️⃣ Run Experiments
Run all experiments:
```bash
python run_experiments.py --exp all
```
Run a specific experiment (1, 2, 3, or 4):
```bash
python run_experiments.py --exp 1
```

---

## 🛠 Key Components
### 1. Conformal Prediction (`conformal.py`)
#### **Main Classes**
- **`ConformalModel(nn.Module)`**: Wraps a neural network to output conformal prediction sets.
- **`ConformalModelLogits(nn.Module)`**: Optimized version using precomputed logits.

#### **Main Functions**
- **`conformal_calibration()`**: Computes the threshold \(q̂\) for valid prediction sets.
- **`platt()`**: Implements Platt scaling for temperature adjustment.
- **`gcq()`**: Determines class inclusion in prediction sets.
- **`pick_parameters()`**: Automatically selects \( k_{reg} \) and \( \lambda \).

### 2. GPU Check (`gpu_check.py`)
Simple script to verify GPU support.

### 3. Utilities (`utils.py`)
Helper functions for:
- **Sorting & Summation** (`sort_sum()`)
- **Accuracy Computation** (`accuracy()`)
- **Model Evaluation** (`validate()`)
- **Dataset Management** (`get_model()`, `get_logits_targets()`)

### 4. Running Experiments (`run_experiments.py`)
Implements 4 experiments:
1. **Coverage vs. Set Size** on ImageNet.
2. **Coverage Evaluation** on ImageNet-V2.
3. **Set Size Distribution** for different \( \lambda \) values.
4. **Adaptiveness** comparison.

---

## 📊 Results & Visualization
- Results are saved in the `outputs/` directory.
- Run `plot_figure2()`, `plot_figure4()` functions to visualize experiment outcomes.
- Generates **LaTeX tables** for scientific reporting.



---



- Done as a part of the class CS517: Socially Responsible AI at UIC