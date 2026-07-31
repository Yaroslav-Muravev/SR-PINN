# SR-PINN for Piezoelectric Resonator Shape Optimization

This repository contains the code and models developed for accelerating shape optimization of piezoelectric resonators using a **Super-Resolution Physics-Informed Neural Network (SR-PINN)**. 

The work focuses on cylindrical (and partly spherical) piezoelectric particles for acoustic energy harvesting and neuromodulation applications.

## 📌 Overview

Direct finite-element simulations on fine meshes are computationally expensive (~5 hours per configuration). Our approach combines:
- **Coarse FEM simulations** (15 min) for fast but low-resolution fields.
- **A neural network (SR‑PINN)** that learns to reconstruct high‑resolution fields (displacements, electric potential) from coarse inputs and geometric parameters (radius, height).
- **Optimization** (RBF interpolation + Nelder–Mead) to find the geometry that maximizes the generated electric voltage.

The method reduces evaluation time from hours to milliseconds, enabling multi‑objective shape optimization.

## 🧠 Key Features

- **Multi‑fidelity training:** Uses both coarse (cheap) and fine (expensive) simulation data.
- **Physics‑informed architecture:** Fourier‑features for coordinates, separate embeddings for shape and local coarse patches, ResNet blocks with LayerNorm and SiLU.
- **High accuracy:** Achieves ~6% relative error on test geometries for voltage prediction.
- **Scalable:** Once trained, surrogate predicts fields in seconds for any new geometry.

## 🚀 Getting Started

### Prerequisites

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Generate Data
If you want to create your own dataset, use the MATLAB scripts in folder matlab_scrpits
This will produce .mat files containing fields and CSV files with voltage values.

### Train the SR‑PINN

```bash
python src/main.py
```
The script will:
- Load and normalize data.
- Train the model using combined MSE loss.
- Save checkpoints with best validation performance.

### Optimize Shape

```bash
python src/optimization.py
```
Finds the optimal (R, H) that maximizes |V| using RBF interpolation of coarse data and Nelder‑Mead optimization.

## 📊 Results (Summary)

| Metric | Value |
|--------|-------|
| Validation voltage relative error | ~10% |
| Test voltage relative error (median) | ~6% |
| Speedup over fine FEM | ~20× |
| Optimized geometry (R, H) | (12.5 µm, 6.0 µm) |
| Voltage improvement over initial | 2.1× |

## 📖 Theory

The mathematical foundation is described in the accompanying report (see `docs/theory.pdf`). Highlights:
- Variational formulation of piezoelectricity.
- Reduction to a parametric family of cylinders.
- Existence of analytic dependence on (R,H) (Lax–Milgram + holomorphic inversion).
- NTK analysis for convergence.
- RBF interpolation and Nelder–Mead for optimization.

## 🤝 Contributing

This project is part of an ongoing research effort. If you want to contribute or use the code for your own studies, please open an issue or contact the authors.

## 📄 License

[MIT](LICENSE) (or your preferred license).

## 📧 Contact

For questions, reach out to the author yaroslav.muravev.work@yandex.ru

