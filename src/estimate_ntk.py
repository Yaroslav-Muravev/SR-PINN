#!/usr/bin/env python3
"""
estimate_ntk.py – Оценка спектра NTK для архитектуры SR‑PINN.
Запускается независимо от обучения.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.datasets import parse_idx_train
from src.model import SRPINN
from src.data_utils import prepare_datasets
from src.config import PATH_TO_FILES

def compute_ntk_spectrum(model, train_loader, device, n_samples=200, output_component=1):
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

    points = []
    for batch in train_loader:
        coords = batch['coords']
        shape = batch['shape_params']
        patch = batch['coarse_patch']
        for i in range(coords.size(0)):
            points.append((
                coords[i:i+1].to(device),
                shape[i:i+1].to(device),
                patch[i:i+1].to(device)
            ))
            if len(points) >= n_samples:
                break
        if len(points) >= n_samples:
            break

    n = len(points)
    print(f"NTK analysis for {n} points, output component {output_component}...")

    grads = []
    for coord, shp, pch in points:
        model.zero_grad()
        pred = model(coord, shp, pch)
        out = pred[0, output_component]
        grad = torch.autograd.grad(out, model.parameters(), retain_graph=False)
        flat = torch.cat([g.detach().cpu().view(-1).float() for g in grad])
        grads.append(flat)
        del grad, flat

    num_params = grads[0].numel()
    print(f"Total parameters: {num_params:,}")

    G = torch.zeros(n, num_params, dtype=torch.float32)
    for i, g in enumerate(grads):
        G[i] = g

    K = G @ G.t()
    K = K.numpy()
    K = 0.5 * (K + K.T)

    eigvals = np.linalg.eigvalsh(K)
    return eigvals, K

def main():
    data_dir = PATH_TO_FILES
    train_ids, val_ids, test_ids = parse_idx_train(data_dir)

    MAX_TRAIN_IDS = 30
    train_ids = train_ids[:MAX_TRAIN_IDS]
    val_ids = train_ids[:1]          # <-- вот исправление
    print(f"Используется {len(train_ids)} train IDs для оценки NTK.")

    train_dataset, _, _, stats = prepare_datasets(
        data_dir, list(range(1, 101)), train_ids, val_ids, n_neighbors=8
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cpu")
    model = SRPINN(n_field_vars=7).to(device)

    n_samples = 200
    eigvals, _ = compute_ntk_spectrum(model, train_loader, device,
                                      n_samples=n_samples, output_component=1)

    print("\n--- Результаты NTK ---")
    print("Первые 10 собственных значений:", eigvals[:10])
    print("Минимальное λ_min:", eigvals[0])
    print("Максимальное λ_max:", eigvals[-1])
    cond = eigvals[-1] / eigvals[0] if eigvals[0] > 1e-12 else np.inf
    print(f"Число обусловленности: {cond:.2e}")

    lr = 2e-6
    if eigvals[0] > 1e-12:
        rate = 1.0 - lr * eigvals[0]
        steps_per_e = 1.0 / (lr * eigvals[0])
        print(f"При η = {lr:.0e}: множитель уменьшения ошибки за шаг ≈ {rate:.6f}")
        print(f"Необходимое число итераций для уменьшения ошибки в e раз: {steps_per_e:.0f}")
    else:
        print("λ_min ≈ 0: ожидается очень медленная сходимость или застревание.")

    np.savez("ntk_spectrum.npz", eigvals=eigvals, n_samples=n_samples)
    print("\nСпектр сохранён в ntk_spectrum.npz")

if __name__ == "__main__":
    main()