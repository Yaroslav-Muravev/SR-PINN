# main1.py
# Главный скрипт для запуска обучения и тестирования SR-PINN

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import PATH_TO_FILES, CACHE_DIR, DEFAULT_NEIGHBORS
from logg import setup_logging
from data_utils import load_mat_with_cache, load_all_csv, parse_complex, prepare_datasets
from datasets import parse_idx_train, CylinderStressDataset, VoltageDataset
from model import SRPINN, train_srpinn, compute_voltage_error



def main():
    logger = setup_logging()

    data_dir = PATH_TO_FILES
    coarse_ids = list(range(1, 101))

    train_ids, val_ids, test_ids = parse_idx_train(data_dir)
    print(f"Train IDs: {train_ids}")
    print(f"Val   IDs: {val_ids}")
    print(f"Test  IDs: {test_ids}")

    train_dataset, val_dataset, colloc_dataset, stats = prepare_datasets(
        data_dir, coarse_ids, train_ids, val_ids, n_neighbors=DEFAULT_NEIGHBORS
    )

    train_ids_for_voltage = [id_ for id_ in train_ids if id_ in train_dataset.id_to_index]
    voltage_dataset = VoltageDataset(train_dataset, train_ids_for_voltage)
    voltage_loader = DataLoader(voltage_dataset, batch_size=1, shuffle=True)

    fine_df = load_all_csv(data_dir, 'results_fine')
    fine_df['voltage_complex'] = fine_df['voltage'].apply(parse_complex)
    test_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=test_ids,
        mesh_type='fine',
        n_neighbors=DEFAULT_NEIGHBORS,
        normalize=True,
        external_stats=(stats['fields_mean'], stats['fields_std']),
        subsample_ratio=1.0
    )

    coarse_coords_dict = {}
    coarse_fields_dict = {}
    for id_ in test_ids:
        try:
            coords, fields = load_mat_with_cache(id_, "coarse", data_dir)
            coarse_coords_dict[id_] = coords
            coarse_fields_dict[id_] = fields
        except FileNotFoundError:
            print(f"Warning: coarse file for ID {id_} not found, skipping")

    filtered = [(id_, coarse_coords_dict[id_], coarse_fields_dict[id_])
                for id_ in test_ids if id_ in coarse_coords_dict]
    if filtered:
        test_coarse_ids, test_coarse_coords, test_coarse_fields = zip(*filtered)
        test_dataset.set_coarse_data(list(test_coarse_coords), list(test_coarse_fields), list(test_coarse_ids))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    colloc_loader = DataLoader(colloc_dataset, batch_size=64, shuffle=True)

    model = SRPINN(n_field_vars=7)
    model = torch.compile(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    component_weights = 1.0 / (train_dataset.fields_std_np ** 2)
    component_weights = component_weights / component_weights.mean()
    component_weights = torch.tensor(component_weights, dtype=torch.float32).to(device)
    component_weights[6] *= 1e4
    component_weights[5] *= 0.1

    train_srpinn(model, train_loader, val_dataset, colloc_loader, component_weights,
                 voltage_loader, n_epochs=300, device=device, lr=5e-4,
                 pde_every=5, voltage_every=100)

    print("\n=== ОЦЕНКА НА ТЕСТОВЫХ ID ===")
    model.load_state_dict(torch.load('../best_srpinn_model_voltage.pth', map_location=device))
    model.eval()
    test_errors = compute_voltage_error(model, test_dataset, device, verbose=True, return_list=True)
    print(f"Test mean relative error: {np.mean(test_errors):.4f}")
    print(f"Test median relative error: {np.median(test_errors):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'coords_mean': stats['coords_mean'],
        'coords_std': stats['coords_std'],
        'shape_mean': stats['shape_mean'],
        'shape_std': stats['shape_std'],
        'fields_mean': stats['fields_mean'],
        'fields_std': stats['fields_std']
    }, 'best_srpinn_model_with_stats.pth')

    print("Модель сохранена: best_srpinn_model_with_stats.pth")


if __name__ == "__main__":
    main()