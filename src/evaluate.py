import torch
import numpy as np


def compute_voltage_error(model, val_dataset, device, verbose=False, return_list=False, eps=1e-20):
    """
    Вычисляет относительную ошибку предсказания напряжения |V_pred - V_true| / |V_true|
    для каждого ID в датасете.

    Аргументы:
        model: обученная модель (SRPINN)
        val_dataset: датасет (CylinderStressDataset или аналогичный)
        device: torch.device
        verbose: печатать детали по каждому ID
        return_list: вернуть список ошибок вместо среднего
        eps: для защиты от деления на ноль

    Возвращает:
        среднюю ошибку (float) или список ошибок (np.ndarray)
    """
    model.eval()
    ids_val = val_dataset.df['id'].values if hasattr(val_dataset, 'df') else []
    errors = []
    with torch.no_grad():
        for id_ in ids_val:
            slc = val_dataset.get_id_slice(id_)
            if slc.stop - slc.start == 0:
                continue
            indices = list(range(slc.start, slc.stop))
            batch_coords = []
            batch_shape = []
            batch_patch = []
            for idx in indices:
                item = val_dataset[idx]
                batch_coords.append(item['coords'].unsqueeze(0))
                batch_shape.append(item['shape_params'].unsqueeze(0))
                batch_patch.append(item['coarse_patch'].unsqueeze(0))
            if not batch_coords:
                continue
            coords = torch.cat(batch_coords, dim=0).to(device)
            shape = torch.cat(batch_shape, dim=0).to(device)
            patch = torch.cat(batch_patch, dim=0).to(device)
            pred_fields = model(coords, shape, patch).cpu().numpy()
            fields_mean = val_dataset.fields_mean_np
            fields_std = val_dataset.fields_std_np
            pred_fields_denorm = pred_fields * fields_std + fields_mean
            phi_pred = pred_fields_denorm[:, 6]  # только реальная часть phi (индекс 6)
            id_idx = val_dataset.id_to_index[id_]
            bottom_mask = val_dataset.bottom_mask_list[id_idx]
            top_mask = val_dataset.top_mask_list[id_idx]
            if not np.any(bottom_mask) or not np.any(top_mask):
                continue
            phi_bottom = phi_pred[bottom_mask].mean()
            phi_top = phi_pred[top_mask].mean()
            V_pred = phi_top - phi_bottom
            row = val_dataset.df[val_dataset.df['id'] == id_].iloc[0]
            V_true = row['voltage_complex']
            abs_pred = abs(V_pred)
            abs_true = abs(V_true)
            error = abs(V_pred - V_true) / abs_true
            errors.append(error)
            if verbose:
                print(f"ID {id_}: V_pred = {V_pred:.3e}, V_true = {V_true:.3e}")
                print(f"  relative error = {error:.4f}")
    if return_list:
        return np.array(errors)
    return np.mean(errors) if errors else float('inf')