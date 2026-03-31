import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re
import h5py
from typing import Optional, Tuple, List, Dict

path_to_files = "./files/"

# ---------------------- Вспомогательные функции ----------------------
def parse_complex(s: str) -> complex:
    s = s.strip().replace(' ', '')
    pattern = r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?([-+]\d*\.?\d+(?:[eE][-+]?\d+)?)i$'
    m = re.match(pattern, s)
    if m:
        re_part = float(m.group(1)) if m.group(1) else 0.0
        im_part = float(m.group(2))
        return complex(re_part, im_part)
    else:
        try:
            return complex(s)
        except:
            return complex(np.nan, np.nan)

def load_complex_from_h5(f, key: str) -> np.ndarray:
    data = f[key][:]
    if data.dtype.fields:
        return data['real'] + 1j * data['imag']
    else:
        return data

def build_kdtree(coords: np.ndarray) -> cKDTree:
    return cKDTree(coords)

def find_coarse_patch(coarse_tree: cKDTree, coarse_coords: np.ndarray,
                      coarse_fields: np.ndarray, query_point: np.ndarray, n_neighbors: int = 8) -> np.ndarray:
    dists, idxs = coarse_tree.query(query_point, k=n_neighbors)
    neighbor_coords = coarse_coords[idxs]
    neighbor_fields = coarse_fields[idxs]
    scale = np.mean(dists) + 1e-8
    rel_coords = (neighbor_coords - query_point) / scale
    patch = np.concatenate([neighbor_fields.ravel(), rel_coords.ravel()])
    return patch

# ---------------------- Загрузка CSV ----------------------
def load_all_csv(data_dir: str, pattern: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

# ---------------------- Датасеты (без изменений) ----------------------
class CylinderStressDataset(Dataset):
    # ... (оставлен полностью как в предыдущей версии, только __getitem__ и __init__ без изменений)
    def __init__(self, data_dir: str, csv_df: pd.DataFrame, ids: List[int], mesh_type: str,
                 n_neighbors: int = 8, normalize: bool = True,
                 external_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 subsample_ratio: float = 1.0):
        # ... (тот же код, что был в предыдущей версии)
        self.data_dir = data_dir
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.mesh_type = mesh_type
        self.subsample_ratio = subsample_ratio
        self.df = csv_df[csv_df['id'].isin(ids)].reset_index(drop=True)
        self.df['voltage_complex'] = self.df['voltage'].apply(parse_complex)
        self.coords_list = []
        self.fields_list = []
        self.shape_params = []
        self.id_to_index = {}
        self.z_min_list = []
        self.z_max_list = []
        self.bottom_mask_list = []
        self.top_mask_list = []
        self.total_points = 0
        self.cumulative_sizes = [0]

        for idx, row in self.df.iterrows():
            id_ = int(row['id'])
            r_um = row['r_um']
            h_um = row['h_um']
            fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_{mesh_type}.mat')
            if not os.path.exists(fname):
                print(f"Warning: file {fname} not found, skipping id {id_}")
                continue
            with h5py.File(fname, 'r') as f:
                X = f['X'][:]; Y = f['Y'][:]; Z = f['Z'][:]
                ux = load_complex_from_h5(f, 'ux')
                uy = load_complex_from_h5(f, 'uy')
                uz = load_complex_from_h5(f, 'uz')
                phi = load_complex_from_h5(f, 'phi')
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            fields = np.stack([ux.real.ravel(), ux.imag.ravel(),
                               uy.real.ravel(), uy.imag.ravel(),
                               uz.real.ravel(), uz.imag.ravel(),
                               phi.real.ravel(), phi.imag.ravel()], axis=1)

            z_vals = Z.ravel()
            z_min = z_vals.min(); z_max = z_vals.max()
            self.z_min_list.append(z_min); self.z_max_list.append(z_max)
            eps_z = 1e-6 * (z_max - z_min)
            bottom_mask = np.abs(z_vals - z_min) < eps_z
            top_mask = np.abs(z_vals - z_max) < eps_z

            if subsample_ratio < 1.0:
                n_total = coords.shape[0]
                n_keep = int(n_total * subsample_ratio)
                idx_keep = np.random.choice(n_total, n_keep, replace=False)
                coords = coords[idx_keep]
                fields = fields[idx_keep]
                bottom_mask = bottom_mask[idx_keep]
                top_mask = top_mask[idx_keep]

            self.coords_list.append(coords)
            self.fields_list.append(fields)
            self.shape_params.append([r_um, h_um])
            self.id_to_index[id_] = len(self.coords_list) - 1
            self.bottom_mask_list.append(bottom_mask)
            self.top_mask_list.append(top_mask)
            self.total_points += coords.shape[0]
            self.cumulative_sizes.append(self.total_points)

        # Нормализация
        if normalize and self.total_points > 0:
            all_coords = np.vstack(self.coords_list)
            self.coords_mean = all_coords.mean(axis=0)
            self.coords_std = np.maximum(all_coords.std(axis=0), 1e-8)

            all_shape = np.array(self.shape_params)
            self.shape_mean = all_shape.mean(axis=0)
            self.shape_std = np.maximum(all_shape.std(axis=0), 1e-8)

            if external_stats is not None:
                self.fields_mean, self.fields_std = external_stats
                print(f"Using external fields stats: mean[6]={self.fields_mean[6]:.3e}, std[6]={self.fields_std[6]:.3e}")
            else:
                all_fields = np.vstack(self.fields_list)
                self.fields_mean = all_fields.mean(axis=0)
                self.fields_std = all_fields.std(axis=0)
                self.fields_std = np.maximum(self.fields_std, 1e-20)   # мягкий clamp
                print(f"Computed from fine: mean[6]={self.fields_mean[6]:.3e}, std[6]={self.fields_std[6]:.3e}")
        else:
            # fallback
            self.coords_mean = np.zeros(3); self.coords_std = np.ones(3)
            self.shape_mean = np.zeros(2);  self.shape_std = np.ones(2)
            self.fields_mean = np.zeros(8); self.fields_std = np.ones(8)

        self.coarse_trees = None
        self.coarse_coords = None
        self.coarse_fields = None

    def set_coarse_data(self, coarse_coords_list, coarse_fields_list, coarse_ids):
        self.coarse_trees = {}
        self.coarse_coords = {}
        self.coarse_fields = {}
        for idx, id_ in enumerate(coarse_ids):
            if id_ in self.id_to_index:
                self.coarse_trees[id_] = build_kdtree(coarse_coords_list[idx])
                self.coarse_coords[id_] = coarse_coords_list[idx]
                self.coarse_fields[id_] = coarse_fields_list[idx]

    def get_id_slice(self, id_: int) -> slice:
        idx = self.id_to_index.get(id_)
        if idx is None:
            return slice(0, 0)
        return slice(self.cumulative_sizes[idx], self.cumulative_sizes[idx + 1])

    def __len__(self):
        return self.total_points

    def __getitem__(self, idx):
        if self.total_points == 0:
            return {}
        id_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[id_idx]
        id_ = int(self.df.iloc[id_idx]['id'])
        coords = self.coords_list[id_idx][local_idx]
        fields = self.fields_list[id_idx][local_idx]
        shape = np.array(self.shape_params[id_idx])

        patch = np.zeros(self.n_neighbors * (8 + 3))
        if self.coarse_trees is not None and id_ in self.coarse_trees:
            patch = find_coarse_patch(self.coarse_trees[id_], self.coarse_coords[id_],
                                      self.coarse_fields[id_], coords, self.n_neighbors)

        if self.normalize:
            coords = (coords - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std
            n_fields_flat = self.n_neighbors * 8
            patch_fields = patch[:n_fields_flat].reshape(self.n_neighbors, 8)
            patch_rel = patch[n_fields_flat:].reshape(self.n_neighbors, 3)
            patch_fields_norm = (patch_fields - self.fields_mean) / (self.fields_std + 1e-20)
            patch = np.concatenate([patch_fields_norm.ravel(), patch_rel.ravel()])

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32),
            'coarse_patch': torch.tensor(patch, dtype=torch.float32),
            'target': torch.tensor(fields, dtype=torch.float32),
            'id': id_,
            'fields_mean': torch.tensor(self.fields_mean, dtype=torch.float32),
            'fields_std':  torch.tensor(self.fields_std,  dtype=torch.float32)
        }

# ---------------------- Датасет для коллокационных точек ----------------------
class CollocationDataset(Dataset):
    def __init__(self,
                 ids: List[int],
                 shape_params: Dict[int, Tuple[float, float]],
                 coarse_data: Dict[int, Tuple[cKDTree, np.ndarray, np.ndarray]],
                 n_points_per_id: int = 200,
                 n_neighbors: int = 8,
                 normalize: bool = True,
                 coords_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 shape_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 fields_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.ids = ids
        self.shape_params = shape_params
        self.coarse_data = coarse_data
        self.n_points_per_id = n_points_per_id
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.coords_mean, self.coords_std = coords_stats if coords_stats else (np.zeros(3), np.ones(3))
        self.shape_mean, self.shape_std = shape_stats if shape_stats else (np.zeros(2), np.ones(2))
        self.fields_mean, self.fields_std = fields_stats if fields_stats else (np.zeros(8), np.ones(8))
        self.total_points = len(ids) * n_points_per_id

    def __len__(self):
        return self.total_points

    def __getitem__(self, idx):
        id_idx = idx // self.n_points_per_id
        local_idx = idx % self.n_points_per_id
        id_ = self.ids[id_idx]
        r_um, h_um = self.shape_params[id_]
        R = r_um * 1e-6
        H = h_um * 1e-6
        z0 = -H / 2
        u = np.random.random()
        v = np.random.random()
        w = np.random.random()
        r = R * np.sqrt(u)
        theta = 2 * np.pi * v
        z = z0 + H * w
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        point = np.array([x, y, z])
        tree, coarse_coords, coarse_fields = self.coarse_data[id_]
        patch = find_coarse_patch(tree, coarse_coords, coarse_fields, point, self.n_neighbors)
        shape = np.array([r_um, h_um])
        if self.normalize:
            point = (point - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std
            n_fields_flat = self.n_neighbors * 8
            patch_fields = patch[:n_fields_flat].reshape(self.n_neighbors, 8)
            patch_rel = patch[n_fields_flat:].reshape(self.n_neighbors, 3)
            patch_fields_norm = (patch_fields - self.fields_mean) / self.fields_std
            patch = np.concatenate([patch_fields_norm.ravel(), patch_rel.ravel()])
        return {
            'coords': torch.tensor(point, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32),
            'coarse_patch': torch.tensor(patch, dtype=torch.float32),
            'id': id_
        }

# ---------------------- Модель SR-PINN ----------------------
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int = 128, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.linear2(x)
        return x + residual

class SRPINN(nn.Module):
    def __init__(self,
                 n_spatial: int = 3,
                 n_shape_params: int = 2,
                 n_coarse_nodes: int = 8,
                 n_field_vars: int = 8,
                 hidden_dim: int = 256,      # увеличено для стабильности
                 n_blocks: int = 6,         # увеличено
                 fourier_mapping_size: int = 128,
                 fourier_scale: float = 5.0):
        super().__init__()
        self.n_coarse_nodes = n_coarse_nodes
        self.n_field_vars = n_field_vars
        self.fourier_embed = FourierFeatureEmbedding(n_spatial, fourier_mapping_size, fourier_scale)
        fourier_dim = 2 * fourier_mapping_size
        self.shape_embed = nn.Linear(n_shape_params, hidden_dim)
        coarse_input_dim = n_coarse_nodes * (n_field_vars + 3)
        self.coarse_embed = nn.Linear(coarse_input_dim, hidden_dim)
        self.input_proj = nn.Linear(fourier_dim + hidden_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, n_field_vars)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_coords, shape_params, coarse_patch):
        fourier_feat = self.fourier_embed(spatial_coords)
        shape_feat = self.shape_embed(shape_params)
        coarse_feat = self.coarse_embed(coarse_patch)
        x = torch.cat([fourier_feat, shape_feat, coarse_feat], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x)
        return out

# ---------------------- Функция потерь (с PINN-регуляризацией + voltage supervision) ----------------------
# ---------------------- Функция потерь (финальная версия) ----------------------
class StressPINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.mse = nn.MSELoss()

    def forward(self, model, batch, batch_pde=None):
        device = next(model.parameters()).device
        loss_data = torch.tensor(0.0, device=device)

        if 'target' in batch:
            pred = model(batch['coords'], batch['shape_params'], batch['coarse_patch'])
            loss_data = self.mse(pred, batch['target'])

        # NO voltage term here (it was broken)
        # NO PDE term here (it was harmful)

        total_loss = self.lambda_data * loss_data
        return total_loss, {
            'loss_data': loss_data.item(),
            'total_loss': total_loss.item()
        }

# ---------------------- Подготовка данных (ИСПРАВЛЕНА: отдельные train/val) ----------------------
# ---------------------- Подготовка данных (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ) ----------------------
def prepare_datasets(data_dir: str,
                     coarse_ids: List[int],
                     train_ids: List[int],
                     val_ids: List[int],
                     n_neighbors: int = 8):
    """
    Подготовка всех датасетов с правильной нормализацией:
    - Статистика mean/std для полей считается ТОЛЬКО по fine-данным (target)
    - Coarse-данные используются только для patch (не для статистики)
    """
    fine_df = load_all_csv(data_dir, 'results_fine')
    coarse_df = load_all_csv(data_dir, 'results_coarse')

    if fine_df.empty:
        raise ValueError("Не найдены results_fine*.csv")

    # === 1. Статистика полей — ТОЛЬКО ПО FINE-ДАННЫМ (целевая правда) ===
    all_fine_fields = []
    for id_ in train_ids:                     # только train, чтобы не было утечки
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            ux = load_complex_from_h5(f, 'ux')
            uy = load_complex_from_h5(f, 'uy')
            uz = load_complex_from_h5(f, 'uz')
            phi = load_complex_from_h5(f, 'phi')
        fields = np.stack([
            ux.real.ravel(), ux.imag.ravel(),
            uy.real.ravel(), uy.imag.ravel(),
            uz.real.ravel(), uz.imag.ravel(),
            phi.real.ravel(), phi.imag.ravel()
        ], axis=1)
        all_fine_fields.append(fields)

    if all_fine_fields:
        all_fields_stack = np.vstack(all_fine_fields)
        fields_mean = all_fields_stack.mean(axis=0)
        fields_std = all_fields_stack.std(axis=0)
        fields_std = np.maximum(fields_std, 1e-20)          # мягкий clamp — phi не обнуляется
        print(f"Fields stats FROM FINE data: mean[6]={fields_mean[6]:.3e}, "
              f"std[6]={fields_std[6]:.3e} | mean[7]={fields_mean[7]:.3e}, std[7]={fields_std[7]:.3e}")
    else:
        fields_mean = np.zeros(8)
        fields_std = np.ones(8)

    external_stats = (fields_mean, fields_std)

    # === 2. Создаём train и val датасеты ===
    train_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=train_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0          # ускорение обучения
    )

    val_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=val_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0          # полная валидация
    )

    # === 3. Загружаем coarse-данные только для train + val ===
    coarse_coords_list = []
    coarse_fields_list = []
    coarse_id_list = []
    needed_ids = set(train_ids + val_ids)

    for id_ in needed_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
        if not os.path.exists(fname):
            print(f"Warning: coarse file for ID {id_} not found")
            continue
        with h5py.File(fname, 'r') as f:
            X = f['X'][:]
            Y = f['Y'][:]
            Z = f['Z'][:]
            ux = load_complex_from_h5(f, 'ux')
            uy = load_complex_from_h5(f, 'uy')
            uz = load_complex_from_h5(f, 'uz')
            phi = load_complex_from_h5(f, 'phi')
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        fields = np.stack([
            ux.real.ravel(), ux.imag.ravel(),
            uy.real.ravel(), uy.imag.ravel(),
            uz.real.ravel(), uz.imag.ravel(),
            phi.real.ravel(), phi.imag.ravel()
        ], axis=1)
        coarse_coords_list.append(coords)
        coarse_fields_list.append(fields)
        coarse_id_list.append(id_)

    # Привязываем coarse к датасетам
    def set_coarse(ds, ids):
        idxs = [i for i, cid in enumerate(coarse_id_list) if cid in ids]
        if idxs:
            ds.set_coarse_data(
                [coarse_coords_list[i] for i in idxs],
                [coarse_fields_list[i] for i in idxs],
                [coarse_id_list[i] for i in idxs]
            )

    set_coarse(train_dataset, train_ids)
    set_coarse(val_dataset, val_ids)

    # === 4. Collocation dataset (только для train) ===
    colloc_ids = [id_ for id_ in train_ids if id_ in needed_ids]
    coarse_data_colloc = {}
    shape_params_colloc = {}

    # Словарь shape_params из coarse CSV
    coarse_shape_dict = {}
    for id_ in needed_ids:
        row = coarse_df[coarse_df['id'] == id_]
        if not row.empty:
            coarse_shape_dict[id_] = (row.iloc[0]['r_um'], row.iloc[0]['h_um'])

    for id_ in colloc_ids:
        try:
            idx = coarse_id_list.index(id_)
        except ValueError:
            continue
        tree = build_kdtree(coarse_coords_list[idx])
        coarse_data_colloc[id_] = (
            tree,
            coarse_coords_list[idx],
            coarse_fields_list[idx]
        )
        if id_ in coarse_shape_dict:
            shape_params_colloc[id_] = coarse_shape_dict[id_]

    colloc_dataset = CollocationDataset(
        ids=colloc_ids,
        shape_params=shape_params_colloc,
        coarse_data=coarse_data_colloc,
        n_points_per_id=200,
        n_neighbors=n_neighbors,
        normalize=True,
        coords_stats=(train_dataset.coords_mean, train_dataset.coords_std),
        shape_stats=(train_dataset.shape_mean, train_dataset.shape_std),
        fields_stats=external_stats
    )

    # === 5. Статистики для сохранения ===
    stats = {
        'coords_mean': train_dataset.coords_mean,
        'coords_std': train_dataset.coords_std,
        'shape_mean': train_dataset.shape_mean,
        'shape_std': train_dataset.shape_std,
        'fields_mean': torch.tensor(fields_mean, dtype=torch.float32),
        'fields_std': torch.tensor(fields_std, dtype=torch.float32)
    }

    return train_dataset, val_dataset, colloc_dataset, stats

# ---------------------- Вычисление ошибки напряжения ----------------------
def compute_voltage_error(model, val_dataset, device, verbose=False):
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
            fields_mean = val_dataset.fields_mean
            fields_std = val_dataset.fields_std
            pred_fields_denorm = pred_fields * fields_std + fields_mean
            phi_pred = pred_fields_denorm[:, 6] + 1j * pred_fields_denorm[:, 7]
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
            error = abs(V_pred - V_true) / (abs(V_true) + 1e-15)
            errors.append(error)
            if verbose:
                print(f"ID {id_}: bottom_mask sum={bottom_mask.sum()}, top_mask sum={top_mask.sum()}")
                print(f"  phi_bottom mean = {phi_bottom:.3e}, phi_top mean = {phi_top:.3e}")
                print(f"  V_pred = {V_pred:.3e}, V_true = {V_true:.3e}")
                print(f"  relative error = {error:.4f}")
    return np.mean(errors) if errors else float('inf')

# ---------------------- Обучение ----------------------
def train_srpinn(model, train_loader, val_dataset, colloc_loader, n_epochs, device, lr=5e-4, pde_every=5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    #criterion = StressPINNLoss(lambda_data=1.0, lambda_pde=0.05, lambda_voltage=15.0)
    criterion = StressPINNLoss(lambda_data=1.0)
    best_voltage_error = float('inf')
    colloc_iter = iter(colloc_loader)
    step_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            compute_pde = (step_counter % pde_every == 0)
            step_counter += 1
            if compute_pde:
                try:
                    batch_pde = next(colloc_iter)
                except StopIteration:
                    colloc_iter = iter(colloc_loader)
                    batch_pde = next(colloc_iter)
                for k in ['coords', 'shape_params', 'coarse_patch']:
                    batch_pde[k] = batch_pde[k].to(device)
            else:
                batch_pde = None

            for k in ['coords', 'shape_params', 'coarse_patch', 'target']:
                if k in batch:
                    batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            loss, loss_dict = criterion(model, batch, batch_pde)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
                val_losses = []
                for vbatch in val_loader:
                    for k in ['coords', 'shape_params', 'coarse_patch', 'target']:
                        if k in vbatch:
                            vbatch[k] = vbatch[k].to(device)
                    _, loss_dict = criterion(model, vbatch, None)
                    val_losses.append(loss_dict)
                avg_val_loss = np.mean([d['total_loss'] for d in val_losses])

            voltage_error = compute_voltage_error(model, val_dataset, device, verbose=True)
            avg_train_loss = np.mean([d['total_loss'] for d in train_losses])

            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={avg_val_loss:.6f}, voltage_rel_error={voltage_error:.4f}")

            if voltage_error < best_voltage_error:
                best_voltage_error = voltage_error
                torch.save(model.state_dict(), 'best_srpinn_model_voltage.pth')
                print(f"  >>> Saved best model (voltage error {best_voltage_error:.4f}) <<<")

# ---------------------- Запуск ----------------------
if __name__ == "__main__":
    data_dir = path_to_files
    coarse_ids = list(range(1, 101))

    # === ИСПРАВЛЕННОЕ РАЗДЕЛЕНИЕ ===
    train_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]   # обучение
    val_ids   = [14, 15]                     # валидация (отдельный ID!)
    test_ids  = [45, 46, 62, 75, 83, 86] # тест

    print(f"Train IDs: {train_ids}")
    print(f"Val   IDs: {val_ids}")
    print(f"Test  IDs: {test_ids}")

    # Подготовка (теперь с отдельными train/val)
    train_dataset, val_dataset, colloc_dataset, stats = prepare_datasets(
        data_dir, coarse_ids, train_ids, val_ids, n_neighbors=8
    )

    # Test dataset (полный)
    fine_df = load_all_csv(data_dir, 'results_fine')
    fine_df['voltage_complex'] = fine_df['voltage'].apply(parse_complex)
    test_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=test_ids,
        mesh_type='fine',
        n_neighbors=8,
        normalize=True,
        external_stats=(stats['fields_mean'], stats['fields_std']),
        subsample_ratio=1.0
    )

    # Coarse для test
    coarse_coords_dict = {}
    coarse_fields_dict = {}
    for id_ in test_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
        if os.path.exists(fname):
            with h5py.File(fname, 'r') as f:
                X = f['X'][:]
                Y = f['Y'][:]
                Z = f['Z'][:]
                ux = load_complex_from_h5(f, 'ux')
                uy = load_complex_from_h5(f, 'uy')
                uz = load_complex_from_h5(f, 'uz')
                phi = load_complex_from_h5(f, 'phi')
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            fields = np.stack([ux.real.ravel(), ux.imag.ravel(),
                               uy.real.ravel(), uy.imag.ravel(),
                               uz.real.ravel(), uz.imag.ravel(),
                               phi.real.ravel(), phi.imag.ravel()], axis=1)
            coarse_coords_dict[id_] = coords
            coarse_fields_dict[id_] = fields

    test_coarse_coords = [coarse_coords_dict[id_] for id_ in test_ids if id_ in coarse_coords_dict]
    test_coarse_fields = [coarse_fields_dict[id_] for id_ in test_ids if id_ in coarse_coords_dict]
    test_coarse_ids = [id_ for id_ in test_ids if id_ in coarse_coords_dict]
    test_dataset.set_coarse_data(test_coarse_coords, test_coarse_fields, test_coarse_ids)

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    colloc_loader = DataLoader(colloc_dataset, batch_size=64, shuffle=True)

    # Модель
    model = SRPINN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Обучение
    train_srpinn(model, train_loader, val_dataset, colloc_loader,
                 n_epochs=300, device=device, lr=5e-4, pde_every=5)

    # Тест
    print("\n=== ОЦЕНКА НА ТЕСТОВЫХ ID ===")
    model.load_state_dict(torch.load('best_srpinn_model_voltage.pth', map_location=device))
    model.eval()
    test_errors = compute_voltage_error(model, test_dataset, device, verbose=True)
    print(f"Test mean voltage error : {np.mean(test_errors):.4f}")
    print(f"Test median voltage error: {np.median(test_errors):.4f}")

    # Сохранение с статистиками
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