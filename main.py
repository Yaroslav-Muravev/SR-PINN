import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.io as sio
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
    """Парсит строку вида '-7.83e-11-4.15e-12i' в комплексное число."""
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

def normalize_global_per_channel(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < eps] = 1.0
    return (data - mean) / std, mean, std

def denormalize_per_channel(data_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data_norm * std + mean

def build_kdtree(coords: np.ndarray) -> cKDTree:
    return cKDTree(coords)

def find_coarse_patch(coarse_tree: cKDTree,
                      coarse_coords: np.ndarray,
                      coarse_fields: np.ndarray,
                      query_point: np.ndarray,
                      n_neighbors: int = 8) -> np.ndarray:
    dists, idxs = coarse_tree.query(query_point, k=n_neighbors)
    neighbor_coords = coarse_coords[idxs]
    neighbor_fields = coarse_fields[idxs]
    scale = np.mean(dists) + 1e-8
    rel_coords = (neighbor_coords - query_point) / scale
    patch = np.concatenate([neighbor_fields.ravel(), rel_coords.ravel()])
    return patch

# ---------------------- Загрузка всех CSV ----------------------
def load_all_csv(data_dir: str, pattern: str) -> pd.DataFrame:
    """
    Загружает все CSV-файлы, начинающиеся с pattern, и объединяет в один DataFrame.
    pattern: например 'results_coarse' или 'results_fine'
    """
    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# ---------------------- Датасет для fine/coarse данных ----------------------
class CylinderStressDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 csv_df: pd.DataFrame,   # объединённый DataFrame для нужного mesh_type
                 ids: List[int],
                 mesh_type: str,
                 n_neighbors: int = 8,
                 normalize: bool = True):
        self.data_dir = data_dir
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.mesh_type = mesh_type

        # Оставляем только строки с нужными id
        self.df = csv_df[csv_df['id'].isin(ids)].reset_index(drop=True)
        # Парсим voltage
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

            self.coords_list.append(coords)
            self.fields_list.append(fields)
            self.shape_params.append([r_um, h_um])
            self.id_to_index[id_] = len(self.coords_list) - 1

            z_vals = Z.ravel()
            z_min = z_vals.min()
            z_max = z_vals.max()
            self.z_min_list.append(z_min)
            self.z_max_list.append(z_max)
            eps_z = 1e-6 * (z_max - z_min)
            bottom_mask = np.abs(z_vals - z_min) < eps_z
            top_mask = np.abs(z_vals - z_max) < eps_z
            self.bottom_mask_list.append(bottom_mask)
            self.top_mask_list.append(top_mask)

            self.total_points += coords.shape[0]
            self.cumulative_sizes.append(self.total_points)

        self.n_ids = len(self.coords_list)

        if normalize and self.total_points > 0:
            all_coords = np.vstack(self.coords_list)
            self.coords_mean = all_coords.mean(axis=0)
            self.coords_std = all_coords.std(axis=0)
            self.coords_std[self.coords_std < 1e-8] = 1.0

            all_shape = np.array(self.shape_params)
            self.shape_mean = all_shape.mean(axis=0)
            self.shape_std = all_shape.std(axis=0)
            self.shape_std[self.shape_std < 1e-8] = 1.0

            all_fields = np.vstack(self.fields_list)
            self.fields_mean = all_fields.mean(axis=0)
            self.fields_std = all_fields.std(axis=0)
            self.fields_std[self.fields_std < 1e-8] = 1.0

            print(f"\n=== Dataset {mesh_type} stats ===")
            print(f"coords_mean = {self.coords_mean}")
            print(f"coords_std  = {self.coords_std}")
            print(f"shape_mean  = {self.shape_mean}")
            print(f"shape_std   = {self.shape_std}")
            print(f"fields_mean (phi real) = {self.fields_mean[6]:.3e}")
            print(f"fields_std  (phi real) = {self.fields_std[6]:.3e}")
            print("==============================\n")
        else:
            # fallback для пустого датасета
            self.coords_mean = np.zeros(3)
            self.coords_std = np.ones(3)
            self.shape_mean = np.zeros(2)
            self.shape_std = np.ones(2)
            self.fields_mean = np.zeros(8)
            self.fields_std = np.ones(8)

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
            return slice(0,0)
        start = self.cumulative_sizes[idx]
        end = self.cumulative_sizes[idx+1]
        return slice(start, end)

    def __len__(self):
        return self.total_points

    def __getitem__(self, idx):
        if self.total_points == 0:
            # пустой датасет
            return {}
        id_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[id_idx]

        id_ = int(self.df.iloc[id_idx]['id'])
        coords = self.coords_list[id_idx][local_idx]
        fields = self.fields_list[id_idx][local_idx]
        shape = np.array(self.shape_params[id_idx])

        if self.coarse_trees is not None and id_ in self.coarse_trees:
            tree = self.coarse_trees[id_]
            coarse_coords = self.coarse_coords[id_]
            coarse_fields = self.coarse_fields[id_]
            patch = find_coarse_patch(tree, coarse_coords, coarse_fields,
                                      coords, self.n_neighbors)
        else:
            patch = np.zeros(self.n_neighbors * (8 + 3))

        if self.normalize:
            coords = (coords - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std
            fields = (fields - self.fields_mean) / self.fields_std
            n_fields_flat = self.n_neighbors * 8
            patch_fields = patch[:n_fields_flat].reshape(self.n_neighbors, 8)
            patch_rel = patch[n_fields_flat:].reshape(self.n_neighbors, 3)
            patch_fields_norm = (patch_fields - self.fields_mean) / self.fields_std
            patch = np.concatenate([patch_fields_norm.ravel(), patch_rel.ravel()])

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32),
            'coarse_patch': torch.tensor(patch, dtype=torch.float32),
            'target': torch.tensor(fields, dtype=torch.float32),
            'id': id_
        }

# ---------------------- Датасет для коллокационных точек ----------------------
class CollocationDataset(Dataset):
    def __init__(self,
                 ids: List[int],
                 shape_params: Dict[int, Tuple[float, float]],
                 coarse_data: Dict[int, Tuple[np.ndarray, np.ndarray, cKDTree]],
                 n_points_per_id: int = 1000,
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
        z0 = -H/2

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
                 hidden_dim: int = 256,
                 n_blocks: int = 8,
                 fourier_mapping_size: int = 128,
                 fourier_scale: float = 10.0):
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

# ---------------------- Функция потерь (только data loss) ----------------------
class StressPINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.mse = nn.MSELoss()

    def forward(self, model, batch, batch_pde=None):
        loss_data = torch.tensor(0.0, device=next(model.parameters()).device)
        if 'target' in batch:
            pred = model(batch['coords'], batch['shape_params'], batch['coarse_patch'])
            loss_data = self.mse(pred, batch['target'])
        total_loss = self.lambda_data * loss_data
        return total_loss, {'loss_data': loss_data.item(), 'total_loss': total_loss.item()}

# ---------------------- Подготовка данных и обучение ----------------------
def prepare_datasets(data_dir: str,
                     coarse_ids: List[int],
                     fine_ids: List[int],
                     n_neighbors: int = 8):
    """
    Загружает все coarse и fine данные, строит k-d деревья.
    coarse_ids: список всех ID, для которых есть coarse .mat (можно передать все 1..100)
    fine_ids: список ID, для которых есть fine .mat (извлечём из наличия файлов)
    """
    # 1. Загружаем все CSV
    coarse_df = load_all_csv(data_dir, 'results_coarse')
    fine_df = load_all_csv(data_dir, 'results_fine')

    if coarse_df.empty or fine_df.empty:
        raise ValueError("Не найдены CSV файлы для coarse или fine")

    # 2. Загружаем coarse данные для всех coarse_ids (или для тех, у кого есть файлы)
    coarse_coords_list = []
    coarse_fields_list = []
    coarse_shape_dict = {}
    coarse_id_list = []

    for id_ in coarse_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
        if not os.path.exists(fname):
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
        # Параметры из coarse CSV
        row = coarse_df[coarse_df['id'] == id_]
        if not row.empty:
            coarse_shape_dict[id_] = (row.iloc[0]['r_um'], row.iloc[0]['h_um'])

    # 3. Fine датасеты: используем только те fine_ids, для которых есть fine .mat
    valid_fine_ids = []
    for id_ in fine_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
        if os.path.exists(fname):
            valid_fine_ids.append(id_)
        else:
            print(f"Warning: fine file for id {id_} not found, skipping")

    train_ids = valid_fine_ids[:]   # можно разделить позже
    val_ids = []   # если хотим валидацию на отдельном ID, пока оставим пустым

    if len(train_ids) == 0:
        raise ValueError("Нет доступных fine файлов для обучения")

    # 4. Создаём датасеты
    train_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=train_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True
    )
    val_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=val_ids if val_ids else [train_ids[0]],   # если нет валидационных, используем первый train для валидации
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True
    )

    # 5. Передаём coarse данные в датасеты
    coarse_for_train = {id_: (coarse_coords_list[i], coarse_fields_list[i], build_kdtree(coarse_coords_list[i]))
                        for i, id_ in enumerate(coarse_id_list) if id_ in train_ids}
    coarse_for_val = {id_: (coarse_coords_list[i], coarse_fields_list[i], build_kdtree(coarse_coords_list[i]))
                      for i, id_ in enumerate(coarse_id_list) if id_ in val_ids}

    train_dataset.set_coarse_data(
        [coarse_for_train[id_][0] for id_ in train_ids if id_ in coarse_for_train],
        [coarse_for_train[id_][1] for id_ in train_ids if id_ in coarse_for_train],
        [id_ for id_ in train_ids if id_ in coarse_for_train]
    )
    val_dataset.set_coarse_data(
        [coarse_for_val[id_][0] for id_ in val_ids if id_ in coarse_for_val],
        [coarse_for_val[id_][1] for id_ in val_ids if id_ in coarse_for_val],
        [id_ for id_ in val_ids if id_ in coarse_for_val]
    )

    # 6. Коллокационные точки (только для train ids)
    colloc_ids = [id_ for id_ in train_ids if id_ in coarse_for_train]
    coarse_data_colloc = {id_: (coarse_for_train[id_][2], coarse_for_train[id_][0], coarse_for_train[id_][1])
                          for id_ in colloc_ids}
    shape_params_colloc = {id_: coarse_shape_dict[id_] for id_ in colloc_ids if id_ in coarse_shape_dict}

    colloc_dataset = CollocationDataset(
        ids=list(coarse_data_colloc.keys()),
        shape_params=shape_params_colloc,
        coarse_data=coarse_data_colloc,
        n_points_per_id=1000,
        n_neighbors=n_neighbors,
        normalize=True,
        coords_stats=(train_dataset.coords_mean, train_dataset.coords_std),
        shape_stats=(train_dataset.shape_mean, train_dataset.shape_std),
        fields_stats=(train_dataset.fields_mean, train_dataset.fields_std)
    )

    stats = {
        'coords_mean': train_dataset.coords_mean,
        'coords_std': train_dataset.coords_std,
        'shape_mean': train_dataset.shape_mean,
        'shape_std': train_dataset.shape_std,
        'fields_mean': train_dataset.fields_mean,
        'fields_std': train_dataset.fields_std
    }
    return train_dataset, val_dataset, colloc_dataset, stats

# ---------------------- Функция вычисления ошибки напряжения ----------------------
def compute_voltage_error(model, val_dataset, device, verbose=False):
    model.eval()
    ids_val = val_dataset.df['id'].values if val_dataset.df is not None else []
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

            phi_re = pred_fields_denorm[:, 6]

            id_idx = val_dataset.id_to_index[id_]
            bottom_mask = val_dataset.bottom_mask_list[id_idx]
            top_mask = val_dataset.top_mask_list[id_idx]

            if not np.any(bottom_mask) or not np.any(top_mask):
                if verbose:
                    print(f"ID {id_}: bottom_mask sum = {bottom_mask.sum()}, top_mask sum = {top_mask.sum()} -> skipping")
                continue

            phi_bottom = phi_re[bottom_mask].mean()
            phi_top = phi_re[top_mask].mean()
            V_pred = phi_top - phi_bottom

            row = val_dataset.df[val_dataset.df['id'] == id_].iloc[0]
            V_true = row['voltage_complex']

            error = abs(V_pred - V_true.real) / (abs(V_true.real) + 1e-8)
            errors.append(error)

            if verbose and id_ == ids_val[0]:
                print(f"ID {id_}: bottom_mask sum={bottom_mask.sum()}, top_mask sum={top_mask.sum()}")
                print(f"  phi_bottom mean = {phi_bottom:.3e}, phi_top mean = {phi_top:.3e}")
                print(f"  V_pred = {V_pred:.3e}, V_true.real = {V_true.real:.3e}")
                print(f"  relative error = {error:.4f}")

    return np.mean(errors) if errors else float('inf')

# ---------------------- Основной цикл обучения ----------------------
def train_srpinn(model, train_loader, val_dataset, colloc_loader, n_epochs, device, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = StressPINNLoss(lambda_data=1.0)

    best_val_loss = float('inf')
    best_voltage_error = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            # Получаем батч коллокационных точек (даже если не используем, для совместимости)
            try:
                batch_pde = next(iter(colloc_loader))
            except StopIteration:
                colloc_iter = iter(colloc_loader)
                batch_pde = next(colloc_iter)

            for k in ['coords', 'shape_params', 'coarse_patch', 'target']:
                if k in batch:
                    batch[k] = batch[k].to(device)
            for k in ['coords', 'shape_params', 'coarse_patch']:
                if k in batch_pde:
                    batch_pde[k] = batch_pde[k].to(device)

            optimizer.zero_grad()
            loss, loss_dict = criterion(model, batch, batch_pde)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss_dict)

        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
                for batch in val_loader:
                    for k in ['coords', 'shape_params', 'coarse_patch', 'target']:
                        if k in batch:
                            batch[k] = batch[k].to(device)
                    loss, loss_dict = criterion(model, batch, batch_pde=None)
                    val_losses.append(loss_dict)

            avg_train = {k: np.mean([d[k] for d in train_losses]) for k in train_losses[0]}
            avg_val_loss = np.mean([d['total_loss'] for d in val_losses])

            voltage_error = compute_voltage_error(model, val_dataset, device, verbose=True)

            print(f"Epoch {epoch}: train_loss={avg_train['total_loss']:.6f}, "
                  f"val_loss={avg_val_loss:.6f}, voltage_rel_error={voltage_error:.4f}")

            if voltage_error < best_voltage_error:
                best_voltage_error = voltage_error
                torch.save(model.state_dict(), 'best_srpinn_model_voltage.pth')
                print(f"  Saved best model (voltage error {best_voltage_error:.4f})")

# ---------------------- Пример запуска ----------------------
if __name__ == "__main__":
    data_dir = path_to_files

    # Определяем все ID, для которых есть coarse файлы (1..100)
    coarse_ids = list(range(1, 101))
    # Fine файлы есть для подмножества: нужно автоматически определить из наличия файлов
    fine_ids = []
    for i in range(1, 101):
        fname = os.path.join(data_dir, f'pinndata_quick_id_{i:04d}_fine.mat')
        if os.path.exists(fname):
            fine_ids.append(i)
    print(f"Найдено fine файлов для ID: {fine_ids}")

    # Подготовка данных
    train_dataset, val_dataset, colloc_dataset, stats = prepare_datasets(
        data_dir, coarse_ids, fine_ids, n_neighbors=8
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    colloc_loader = DataLoader(colloc_dataset, batch_size=64, shuffle=True)

    # Модель
    model = SRPINN(
        n_spatial=3,
        n_shape_params=2,
        n_coarse_nodes=8,
        n_field_vars=8,
        hidden_dim=256,
        n_blocks=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Обучение
    train_srpinn(model, train_loader, val_dataset, colloc_loader, n_epochs=200, device=device)
