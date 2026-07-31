import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import DataLoader

def to_device(batch, device):
    """Переносит все тензоры в словаре на указанное устройство."""
    if batch is None:
        return None
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


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
                 n_blocks: int = 6,
                 fourier_mapping_size: int = 128,
                 fourier_scale: float = 5.0,
                 output_scale_init=1000.0):
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
        # self.output_scale = nn.Parameter(torch.ones(n_field_vars))
        # self.output_scale.data[6] = output_scale_init
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
        # out = out * self.output_scale
        return out

class StressPINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_voltage: float = 10.0, component_weights=None):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_voltage = lambda_voltage
        self.mse = nn.MSELoss(reduction='none')
        self.component_weights = component_weights

    def forward(self, model, batch, batch_pde=None):
        device = next(model.parameters()).device
        log = logging.getLogger()

        coords = batch['coords']
        shape = batch['shape_params']
        patch = batch['coarse_patch']
        pred_norm = model(coords, shape, patch)

        loss_data = torch.tensor(0.0, device=device)
        if 'target' in batch:
            diff = (pred_norm - batch['target']) ** 2
            if self.component_weights is not None:
                diff = diff * self.component_weights.to(device)
            loss_data = diff.mean()

        loss_voltage = torch.tensor(0.0, device=device)
        if 'is_top' in batch and 'is_bottom' in batch and 'id' in batch:
            fields_mean = batch['fields_mean']
            fields_std = batch['fields_std']
            phi_real = pred_norm[:, 6] * fields_std[..., 6] + fields_mean[..., 6]
            phi = phi_real
            ids = batch['id']
            is_top = batch['is_top']
            is_bottom = batch['is_bottom']
            unique_ids = torch.unique(ids)
            count = 0
            for uid in unique_ids:
                mask = (ids == uid)
                top_mask = mask & is_top
                bottom_mask = mask & is_bottom
                if top_mask.any() and bottom_mask.any():
                    phi_top = phi[top_mask].mean()
                    phi_bottom = phi[bottom_mask].mean()
                    V_pred = phi_top - phi_bottom
                    V_true = batch['voltage_true'][mask].mean()
                    rel_error = (V_pred.abs() - V_true) / (V_true + 1e-8)
                    loss_voltage += (rel_error ** 2).mean()
                    count += 1
            if count > 0:
                loss_voltage = loss_voltage / count
                log.debug(f"[DEBUG] voltage loss: {loss_voltage.item():.6f}")

        total_loss = loss_data + self.lambda_voltage * loss_voltage
        return total_loss, {
            'loss_data': loss_data.item(),
            'loss_voltage': loss_voltage.item(),
            'total_loss': total_loss.item()
        }


def compute_voltage_error(model, val_dataset, device, verbose=False, return_list=False, eps=1e-12):
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
            phi_pred = pred_fields_denorm[:, 6]  # + 1j * pred_fields_denorm[:, 7]
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


def train_srpinn(model, train_loader, val_dataset, colloc_loader, component_weights,
                 voltage_loader, n_epochs, device, lr=5e-4, pde_every=5, voltage_every=100):
    log = logging.getLogger()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = StressPINNLoss(lambda_data=1.0, lambda_voltage=10.0, component_weights=component_weights)
    best_voltage_error = float('inf')
    colloc_iter = iter(colloc_loader)
    step_counter = 0
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    voltage_iter = iter(voltage_loader)

    log.debug(f"[DEBUG] voltage_loader has {len(voltage_loader)} batches")
    if len(voltage_loader) > 0:
        sample_batch = next(iter(voltage_loader))
        for key in sample_batch:
            shape = sample_batch[key].shape if hasattr(sample_batch[key], 'shape') else None
            log.debug(f"[DEBUG] sample batch key '{key}': shape = {shape}")

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            compute_voltage = (step_counter % voltage_every == 0)
            if compute_voltage:
                log.debug(f"[DEBUG] step {step_counter} computing voltage loss")

            batch = to_device(batch, device)
            optimizer.zero_grad()
            loss, loss_dict = criterion(model, batch, None)

            if compute_voltage:
                try:
                    vbatch = next(voltage_iter)
                except StopIteration:
                    voltage_iter = iter(voltage_loader)
                    vbatch = next(voltage_iter)
                vbatch = to_device(vbatch, device)
                for key in vbatch:
                    if isinstance(vbatch[key], torch.Tensor) and vbatch[key].dim() > 0 and vbatch[key].size(0) == 1:
                        vbatch[key] = vbatch[key].squeeze(0)
                _, vloss_dict = criterion(model, vbatch, None)
                log.debug(f"[DEBUG] vloss_dict = {vloss_dict}")
                loss = loss + vloss_dict['loss_voltage']
                loss_dict['loss_voltage'] = vloss_dict['loss_voltage']
            else:
                loss_dict['loss_voltage'] = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict)

            step_counter += 1

            if step_counter % 100 == 0:
                log.debug(
                    f"step {step_counter}: loss_data={loss_dict['loss_data']:.6f}, loss_voltage={loss_dict['loss_voltage']:.6f}")

        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = to_device(vbatch, device)
                    _, loss_dict = criterion(model, vbatch, None)
                    val_losses.append(loss_dict)
            avg_val_loss = np.mean([d['total_loss'] for d in val_losses])
            voltage_error = compute_voltage_error(model, val_dataset, device, verbose=True)
            avg_train_loss = np.mean([d['total_loss'] for d in train_losses])
            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, voltage_rel_error={voltage_error:.4f}")
            log.info(
                f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, voltage_rel_error={voltage_error:.4f}")
            if voltage_error < best_voltage_error:
                best_voltage_error = voltage_error
                torch.save(model.state_dict(), 'best_srpinn_model_voltage.pth')
                log.info(f">>> Saved best model (voltage error {best_voltage_error:.4f}) <<<")