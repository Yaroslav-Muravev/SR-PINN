function export_PINN_data_quick(model, id, mesh_type, r_um, h_um, outdir)
% QUICK EXPORT (debug/prototype) - much lighter than full export_PINN_data
% Use this to get a fast, small dataset for testing/training iteration.

if nargin < 6, outdir = pwd; end

R = double(r_um) * 1e-6;
H = double(h_um) * 1e-6;
z0 = -H/2;

% Build grid (strictly inside the cylinder, avoid boundary points)
eps_r = 1e-4 * R;
eps_z = 1e-4 * H;

switch mesh_type
    case 'coarse'
        Nr = 16; Ntheta = 24; Nz = 8;
    case 'fine'
        Nr = 40; Ntheta = 48; Nz = 16;
    otherwise
        Nr = 24; Ntheta = 32; Nz = 10;
end

% Avoid r=0 and r=R boundary, and z top/bottom boundaries
r_vec = linspace(eps_r, R - eps_r, Nr);
theta_vec = linspace(0, 2*pi, Ntheta+1); 
theta_vec(end) = [];
z_vec = linspace(z0 + eps_z, z0 + H - eps_z, Nz);

[Rg, Tg, Zg] = ndgrid(r_vec, theta_vec, z_vec);
X = (Rg .* cos(Tg));
Y = (Rg .* sin(Tg));
Z = Zg;
coords = [X(:), Y(:), Z(:)];
Npt = size(coords,1);

% Minimal expressions - only what we need (change if names differ)
%exprs = { 'solid.u', 'solid.v', 'solid.w', 'V' };  % keep minimal

exprs = { 'u', 'v', 'w', 'V' };
% Use try_eval_points with small chunk (robust, but conservative)
try
    [Vmat, info_eval] = try_eval_points(model, exprs, coords, 'dset2', struct('chunk_size',500, 'verbose', true));
catch ME
    warning('Quick export failed in try_eval_points: %s', ME.message);
    % fallback: set NaNs and continue (we still produce meta file)
    Vmat = nan(Npt, numel(exprs));
    info_eval = [];
end

% Ensure shape
if isempty(Vmat) || size(Vmat,1) ~= Npt
    % try transpose
    if ~isempty(Vmat) && size(Vmat,2) == Npt
        Vmat = Vmat';
    else
        % if still wrong, create NaN matrix
        Vmat = nan(Npt, numel(exprs));
    end
end

% Map to arrays (safe reshape)
ux = reshape(Vmat(:,1), size(X));
uy = reshape(Vmat(:,2), size(X));
uz = reshape(Vmat(:,3), size(X));
phi = reshape(Vmat(:,4), size(X));

% ---------------- minimal physics sanity check ----------------
sanity = struct();

% 1) Проверка на конечность всех полей
allvals = [ux(:); uy(:); uz(:); phi(:)];
finite_mask = isfinite(allvals);
sanity.finite_frac = mean(finite_mask);
sanity.nan_count   = sum(~finite_mask);

% 2) Proxy voltage from two inner z-slices (not true boundaries)
k1 = max(1, 2);
k2 = min(size(phi,3), size(phi,3)-1);

phi_bot = phi(:,:,k1);
phi_top = phi(:,:,k2);

bot_abs = abs(phi_bot(:));
top_abs = abs(phi_top(:));
bot_abs = bot_abs(isfinite(bot_abs));
top_abs = top_abs(isfinite(top_abs));

if isempty(bot_abs)
    sanity.mean_abs_phi_bot = NaN;
else
    sanity.mean_abs_phi_bot = mean(bot_abs);
end

if isempty(top_abs)
    sanity.mean_abs_phi_top = NaN;
else
    sanity.mean_abs_phi_top = mean(top_abs);
end

% 3) Прокси-напряжение из экспортированного поля
phi_bot_mean = mean(phi_bot(isfinite(phi_bot)));
phi_top_mean = mean(phi_top(isfinite(phi_top)));
sanity.V_export_proxy = phi_top_mean - phi_bot_mean;

% 4) Сравнение с глобальным COMSOL-значением
try
    Vref = mphglobal(model, 'intop_top(V)/intop_top(1)', 'dataset', 'dset2');
    if iscell(Vref)
        Vref = Vref{1};
    end
    if isnumeric(Vref) && ~isempty(Vref)
        Vref = Vref(1);
        sanity.V_ref = Vref;
        sanity.relerr_V = abs(sanity.V_export_proxy - Vref) / max(1, abs(Vref));
    else
        sanity.V_ref = NaN;
        sanity.relerr_V = NaN;
    end
catch
    sanity.V_ref = NaN;
    sanity.relerr_V = NaN;
end

% 5) Ещё одна простая проверка: центр области
ic = round(size(phi,1)/2);
jc = round(size(phi,2)/2);
kc = round(size(phi,3)/2);
sanity.phi_center = phi(ic,jc,kc);
sanity.ux_center  = ux(ic,jc,kc);
sanity.uy_center  = uy(ic,jc,kc);
sanity.uz_center  = uz(ic,jc,kc);

% 6) Лог
fprintf('SANITY: finite_frac=%.6f, NaNs=%d, mean|phi| bot=%.3e, top=%.3e, Vproxy=%.3e, Vref=%.3e, relerr=%.3e\n', ...
    sanity.finite_frac, sanity.nan_count, ...
    sanity.mean_abs_phi_bot, sanity.mean_abs_phi_top, ...
    sanity.V_export_proxy, sanity.V_ref, sanity.relerr_V);

if sanity.finite_frac < 1
    warning('SANITY: exported fields contain NaN/Inf values.');
end

meta.sanity = sanity;
% ---------------- end minimal physics sanity check ----------------

% Save compact MAT only (no huge CSV, no collocation)
outmat = fullfile(outdir, sprintf('pinndata_quick_id_%04d_%s.mat', id, mesh_type));
meta.id = id; meta.r_m = R; meta.h_m = H; meta.mesh_type = mesh_type;
meta.r_vec = r_vec; meta.theta_vec = theta_vec; meta.z_vec = z_vec;
meta.exprs = exprs; meta.timestamp = datestr(now); meta.info_eval = info_eval;
save(outmat, 'X','Y','Z','ux','uy','uz','phi','meta', '-v7.3');

fprintf('Quick PINN data exported: %s  (N=%d). Collocation disabled.\n', outmat, Npt);
end