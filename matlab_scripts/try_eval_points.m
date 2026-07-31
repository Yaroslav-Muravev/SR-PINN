% function [val, info] = try_eval_points(model, exprs, coords, dataset, opts)
% % TRY_EVAL_POINTS  Robust evaluation of COMSOL expressions at many points.
% % Returns val as N x nExpr (NaN where failed).
% %
% % USAGE:
% %   [val, info] = try_eval_points(model, exprs, coords, dataset, opts)
% % - coords: N x 3
% % - exprs: cell array of strings or char
% % - dataset: dataset name (e.g. 'dset2') or '' if none
% % - opts.chunk_size (default 1000), opts.verbose (default false)
% %
% % This version is defensive: small chunk size, clear try/catch, saves debug file
% % when lots of NaNs are found.
% 
% if nargin < 4, dataset = ''; end
% if nargin < 5, opts = struct(); end
% if ~isfield(opts,'chunk_size'), opts.chunk_size = 500; end
% if ~isfield(opts,'verbose'), opts.verbose = false; end
% if ~isfield(opts,'debug_on_nan'), opts.debug_on_nan = true; end
% 
% % Validate inputs
% if size(coords,2) ~= 3
%     error('coords must be Nx3 (x y z)');
% end
% if ischar(exprs), exprs = {exprs}; end
% 
% N = size(coords,1);
% nExpr = numel(exprs);
% val = nan(N, nExpr);
% info = struct();
% attempts = {};
% 
% % iterate in chunks to avoid memory issues
% chunk = opts.chunk_size;
% idx = 1;
% while idx <= N
%     j = min(N, idx + chunk - 1);
%     pts = coords(idx:j, :);  % M x 3
%     M = size(pts,1);
%     chunk_msgs = {};
%     success = false;
% 
%     % 1) Try mphinterp with coords as 3xM
%     try
%         if ~isempty(dataset)
%             res = mphinterp(model, exprs, 'coord', pts', 'dataset', dataset);
%         else
%             res = mphinterp(model, exprs, 'coord', pts');
%         end
%         Mtx = normalize_result(res, M, nExpr);
%         if ~isempty(Mtx)
%             val(idx:j, :) = Mtx;
%             success = true;
%             chunk_msgs{end+1} = 'mphinterp(coord 3xM) OK';
%         else
%             chunk_msgs{end+1} = 'mphinterp(coord 3xM) returned unexpected shape';
%         end
%     catch ME
%         chunk_msgs{end+1} = ['mphinterp(coord 3xM) failed: ' ME.message];
%     end
% 
%     % 2) Try mphinterp with Nx3
%     if ~success
%         try
%             res = mphinterp(model, exprs, pts);
%             Mtx = normalize_result(res, M, nExpr);
%             if ~isempty(Mtx)
%                 val(idx:j, :) = Mtx;
%                 success = true;
%                 chunk_msgs{end+1} = 'mphinterp(pts Nx3) OK';
%             else
%                 chunk_msgs{end+1} = 'mphinterp(pts Nx3) returned unexpected shape';
%             end
%         catch ME
%             chunk_msgs{end+1} = ['mphinterp(pts) failed: ' ME.message];
%         end
%     end
% 
%     % 3) Try mpheval with 'coord' 3xM
% %     if ~success
% %         try
% %             if ~isempty(dataset)
% %                 res = mpheval(model, exprs, 'coord', pts', 'dataset', dataset);
% %             else
% %                 res = mpheval(model, exprs, 'coord', pts');
% %             end
% %             Mtx = normalize_result(res, M, nExpr);
% %             if ~isempty(Mtx)
% %                 val(idx:j, :) = Mtx;
% %                 success = true;
% %                 chunk_msgs{end+1} = 'mpheval(coord) OK';
% %             else
% %                 chunk_msgs{end+1} = 'mpheval(coord) returned unexpected shape';
% %             end
% %         catch ME
% %             chunk_msgs{end+1} = ['mpheval(coord) failed: ' ME.message];
% %         end
% %     end
% 
%     % 4) Fallback per-point (slow)
% %     if ~success
% %         try
% %             tmp = nan(M, nExpr);
% %             for p = 1:M
% %                 pt = pts(p, :);
% %                 got = false;
% %                 % try mphinterp single
% %                 try
% %                     r = mphinterp(model, exprs, 'coord', pt');
% %                     rM = normalize_result(r, 1, nExpr);
% %                     if ~isempty(rM), tmp(p,:) = rM; got = true; end
% %                 catch
% %                 end
% %                 if ~got
% %                     try
% %                         r = mphinterp(model, exprs, pt);
% %                         rM = normalize_result(r, 1, nExpr);
% %                         if ~isempty(rM), tmp(p,:) = rM; got = true; end
% %                     catch
% %                     end
% %                 end
% %                 if ~got
% %                     try
% %                         r = mpheval(model, exprs, 'coord', pt');
% %                         rM = normalize_result(r, 1, nExpr);
% %                         if ~isempty(rM), tmp(p,:) = rM; got = true; end
% %                     catch
% %                     end
% %                 end
% %                 % if still not got -> leave NaNs
% %             end
% %             val(idx:j, :) = tmp;
% %             chunk_msgs{end+1} = 'per-point fallback used';
% %             success = true;
% %         catch ME
% %             chunk_msgs{end+1} = ['per-point fallback failed: ' ME.message];
% %         end
% %     end
% 
%     attempts{end+1} = sprintf('chunk %d..%d: %s', idx, j, strjoin(chunk_msgs, ' || '));
%     if opts.verbose
%         fprintf('%s\n', attempts{end});
%     end
%     idx = j + 1;
% end
% 
% info.attempts = attempts;
% info.summary = sprintf('Evaluated %d points in chunks of %d', N, chunk);
% 
% % If many NaNs, save small debug file (first problematic points)
% nan_idx = find(any(isnan(val),2));
% if ~isempty(nan_idx) && opts.debug_on_nan
%     sel = nan_idx(1:min(50, numel(nan_idx)));
%     dbg.coords = coords(sel,:);
%     dbg.exprs = exprs;
%     dbg.val_partial = val(sel,:);
%     debugfn = fullfile(pwd, sprintf('pinndata_eval_debug_%s.mat', datestr(now,'yyyymmdd_HHMMSS')));
%     try
%         save(debugfn, 'dbg', 'info', '-v7.3');
%         info.debug_file = debugfn;
%         if opts.verbose
%             fprintf('Saved debug file: %s\n', debugfn);
%         end
%     catch
%         info.debug_save_error = 'Could not save debug file';
%     end
% end
% 
% end
% 
% % ---------------- helper ----------------
% function Mtx = normalize_result(res, M, nExpr)
% % Normalize different COMSOL return formats to M x nExpr
% Mtx = [];
% if isempty(res), return; end
% % cell
% if iscell(res)
%     try
%         tmp = cellfun(@(c) double(c(:)), res, 'UniformOutput', false);
%         if numel(tmp) >= nExpr && all(cellfun(@(v) numel(v) >= M, tmp(1:nExpr)))
%             Mtx = nan(M, nExpr);
%             for k = 1:nExpr
%                 Mtx(:,k) = tmp{k}(1:M);
%             end
%             return;
%         end
%         flat = cell2mat(res(:));
%         if numel(flat) == M * nExpr
%             Mtx = reshape(double(flat), M, nExpr);
%             return;
%         end
%     catch
%         Mtx = [];
%         return;
%     end
% end
% 
% % numeric
% if isnumeric(res)
%     [r,c] = size(res);
%     if r == M && c >= nExpr
%         Mtx = double(res(:,1:nExpr)); return;
%     elseif c == M && r >= nExpr
%         Mtx = double(res(1:nExpr,1:M))'; return;
%     elseif numel(res) == M * nExpr
%         Mtx = reshape(double(res(:)), M, nExpr); return;
%     else
%         Mtx = [];
%     end
%     return;
% end
% 
% % struct output (mpheval)
% % if isstruct(res)
% %     try
% %         if numel(res) >= nExpr
% %             Mtx = nan(M, nExpr);
% %             for k=1:nExpr
% %                 fld = res(k);
% %                 if isfield(fld,'d1'), v = double(fld.d1(:));
% %                 elseif isfield(fld,'d'), v = double(fld.d(:));
% %                 else
% %                     % pick first numeric field
% %                     fns = fieldnames(fld);
% %                     v = [];
% %                     for f = 1:numel(fns)
% %                         if isnumeric(fld.(fns{f}))
% %                             v = double(fld.(fns{f})(:)); break;
% %                         end
% %                     end
% %                     if isempty(v), v = nan(M,1); end
% %                 end
% %                 if numel(v) >= M
% %                     Mtx(:,k) = v(1:M);
% %                 else
% %                     Mtx(:,k) = nan(M,1);
% %                 end
% %             end
% %             return;
% %         end
% %     catch
% %         Mtx = [];
% %         return;
% %     end
% % end
% 
% end


function [val, info] = try_eval_points(model, exprs, coords, dataset, opts)
%TRY_EVAL_POINTS  Minimal debug-friendly evaluation of COMSOL expressions.
%
% This version is intentionally simple:
%   - uses ONLY mphinterp
%   - evaluates ONE expression at a time
%   - accepts outputs shaped as:
%       * Mx1 or 1xM  -> direct scalar values
%       * 3xM         -> takes first row (debug heuristic)
%       * Mx3         -> takes first column (debug heuristic)
%   - saves a debug .mat file if NaNs remain
%
% INPUTS:
%   model   - COMSOL model object
%   exprs   - cell array of expressions, e.g. {'u','v','w','V'} or {'solid.u','solid.v','solid.w','V'}
%   coords  - Nx3 array of query points [x y z]
%   dataset - dataset name, e.g. 'dset2' or '' to omit
%   opts    - optional struct:
%               opts.chunk_size   (default 500)
%               opts.verbose      (default false)
%               opts.debug_on_nan (default true)
%
% OUTPUTS:
%   val  - NxNexpr numeric array, NaN where evaluation failed
%   info - struct with debug info

if ~isfield(opts, 'sanity_check') || isempty(opts.sanity_check)
    opts.sanity_check = true;
end
if ~isfield(opts, 'sanity_tol_rel') || isempty(opts.sanity_tol_rel)
    opts.sanity_tol_rel = 1e-6;
end
if ~isfield(opts, 'sanity_nanchors') || isempty(opts.sanity_nanchors)
    opts.sanity_nanchors = 3;
end

if nargin < 4 || isempty(dataset)
    dataset = '';
end
if nargin < 5
    opts = struct();
end
if ~isfield(opts, 'chunk_size') || isempty(opts.chunk_size)
    opts.chunk_size = 500;
end
if ~isfield(opts, 'verbose') || isempty(opts.verbose)
    opts.verbose = false;
end
if ~isfield(opts, 'debug_on_nan') || isempty(opts.debug_on_nan)
    opts.debug_on_nan = true;
end

% ---------------- validation ----------------
if ~isnumeric(coords) || size(coords,2) ~= 3
    error('coords must be an Nx3 numeric array.');
end
if ischar(exprs) || isstring(exprs)
    exprs = cellstr(exprs);
end
if ~iscell(exprs) || isempty(exprs)
    error('exprs must be a non-empty cell array of expressions.');
end

N = size(coords,1);
nExpr = numel(exprs);
val = nan(N, nExpr);

info = struct();
info.attempts = {};
info.summary = '';
info.nan_rows = [];
info.debug_file = '';
info.special_shapes = {};

chunk = opts.chunk_size;
idx = 1;

while idx <= N
    j = min(N, idx + chunk - 1);
    pts = coords(idx:j, :);   % Mx3
    M = size(pts,1);

    tmp = nan(M, nExpr);
    chunk_msgs = {};

    for e = 1:nExpr
        expr = exprs{e};
        try
            if ~isempty(dataset)
                r = mphinterp(model, expr, 'coord', pts', 'dataset', dataset);
            else
                r = mphinterp(model, expr, 'coord', pts');
            end

            % --- minimal sanity check on the first chunk / first expression only ---
            if opts.sanity_check && idx == 1 && e == 1
                sanity = run_min_sanity_check(model, expr, pts, dataset, r, opts);
                info.sanity = sanity;
            
                if opts.verbose
                    fprintf('SANITY CHECK expr=%s: rel_err=%.3e, raw_shape=%s\n', ...
                        expr, sanity.rel_err, mat2str(sanity.raw_shape));
                end
            
                if isfield(sanity, 'rel_err') && sanity.rel_err > opts.sanity_tol_rel
                    warning('Sanity check failed for expr=%s: rel_err=%.3e > %.3e', ...
                        expr, sanity.rel_err, opts.sanity_tol_rel);
                end
            end

            if opts.verbose
                fprintf('chunk %d..%d, expr=%s: class=%s size=%s\n', ...
                    idx, j, expr, class(r), mat2str(size(r)));
            end

            [rvec, note] = normalize_mphinterp_result_debug(r, M);

            if ~isempty(note)
                info.special_shapes{end+1} = sprintf('chunk %d..%d expr=%s: %s', idx, j, expr, note); %#ok<AGROW>
                if opts.verbose
                    fprintf('  NOTE: %s\n', note);
                end
            end

            if isempty(rvec)
                chunk_msgs{end+1} = sprintf('expr=%s returned unexpected shape', expr); %#ok<AGROW>
                continue;
            end

            tmp(:, e) = rvec;
            chunk_msgs{end+1} = sprintf('expr=%s OK', expr); %#ok<AGROW>

        catch ME
            chunk_msgs{end+1} = sprintf('expr=%s failed: %s', expr, ME.message); %#ok<AGROW>
            if opts.verbose
                fprintf('chunk %d..%d, expr=%s FAILED: %s\n', idx, j, expr, ME.message);
            end
        end
    end

    val(idx:j, :) = tmp;
    info.attempts{end+1} = sprintf('chunk %d..%d: %s', idx, j, strjoin(chunk_msgs, ' || ')); %#ok<AGROW>

    if opts.verbose
        fprintf('%s\n', info.attempts{end});
    end

    idx = j + 1;
end

info.summary = sprintf('Evaluated %d points in chunks of %d for %d expressions.', N, chunk, nExpr);

% ---------------- debug on NaN ----------------
nan_rows = find(any(isnan(val), 2));
info.nan_rows = nan_rows;

if ~isempty(nan_rows) && opts.debug_on_nan
    sel = nan_rows(1:min(50, numel(nan_rows)));
    dbg = struct();
    dbg.coords = coords(sel,:);
    dbg.exprs = exprs;
    dbg.val_partial = val(sel,:);
    dbg.nan_rows = sel;
    dbg.summary = info.summary;
    dbg.special_shapes = info.special_shapes;

    debugfn = fullfile(pwd, sprintf('pinndata_eval_debug_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
    try
        save(debugfn, 'dbg', 'info', '-v7.3');
        info.debug_file = debugfn;
        if opts.verbose
            fprintf('Saved debug file: %s\n', debugfn);
        end
    catch ME
        info.debug_file = '';
        info.debug_save_error = ME.message;
        if opts.verbose
            fprintf('Could not save debug file: %s\n', ME.message);
        end
    end
end

end

% ========================================================================
function [rvec, note] = normalize_mphinterp_result_debug(r, M)
%NORMALIZE_MPHINTERP_RESULT_DEBUG Convert mphinterp output to Mx1 vector.
%
% Debug heuristic:
%   - scalar / 1xM / Mx1 -> use directly
%   - 3xM -> take first row
%   - Mx3 -> take first column
%
% Returns:
%   rvec - Mx1 double vector, or []
%   note - short text describing special handling

rvec = [];
note = '';

if isempty(r)
    note = 'empty result';
    return;
end

% unwrap cell
if iscell(r)
    if numel(r) == 1
        r = r{1};
    else
        try
            rr = cellfun(@(x) double(x(:)), r(:), 'UniformOutput', false);
            flat = vertcat(rr{:});
            if numel(flat) == M
                rvec = flat(:);
                note = 'cell array collapsed to vector';
                return;
            end
        catch
            note = 'cell result could not be collapsed';
            return;
        end
    end
end

if ~isnumeric(r)
    note = sprintf('unsupported class %s', class(r));
    return;
end

r = double(r);
[sr, sc] = size(r);

% scalar
if isscalar(r)
    rvec = repmat(r, M, 1);
    note = 'scalar replicated to all points';
    return;
end

% exact vector forms
if sr == M && sc == 1
    rvec = r(:,1);
    return;
elseif sr == 1 && sc == M
    rvec = r(1,:).';
    return;
end

% debug heuristic for COMSOL weird shapes
if sr == 3 && sc == M
    rvec = r(1,:).';
    note = 'shape 3xM -> took first row';
    return;
elseif sr == M && sc == 3
    rvec = r(:,1);
    note = 'shape Mx3 -> took first column';
    return;
end

% if it is a plain vector but wrong length, try to salvage
if isvector(r)
    rv = r(:);
    if numel(rv) >= M
        rvec = rv(1:M);
        note = sprintf('vector length %d -> truncated to M', numel(rv));
        return;
    elseif numel(rv) == 1
        rvec = repmat(rv, M, 1);
        note = 'single vector element replicated to all points';
        return;
    end
end

% last resort: if total numel matches M, reshape
if numel(r) == M
    rvec = reshape(r, M, 1);
    note = 'reshaped to Mx1';
    return;
end

note = sprintf('unexpected size %dx%d', sr, sc);
end

function sanity = run_min_sanity_check(model, expr, pts, dataset, r, opts)
%RUN_MIN_SANITY_CHECK
% Minimal debug check:
%   - takes a few anchor points from the first chunk
%   - compares the batch-based extracted value to a direct single-point eval
%   - reports relative error
%
% Important:
%   This checks internal consistency of the parser / extraction logic,
%   not the physical correctness of the COMSOL model.

M = size(pts,1);
sanity = struct();
sanity.expr = expr;
sanity.raw_shape = size(r);
sanity.anchor_idx = unique([1, max(1, round(M/2)), M]);
sanity.anchor_idx = sanity.anchor_idx(1:min(numel(sanity.anchor_idx), opts.sanity_nanchors));

nA = numel(sanity.anchor_idx);
sanity.batch_vals = nan(nA,1);
sanity.single_vals = nan(nA,1);
sanity.single_raw = cell(nA,1);

% batch-based value: use the same rule as main parser
batch_vec = normalize_mphinterp_result_debug(r, M);
if isempty(batch_vec)
    sanity.rel_err = inf;
    sanity.note = 'batch normalization failed';
    return;
end

for a = 1:nA
    k = sanity.anchor_idx(a);

    % value from batch-parsed result
    if k <= numel(batch_vec)
        sanity.batch_vals(a) = batch_vec(k);
    end

    % direct single-point call
    pt = pts(k, :)';
    if ~isempty(dataset)
        rs = mphinterp(model, expr, 'coord', pt, 'dataset', dataset);
    else
        rs = mphinterp(model, expr, 'coord', pt);
    end
    sanity.single_raw{a} = rs;

    rs_vec = normalize_mphinterp_result_debug(rs, 1);
    if ~isempty(rs_vec)
        sanity.single_vals(a) = rs_vec(1);
    end
end

den = max(1, abs(sanity.single_vals));
diffv = abs(sanity.batch_vals - sanity.single_vals) ./ den;
sanity.rel_err = max(diffv(~isnan(diffv)));

if isempty(sanity.rel_err) || isnan(sanity.rel_err)
    sanity.rel_err = inf;
end

sanity.note = sprintf('anchors=%s', mat2str(sanity.anchor_idx));
end