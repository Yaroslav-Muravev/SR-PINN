% run_batch.m  — обновлённая версия с robust mphglobal и диагностикой
function run_batch_main(params_file, idx_start, idx_end, mesh_type)
    arguments
        params_file (1,:) char
        idx_start (1,1) double
        idx_end (1,1) double
        mesh_type (1,:) char {mustBeMember(mesh_type, {'coarse','fine'})}
    end

    addpath(pwd);
    S = load(params_file,'params');
    params = S.params;

    out_csv = sprintf('results_%s_%04d_%04d.csv', mesh_type, idx_start, idx_end);
    logfn = sprintf('runlog_%s_%04d_%04d.txt', mesh_type, idx_start, idx_end);

    % Header with complex decomposition
    header = {'id','r_um','h_um','mesh_type','voltage','voltage_re','voltage_im','voltage_abs','voltage_angle','solver_time_s','total_time_s','timestamp'};
    if ~exist(out_csv,'file')
        fid = fopen(out_csv,'w');
        fprintf(fid, '%s,', header{1,1:end-1});
        fprintf(fid, '%s\n', header{1,end});
        fclose(fid);
    end

    % --- Helper formatters (use sprintf, not num2str with format) ---
    complex_fmt = @(z) sprintf('%.12g%+.12gi', real(z), imag(z));
    real_fmt    = @(x) sprintf('%.12g', x);

    for idx = idx_start:idx_end
        try
            row = params.list(idx,:);
        catch
            warning('Index %d out of bounds (total %d). Stopping.\n', idx, params.total);
            break;
        end

        id = row.id; r_um = row.r_um; h_um = row.h_um;
        tic_all = tic;

        % load/create model
        try
            mph_file = 'model_template.mph';
            if exist(mph_file,'file')
                ModelUtil.load(mph_file);
                model = ModelUtil.model();
            else
                addpath(pwd);
                model = feval('model');
            end
        catch ME
            fidlog = fopen(logfn,'a');
            if fidlog~=-1
                fprintf(fidlog, '%s ERROR loading/creating model id=%d: %s\n', datestr(now), id, ME.message);
                fprintf(fidlog, '%s\n', getReport(ME));
                fclose(fidlog);
            end
            rethrow(ME);
        end

        % set params
        try
            model.param.set('r_cylinder', sprintf('%g[um]', r_um));
            model.param.set('h_cylinder', sprintf('%g[um]', h_um));
        catch ME
            warning('Failed to set params id=%d: %s', id, ME.message);
        end

        % mesh change
        try
            if strcmp(mesh_type,'fine')
                model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmin','0.3E-6');
                model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmax','0.8E-6');
                model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hgrad',1.3);
            else
                model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmin','0.9E-6');
                model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmax','2.4E-6');
            end
            model.component('comp1').mesh('mesh1').run;
        catch ME
            warning('Mesh update failed id=%d: %s', id, ME.message);
        end

        % run solver
        try
            t0 = tic;
            model.study('std1').run;
            model.sol('sol1').runAll;
            try
                model.result.dataset.remove('dset2');
            catch
            end
            model.result.dataset.create('dset2','Solution');
            model.result.dataset('dset2').set('solution','sol1');
            solver_time = toc(t0);
        catch ME
            solver_time = NaN;
            fidlog = fopen(logfn,'a');
            if fidlog~=-1
                fprintf(fidlog, '%s SOLVER FAIL id=%d: %s\n', datestr(now), id, ME.message);
                fprintf(fidlog, '%s\n', getReport(ME));
                fclose(fidlog);
            end
            continue;
        end

        % ---------------- export data for PINN ----------------
        try
            outdir = pwd; % или указать явную папку
            export_PINN_data_quick(model, id, mesh_type, r_um, h_um, outdir);
        catch MEpin
            fidlog = fopen(logfn,'a');
            if fidlog~=-1
                fprintf(fidlog, '%s PINN EXPORT FAIL id=%d: %s\n', datestr(now), id, MEpin.message);
                fprintf(fidlog, '%s\n', getReport(MEpin));
                fclose(fidlog);
            end
        end
        % ---------------- end export ----------------

        % ---------------- Diagnostic export: full spectrum ----------------
%         try
%             nidn = ['gev_num_all_' num2str(id)];
%             tidn = ['tbl_num_all_' num2str(id)];
%             try model.result.numerical.remove(nidn); catch; end
%             try model.result.table.remove(tidn); catch; end
%             model.result.numerical.create(nidn,'EvalGlobal');
%             model.result.numerical(nidn).setIndex('expr','intop_top(V)',0);
%             model.result.table.create(tidn,'Table');
%             model.result.numerical(nidn).set('table', tidn);
%             model.result.numerical(nidn).setResult;
%             csvnum_all = sprintf('intop_top_V_full_id_%04d.csv', id);
%             model.result.table(tidn).save(csvnum_all);
%         catch MEcsvnall
%             csvnum_all = '';
%             fidlog = fopen(logfn,'a');
%             if fidlog~=-1, fprintf(fidlog,'%s DEBUG: failed intop_top(V) full table id=%d: %s\n', datestr(now), id, MEcsvnall.message); fclose(fidlog); end
%         end
% 
%         try
%             nidd = ['gev_den_all_' num2str(id)];
%             tidd = ['tbl_den_all_' num2str(id)];
%             try model.result.numerical.remove(nidd); catch; end
%             try model.result.table.remove(tidd); catch; end
%             model.result.numerical.create(nidd,'EvalGlobal');
%             model.result.numerical(nidd).setIndex('expr','intop_top(1)',0);
%             model.result.table.create(tidd,'Table');
%             model.result.numerical(nidd).set('table', tidd);
%             model.result.numerical(nidd).setResult;
%             csvden_all = sprintf('intop_top_1_full_id_%04d.csv', id);
%             model.result.table(tidd).save(csvden_all);
%         catch MEcsvdall
%             csvden_all = '';
%             fidlog = fopen(logfn,'a');
%             if fidlog~=-1, fprintf(fidlog,'%s DEBUG: failed intop_top(1) full table id=%d: %s\n', datestr(now), id, MEcsvdall.message); fclose(fidlog); end
%         end
% 
%         % parse both CSVs robustly and write full_spectrum file
%         if ~isempty(csvnum_all) && exist(csvnum_all,'file') && ~isempty(csvden_all) && exist(csvden_all,'file')
%             try
%                 % read as text and parse lines to support complex strings like "a+bi"
%                 fidn = fopen(csvnum_all,'r'); datn = {};
%                 while ~feof(fidn)
%                     ln = fgetl(fidn);
%                     if ~ischar(ln), break; end
%                     ln = strtrim(ln);
%                     if isempty(ln) || startsWith(ln,'%'), continue; end
%                     tokens = strsplit(ln,',');
%                     if numel(tokens)>=2
%                         datn(end+1,:) = {strtrim(tokens{1}), strtrim(tokens{2})}; %#ok<AGROW>
%                     end
%                 end
%                 fclose(fidn);
% 
%                 fidd = fopen(csvden_all,'r'); datd = {};
%                 while ~feof(fidd)
%                     ld = fgetl(fidd);
%                     if ~ischar(ld), break; end
%                     ld = strtrim(ld);
%                     if isempty(ld) || startsWith(ld,'%'), continue; end
%                     tokens = strsplit(ld,',');
%                     if numel(tokens)>=2
%                         datd(end+1,:) = {strtrim(tokens{1}), strtrim(tokens{2})}; %#ok<AGROW>
%                     end
%                 end
%                 fclose(fidd);
% 
%                 % ensure sizes match
%                 N = min(size(datn,1), size(datd,1));
%                 freq = zeros(N,1); numc = complex(zeros(N,1)); denc = complex(zeros(N,1));
%                 for k=1:N
%                     % parse freq (first column) and value (second)
%                     freq(k) = str2double(datn{k,1});
%                     % parse complex value robustly
%                     try
%                         vnum = str2num(datn{k,2}); %#ok<ST2NM>
%                         if isempty(vnum), vnum = 0; end
%                     catch
%                         vnum = 0;
%                     end
%                     try
%                         vden = str2num(datd{k,2}); %#ok<ST2NM>
%                         if isempty(vden), vden = 0; end
%                     catch
%                         vden = 0;
%                     end
%                     numc(k) = vnum;
%                     denc(k) = vden;
%                 end
% 
%                 % compute ratio safely
%                 ratio = complex(zeros(N,1));
%                 for k=1:N
%                     if denc(k) ~= 0
%                         ratio(k) = numc(k)/denc(k);
%                     else
%                         ratio(k) = NaN + 1i*NaN;
%                     end
%                 end
% 
%                 % assemble table and save
%                 T = table(freq, real(numc), imag(numc), real(denc), imag(denc), real(ratio), imag(ratio), abs(ratio), angle(ratio), ...
%                     'VariableNames', {'freq_Hz','num_re','num_im','den_re','den_im','ratio_re','ratio_im','ratio_abs','ratio_phase'});
%                 outfull = sprintf('full_spectrum_id_%04d.csv', id);
%                 writetable(T, outfull);
% 
%                 fidlog = fopen(logfn,'a');
%                 if fidlog~=-1
%                     fprintf(fidlog, '%s DEBUG: saved full spectrum id=%d -> %s (N=%d)\n', datestr(now), id, outfull, N);
%                     fclose(fidlog);
%                 end
%             catch MEfull
%                 fidlog = fopen(logfn,'a');
%                 if fidlog~=-1
%                     fprintf(fidlog, '%s DEBUG: failed to parse/save full spectrum id=%d: %s\n', datestr(now), id, MEfull.message);
%                     fclose(fidlog);
%                 end
%             end
%         end
        % ---------------- end diagnostic export ----------------

        % --- Try mphglobal first (most reliable) ---
        val_complex = NaN;
        val_str = '';
        tried_mphglobal = false;
        try
            tried_mphglobal = true;
            val_try = mphglobal(model,'intop_top(V)/intop_top(1)','dataset','dset2');

            % normalize cell -> numeric if possible
            if iscell(val_try) && numel(val_try)==1
                val_try = val_try{1};
            end

            % Case A: clean numeric scalar (possibly complex)
            if isnumeric(val_try) && isscalar(val_try) && isfinite(val_try)
                val_complex = double(val_try);
                val_str = complex_fmt(val_complex);

            % Case B: numeric vector (multiple freq points)
            elseif isnumeric(val_try) && isvector(val_try)
                % choose representative element: prefer closest to model frequency if possible, otherwise max magnitude
                chosen_idx = 1;
                chosen_reason = 'default';
                try
                    % attempt to find parameter f from model
                    try
                        fparam = model.param.evaluate('f');
                    catch
                        fparam = model.param.get('f');
                    end
                    if ischar(fparam)
                        mm = regexp(fparam, '([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)', 'match', 'once');
                        if ~isempty(mm), fnum = str2double(mm); else fnum = NaN; end
                    elseif isnumeric(fparam)
                        fnum = double(fparam);
                    else
                        fnum = NaN;
                    end
                catch
                    fnum = NaN;
                end

                if ~isnan(fnum)
                    % try to get freq vector from model via EvalGlobal intop_top(1) — quick attempt
                    try
                        tmp = mphglobal(model,'intop_top(1)','dataset','dset2');
                        if isnumeric(tmp) && size(tmp,2)>=1 && numel(tmp) == numel(val_try)
                            % no freq info here typically — fall back
                        end
                    catch
                    end
                end

                % fallback: choose element with largest magnitude
                [~, idx_max] = max(abs(val_try));
                chosen_idx = idx_max;
                chosen_reason = 'max-abs fallback';

                val_complex = double(val_try(chosen_idx));
                val_str = complex_fmt(val_complex);

                % Log selection
                fidlog = fopen(logfn,'a');
                if fidlog~=-1
                    fprintf(fidlog, '%s mphglobal returned numeric vector for id=%d (len=%d): choosing index %d (%s).\n', datestr(now), id, numel(val_try), chosen_idx, chosen_reason);
                    fclose(fidlog);
                end

            % Case C: numeric Nx2 table -> [freq, value]
            elseif isnumeric(val_try) && size(val_try,2)==2
                freqs = val_try(:,1);
                vals  = val_try(:,2);
                chosen_idx = 1; reason = 'default';
                % try to read f parameter
                try
                    fparam = model.param.get('f');
                    if ischar(fparam)
                        mm = regexp(fparam, '([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)', 'match', 'once');
                        if ~isempty(mm), fnum = str2double(mm); else fnum = NaN; end
                    elseif isnumeric(fparam)
                        fnum = double(fparam);
                    else
                        fnum = NaN;
                    end
                catch
                    fnum = NaN;
                end
                if ~isnan(fnum) && any(freqs>0)
                    if fnum < min(freqs)/2
                        fnum_try = fnum * 1e6;
                    else
                        fnum_try = fnum;
                    end
                    [~, chosen_idx] = min(abs(freqs - fnum_try));
                    reason = 'matched parameter f';
                else
                    [~, chosen_idx] = max(abs(vals));
                    reason = 'max-abs fallback';
                end
                val_complex = double(vals(chosen_idx));
                val_str = complex_fmt(val_complex);

                fidlog = fopen(logfn,'a');
                if fidlog~=-1
                    fprintf(fidlog, '%s mphglobal returned Nx2 for id=%d (N=%d). Chosen idx=%d (%s).\n', datestr(now), id, size(val_try,1), chosen_idx, reason);
                    fclose(fidlog);
                end

            else
                % Unknown return type: log and force fallback
                fidlog = fopen(logfn,'a');
                if fidlog~=-1
                    fprintf(fidlog, '%s mphglobal returned unsupported type for id=%d: class=%s size=%s. Falling back to table method.\n', ...
                        datestr(now), id, class(val_try), mat2str(size(val_try)));
                    fclose(fidlog);
                end
            end

        catch ME
            % mphglobal not available or threw — we'll fallback to EvalGlobal CSV
            fidlog = fopen(logfn,'a');
            if fidlog~=-1
                fprintf(fidlog, '%s mphglobal failed for id=%d: %s\n', datestr(now), id, ME.message);
                fclose(fidlog);
            end
        end

        % Additional diagnostic flow: if val_try was a vector, try to explicitly export numerator/denominator tables
%         if exist('val_try','var') && isnumeric(val_try) && isvector(val_try)
%             try
%                 fidlog = fopen(logfn,'a');
%                 if fidlog~=-1
%                     fprintf(fidlog, '%s DEBUG: mphglobal numeric vector for id=%d: %s\n', datestr(now), id, mat2str(double(val_try(:).')));
%                     fclose(fidlog);
%                 end
%             catch; end
% 
%             % attempt export intop_top(V) and intop_top(1) tables (already done above in full spectrum, but keep local names)
%             try
%                 nidn = ['gev_num_' num2str(id)];
%                 tidn = ['tbl_num_' num2str(id)];
%                 try model.result.numerical.remove(nidn); catch; end
%                 try model.result.table.remove(tidn); catch; end
%                 model.result.numerical.create(nidn,'EvalGlobal');
%                 model.result.numerical(nidn).setIndex('expr','intop_top(V)',0);
%                 model.result.table.create(tidn,'Table');
%                 model.result.numerical(nidn).set('table', tidn);
%                 model.result.numerical(nidn).setResult;
%                 csvnum = sprintf('intop_top_V_id_%04d.csv', id);
%                 model.result.table(tidn).save(csvnum);
%             catch; csvnum = ''; end
% 
%             try
%                 nidd = ['gev_den_' num2str(id)];
%                 tidd = ['tbl_den_' num2str(id)];
%                 try model.result.numerical.remove(nidd); catch; end
%                 try model.result.table.remove(tidd); catch; end
%                 model.result.numerical.create(nidd,'EvalGlobal');
%                 model.result.numerical(nidd).setIndex('expr','intop_top(1)',0);
%                 model.result.table.create(tidd,'Table');
%                 model.result.numerical(nidd).set('table', tidd);
%                 model.result.numerical(nidd).setResult;
%                 csvden = sprintf('intop_top_1_id_%04d.csv', id);
%                 model.result.table(tidd).save(csvden);
%             catch; csvden = ''; end
% 
%             % try to parse csvnum/csvden, choose nearest to parameter f or max ratio
%             if ~isempty(csvnum) && exist(csvnum,'file') && ~isempty(csvden) && exist(csvden,'file')
%                 try
%                     datn = readmatrix(csvnum);
%                     datd = readmatrix(csvden);
%                     if size(datn,2) >= 2 && size(datd,2) >= 2 && size(datn,1) == size(datd,1)
%                         freqs = datn(:,1);
%                         nums  = datn(:,2);
%                         dens  = datd(:,2);
%                         % get f param
%                         try
%                             fr = model.param.get('f');
%                             if ischar(fr)
%                                 mm = regexp(fr, '([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)', 'match', 'once');
%                                 if ~isempty(mm), fparam_val = str2double(mm); else fparam_val = NaN; end
%                             elseif isnumeric(fr)
%                                 fparam_val = double(fr);
%                             else
%                                 fparam_val = NaN;
%                             end
%                         catch
%                             fparam_val = NaN;
%                         end
%                         if ~isnan(fparam_val) && (fparam_val < min(freqs)/2)
%                             fparam_hz = fparam_val * 1e6;
%                         else
%                             fparam_hz = fparam_val;
%                         end
%                         if ~isnan(fparam_hz) && any(freqs>0)
%                             [~, chosen_idx] = min(abs(freqs - fparam_hz));
%                             reason = 'matched f';
%                         else
%                             mag = zeros(size(nums));
%                             nonzero = dens ~= 0;
%                             mag(nonzero) = abs(nums(nonzero)./dens(nonzero));
%                             if all(~nonzero)
%                                 [~, chosen_idx] = max(abs(nums));
%                                 reason = 'denominator all zero, chose max |num|';
%                             else
%                                 [~, chosen_idx] = max(mag);
%                                 reason = 'max |num/den|';
%                             end
%                         end
%                         chosen_num = nums(chosen_idx);
%                         chosen_den = dens(chosen_idx);
%                         if chosen_den ~= 0
%                             chosen_complex = chosen_num / chosen_den;
%                         else
%                             chosen_complex = NaN;
%                         end
%                         fidlog = fopen(logfn,'a');
%                         if fidlog~=-1
%                             fprintf(fidlog, '%s DEBUG: parsed CSVs for id=%d, chosen_idx=%d (freq=%.6g): reason=%s, num=%.12g, den=%.12g, ratio=%.12g\n', ...
%                                 datestr(now), id, chosen_idx, freqs(chosen_idx), reason, chosen_num, chosen_den, chosen_complex);
%                             fclose(fidlog);
%                         end
%                         if (~tried_mphglobal || ~has_good_val) && ~isnan(chosen_complex) && isfinite(chosen_complex)
%                             val_complex = chosen_complex;
%                             val_str = num2str(val_complex, '%.12g%+.12gi');
%                         end
%                     end
%                 catch
%                     % ignore parsing errors here
%                 end
%             end
%         end
        % ---------------- end diagnostics for vector returns ----------------

        % robust check if val_complex is numeric scalar finite
        has_good_val = isnumeric(val_complex) && isscalar(val_complex) && isfinite(val_complex);

        % If mphglobal didn't yield a scalar finite value, or val_str is empty, fallback to table export + parsing
        if ~has_good_val || isempty(val_str)
            out_table_csv = '';
            try
                nid = ['gev_batch_' num2str(id)];
                tid = ['tbl_batch_' num2str(id)];
                try model.result.numerical.remove(nid); catch; end
                try model.result.table.remove(tid); catch; end

                model.result.numerical.create(nid, 'EvalGlobal');
                model.result.numerical(nid).setIndex('expr', 'intop_top(V)/intop_top(1)', 0);
                model.result.table.create(tid,'Table');
                model.result.numerical(nid).set('table', tid);
                model.result.numerical(nid).setResult;
                out_table_csv = sprintf('voltage_id_%04d_%s.csv', id, mesh_type);
                model.result.table(tid).save(out_table_csv);
            catch MEeval
                % fallback attempt using gev1/tbl1
                try
                    model.result.numerical('gev1').setResult;
                    model.result.table('tbl1').save('tmp_tbl.csv');
                    out_table_csv = 'tmp_tbl.csv';
                catch
                    out_table_csv = '';
                    fidlog = fopen(logfn,'a');
                    if fidlog~=-1
                        fprintf(fidlog, '%s EVAL FAIL id=%d (no table produced): %s\n', datestr(now), id, MEeval.message);
                        fprintf(fidlog, '%s\n', getReport(MEeval));
                        fclose(fidlog);
                    end
                end
            end

            % parse the CSV: skip lines starting with %, find first numeric token parseable by str2num
            if ~isempty(out_table_csv) && exist(out_table_csv,'file')
                fid = fopen(out_table_csv,'r');
                if fid ~= -1
                    parsed = false;
                    while ~feof(fid) && ~parsed
                        line = fgetl(fid);
                        if ~ischar(line), break; end
                        line = strtrim(line);
                        if isempty(line), continue; end
                        if startsWith(line, '%'), continue; end
                        tokens = strsplit(line, ',');
                        for k2 = 1:length(tokens)
                            tok = strtrim(tokens{k2});
                            if isempty(tok), continue; end
                            if startsWith(tok, '"') && endsWith(tok, '"'); tok = tok(2:end-1); end
                            num = str2num(tok); %#ok<ST2NM>
                            if ~isempty(num) && isfinite(num)
                                val_complex = double(num);
                                val_str = complex_fmt(val_complex);
                                parsed = true;
                                break;
                            end
                            m = regexp(tok, '[-+]?\d+(\.\d+)?([eE][-+]?\d+)?([+\-]\d+(\.\d+)?([eE][-+]?\d+)?)?i?', 'match', 'once');
                            if ~isempty(m)
                                num2 = str2num(m); %#ok<ST2NM>
                                if ~isempty(num2) && isfinite(num2)
                                    val_complex = double(num2);
                                    val_str = complex_fmt(val_complex);
                                    parsed = true;
                                    break;
                                end
                            end
                        end
                    end
                    fclose(fid);
                end
            end
        end

        % If still NaN, save debug model for manual inspection (intop selection likely wrong)
%         if ~isnumeric(val_complex) || ~isfinite(val_complex) || isempty(val_str) || strcmp(val_str,'NaN')
%             try
%                 debugname = sprintf('debug_model_id_%04d.mph', id);
%                 model.save(debugname);
%                 fidlog = fopen(logfn,'a');
%                 if fidlog~=-1
%                     fprintf(fidlog, '%s WARNING id=%d: voltage parsing failed; saved debug model: %s\n', datestr(now), id, debugname);
%                     fclose(fidlog);
%                 end
%             catch
%                 fidlog = fopen(logfn,'a');
%                 if fidlog~=-1
%                     fprintf(fidlog, '%s WARNING id=%d: voltage parsing failed; could not save debug model\n', datestr(now), id);
%                     fclose(fidlog);
%                 end
%             end
%         end

        % compute decomposition for CSV output
        if isnumeric(val_complex) && isfinite(val_complex)
            v_re = real(val_complex);
            v_im = imag(val_complex);
            v_abs = abs(val_complex);
            v_angle = angle(val_complex);
            v_print = complex_fmt(val_complex);
        else
            v_re = NaN; v_im = NaN; v_abs = NaN; v_angle = NaN;
            v_print = 'NaN';
        end

        elapsed = toc(tic_all);

        % write row
        rowcell = {id, r_um, h_um, mesh_type, v_print, v_re, v_im, v_abs, v_angle, solver_time, elapsed, datestr(now)};
        Trow = cell2table(rowcell,'VariableNames',header);
        writetable(Trow, out_csv, 'WriteMode','append','WriteVariableNames',false);

        % optionally export mesh and model
        try
            model.component('comp1').mesh('mesh1').export.set('filename', sprintf('mesh_id_%04d_%s.txt', id, mesh_type));
            model.component('comp1').mesh('mesh1').export;
        catch; end
%         try
%             model.save(sprintf('model_id_%04d_%s.mph', id, mesh_type));
%         catch; end

        % log
        fidlog = fopen(logfn,'a');
        if fidlog~=-1
            fprintf(fidlog, '%s Finished id=%d (r=%g um, h=%g um) mesh=%s val=%s solver_s=%.3f total_s=%.3f mphglobal_used=%d\n', ...
                datestr(now), id, r_um, h_um, mesh_type, v_print, solver_time, elapsed, tried_mphglobal);
            fclose(fidlog);
        end
        fprintf('Finished id=%d (r=%g um, h=%g um) mesh=%s val=%s solver_s=%.3f total_s=%.3f mphglobal_used=%d\n', ...
            id, r_um, h_um, mesh_type, v_print, solver_time, elapsed, tried_mphglobal);
    end

    fprintf('All done for indices %d..%d\n', idx_start, idx_end);
end