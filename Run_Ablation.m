% =========================================================================
%  Run_Ablation.m
%  Ablation Study — Yakka-lite and Garo mechanism isolation
%
%  Variants:
%    LOA-Full     — both Yakka-lite and Garo enabled   (baseline)
%    LOA-NoYakka  — Yakka-lite disabled, Garo enabled
%    LOA-NoGaro   — Yakka-lite enabled,  Garo disabled
%    LOA-None     — both Yakka-lite and Garo disabled
%
%  Experiment groups:
%    G6  Local-Trap Init  D=30,50  T1/T2/T3   (6 sub-exp)  PRIMARY
%    G2  Shifted Schwefel D=10,30,50           (3 sub-exp)  CORE CLAIM
%    G5  Distant Init     D=30,50  I3 & I5     (4 sub-exp)  RECOVERY
%    Total: 13 sub-exp x 4 variants x 30 runs = 1,560 runs
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Ablation';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

diary(fullfile(RESULTS_DIR, 'Ablation_log.txt'));
diary on;

%% ── 1. SETTINGS ─────────────────────────────────────────────────────────
VARIANTS = {
    'LOA-Full',    false, false;
    'LOA-NoYakka', true,  false;
    'LOA-NoGaro',  false, true;
    'LOA-None',    true,  true;
};
N_VAR          = size(VARIANTS, 1);
N_RUNS         = 30;
N_POP          = 30;
N_GRADES       = 5;
N_g            = max(4, floor(N_POP / N_GRADES));   % = 6
SUCCESS_THRESH = 1.0;
SHIFT_SEED     = 9999;

fprintf('=========================================================\n');
fprintf('  LOA Ablation Study\n');
fprintf('  Variants : LOA-Full | LOA-NoYakka | LOA-NoGaro | LOA-None\n');
fprintf('  Runs     : %d per (variant, sub-exp)\n', N_RUNS);
fprintf('  N_g      : %d per grade (%d total)\n', N_g, N_g*N_GRADES);
fprintf('=========================================================\n\n');

%% ── 2. STORAGE ───────────────────────────────────────────────────────────
all_results = struct('label',{},'group',{},'D',{},...
                     'best_f_mat',{},'fe_to_tgt',{},...
                     'stat_mean',{},'stat_std',{},...
                     'stat_sr',{},'stat_fet',{});
n_subexp = 0;

%% ── 3. G2 — SHIFTED SCHWEFEL ─────────────────────────────────────────────
fprintf('=========================================================\n');
fprintf('  G2 — Shifted Schwefel (core claim)\n');
fprintf('=========================================================\n\n');

for D = [10, 30, 50]
    Max_FE = D * 10000;
    lb     = -500 * ones(1,D);
    ub     =  500 * ones(1,D);

    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);

    label = sprintf('Sft_D%d', D);
    fprintf('  Sub-exp: %s\n', label);

    best_f_mat = zeros(N_VAR, N_RUNS);
    fe_to_tgt  = nan(N_VAR, N_RUNS);

    for vi = 1:N_VAR
        vname     = VARIANTS{vi,1};
        dis_yakka = VARIANTS{vi,2};
        dis_garo  = VARIANTS{vi,3};
        fprintf('    %-14s ', vname);
        t0 = tic;
        for r = 1:N_RUNS
            rng(r);
            opts              = struct();
            opts.seed         = [];
            opts.disable_yakka = dis_yakka;
            opts.disable_garo  = dis_garo;
            obj_loa = @(x) deal(schwefel_shifted_fn(x, shift_vec, D), 0);
            [~, fbest, conv_raw, ~, ~] = LOA_ablation(obj_loa, lb, ub, N_g, Max_FE, opts);
            best_f_mat(vi,r) = fbest;
            conv = resample_conv(conv_raw, Max_FE, N_g, N_GRADES);
            hit = find(conv < SUCCESS_THRESH, 1, 'first');
            if ~isempty(hit)
                fe_to_tgt(vi,r) = hit * (N_g * N_GRADES);
            end
        end
        vals = best_f_mat(vi,:);
        sr   = sum(vals < SUCCESS_THRESH) / N_RUNS * 100;
        fprintf('mean=%10.3e  SR=%5.1f%%  [%.1fs]\n', mean(vals), sr, toc(t0));
    end

    n_subexp = n_subexp + 1;
    all_results(n_subexp) = build_entry(label,'G2',D,best_f_mat,fe_to_tgt,...
                                         N_VAR,N_RUNS,SUCCESS_THRESH);
    save_subexp(RESULTS_DIR, label, all_results(n_subexp), VARIANTS);
    fprintf('\n');
end

%% ── 4. G5 — DISTANT INIT (I3=50%% and I5=Extreme) ────────────────────────
fprintf('=========================================================\n');
fprintf('  G5 — Distant Init (I3 and I5)\n');
fprintf('=========================================================\n\n');

HALF_WIDTH  = 50;
CASE_LABELS = {'I3','I5'};
CASE_FRACS  = [0.50, 1.00];

for D = [30, 50]
    Max_FE = D * 10000;
    lb     = -500 * ones(1,D);
    ub     =  500 * ones(1,D);

    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);
    x_opt     = shift_vec;
    x_far     = zeros(1,D);
    x_far(x_opt >= 0) = -500;
    x_far(x_opt <  0) =  500;

    for ci = 1:numel(CASE_LABELS)
        frac    = CASE_FRACS(ci);
        center  = x_opt + frac * (x_far - x_opt);
        lb_init = max(-500, center - HALF_WIDTH);
        ub_init = min( 500, center + HALF_WIDTH);

        label = sprintf('DI_D%d_%s', D, CASE_LABELS{ci});
        fprintf('  Sub-exp: %s\n', label);

        best_f_mat = zeros(N_VAR, N_RUNS);
        fe_to_tgt  = nan(N_VAR, N_RUNS);

        for vi = 1:N_VAR
            vname     = VARIANTS{vi,1};
            dis_yakka = VARIANTS{vi,2};
            dis_garo  = VARIANTS{vi,3};
            fprintf('    %-14s ', vname);
            t0 = tic;
            for r = 1:N_RUNS
                rng(r);
                opts              = struct();
                opts.seed         = [];
                opts.disable_yakka = dis_yakka;
                opts.disable_garo  = dis_garo;
                opts.X0 = lb_init + rand(N_g*N_GRADES, D) .* (ub_init - lb_init);
                obj_loa = @(x) deal(schwefel_shifted_fn(x, shift_vec, D), 0);
                [~, fbest, conv_raw, ~, ~] = LOA_ablation(obj_loa, lb, ub, N_g, Max_FE, opts);
                best_f_mat(vi,r) = fbest;
                conv = resample_conv(conv_raw, Max_FE, N_g, N_GRADES);
                hit = find(conv < SUCCESS_THRESH, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(vi,r) = hit * (N_g * N_GRADES);
                end
            end
            vals = best_f_mat(vi,:);
            sr   = sum(vals < SUCCESS_THRESH) / N_RUNS * 100;
            fprintf('mean=%10.3e  SR=%5.1f%%  [%.1fs]\n', mean(vals), sr, toc(t0));
        end

        n_subexp = n_subexp + 1;
        all_results(n_subexp) = build_entry(label,'G5',D,best_f_mat,fe_to_tgt,...
                                             N_VAR,N_RUNS,SUCCESS_THRESH);
        save_subexp(RESULTS_DIR, label, all_results(n_subexp), VARIANTS);
        fprintf('\n');
    end
end

%% ── 5. G6 — LOCAL TRAP (PRIMARY) ─────────────────────────────────────────
fprintf('=========================================================\n');
fprintf('  G6 — Local-Trap Init (PRIMARY ablation test)\n');
fprintf('=========================================================\n\n');

LOCAL_MIN_DIST  = 302.5232;
TRAP_LABELS     = {'T1','T2','T3'};
TRAP_HALFWIDTHS = [25, 50, 150];

for D = [30, 50]
    Max_FE = D * 10000;
    lb     = -500 * ones(1,D);
    ub     =  500 * ones(1,D);

    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);
    x_trap    = max(-500, min(500, shift_vec - LOCAL_MIN_DIST));

    for ti = 1:numel(TRAP_LABELS)
        half_w  = TRAP_HALFWIDTHS(ti);
        lb_init = max(-500, x_trap - half_w);
        ub_init = min( 500, x_trap + half_w);

        label = sprintf('LT_D%d_%s', D, TRAP_LABELS{ti});
        fprintf('  Sub-exp: %s\n', label);

        best_f_mat = zeros(N_VAR, N_RUNS);
        fe_to_tgt  = nan(N_VAR, N_RUNS);

        for vi = 1:N_VAR
            vname     = VARIANTS{vi,1};
            dis_yakka = VARIANTS{vi,2};
            dis_garo  = VARIANTS{vi,3};
            fprintf('    %-14s ', vname);
            t0 = tic;
            for r = 1:N_RUNS
                rng(r);
                opts              = struct();
                opts.seed         = [];
                opts.disable_yakka = dis_yakka;
                opts.disable_garo  = dis_garo;
                opts.X0 = lb_init + rand(N_g*N_GRADES, D) .* (ub_init - lb_init);
                obj_loa = @(x) deal(schwefel_shifted_fn(x, shift_vec, D), 0);
                [~, fbest, conv_raw, ~, ~] = LOA_ablation(obj_loa, lb, ub, N_g, Max_FE, opts);
                best_f_mat(vi,r) = fbest;
                conv = resample_conv(conv_raw, Max_FE, N_g, N_GRADES);
                hit = find(conv < SUCCESS_THRESH, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(vi,r) = hit * (N_g * N_GRADES);
                end
            end
            vals = best_f_mat(vi,:);
            sr   = sum(vals < SUCCESS_THRESH) / N_RUNS * 100;
            fprintf('mean=%10.3e  SR=%5.1f%%  [%.1fs]\n', mean(vals), sr, toc(t0));
        end

        n_subexp = n_subexp + 1;
        all_results(n_subexp) = build_entry(label,'G6',D,best_f_mat,fe_to_tgt,...
                                             N_VAR,N_RUNS,SUCCESS_THRESH);
        save_subexp(RESULTS_DIR, label, all_results(n_subexp), VARIANTS);
        fprintf('\n');
    end
end

%% ── 6. PRINT SUMMARY ─────────────────────────────────────────────────────
fprintf('=========================================================\n');
fprintf('  ABLATION SUMMARY — Success Rate (%%)\n');
fprintf('  %-22s', 'Sub-exp');
for vi = 1:N_VAR, fprintf('  %-13s', VARIANTS{vi,1}); end
fprintf('\n  %s\n', repmat('-',1,78));

for se = 1:n_subexp
    e = all_results(se);
    fprintf('  %-22s', e.label);
    for vi = 1:N_VAR
        fprintf('  %11.1f%%', e.stat_sr(vi)*100);
    end
    fprintf('\n');
end
fprintf('\n');

%% ── 7. SAVE ──────────────────────────────────────────────────────────────
save(fullfile(RESULTS_DIR,'Ablation_summary.mat'), ...
     'all_results','VARIANTS','N_VAR','N_RUNS','SUCCESS_THRESH');
fprintf('Saved: Ablation_summary.mat\n');

fid = fopen(fullfile(RESULTS_DIR,'Ablation_summary.csv'),'w');
fprintf(fid,'SubExp,Group,D');
for vi = 1:N_VAR
    fprintf(fid,',%s_Mean,%s_SR,%s_FEt', ...
            VARIANTS{vi,1},VARIANTS{vi,1},VARIANTS{vi,1});
end
fprintf(fid,'\n');
for se = 1:n_subexp
    e = all_results(se);
    fprintf(fid,'%s,%s,%d', e.label, e.group, e.D);
    for vi = 1:N_VAR
        fet_str = 'NaN';
        if ~isnan(e.stat_fet(vi))
            fet_str = sprintf('%d', round(e.stat_fet(vi)));
        end
        fprintf(fid,',%.6e,%.4f,%s', e.stat_mean(vi), e.stat_sr(vi), fet_str);
    end
    fprintf(fid,'\n');
end
fclose(fid);
fprintf('Saved: Ablation_summary.csv\n\n');

fprintf('=========================================================\n');
fprintf('  Ablation complete.  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS  (must appear after all script code in MATLAB)
% =========================================================================

function f = schwefel_shifted_fn(x, shift_vec, d)
    y = max(-500, min(500, x - shift_vec + 420.9687));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end

function conv = resample_conv(conv_raw, Max_FE, N_g, N_GRADES)
    Max_iter = floor(Max_FE / (N_g * N_GRADES));
    if length(conv_raw) >= 2
        src_idx = linspace(1, length(conv_raw), Max_iter);
        conv    = interp1(1:length(conv_raw), conv_raw, src_idx, 'previous');
    else
        conv = repmat(conv_raw(end), 1, Max_iter);
    end
    conv = conv(:)';
end

function entry = build_entry(label, group, D, best_f_mat, fe_to_tgt, ...
                               N_VAR, N_RUNS, SUCCESS_THRESH)
    entry.label      = label;
    entry.group      = group;
    entry.D          = D;
    entry.best_f_mat = best_f_mat;
    entry.fe_to_tgt  = fe_to_tgt;
    entry.stat_mean  = zeros(N_VAR,1);
    entry.stat_std   = zeros(N_VAR,1);
    entry.stat_sr    = zeros(N_VAR,1);
    entry.stat_fet   = nan(N_VAR,1);
    for vi = 1:N_VAR
        v = best_f_mat(vi, isfinite(best_f_mat(vi,:)));
        if ~isempty(v)
            entry.stat_mean(vi) = mean(v);
            entry.stat_std(vi)  = std(v);
        else
            entry.stat_mean(vi) = NaN;
            entry.stat_std(vi)  = NaN;
        end
        entry.stat_sr(vi)  = sum(best_f_mat(vi,:) < SUCCESS_THRESH) / N_RUNS;
        entry.stat_fet(vi) = median(fe_to_tgt(vi,:), 'omitnan');
    end
end

function save_subexp(RESULTS_DIR, label, entry, VARIANTS)
    mat_path = fullfile(RESULTS_DIR, sprintf('Ablation_%s.mat', label));
    save(mat_path, 'entry', 'VARIANTS');
    fprintf('  Saved: Ablation_%s.mat\n', label);
end
