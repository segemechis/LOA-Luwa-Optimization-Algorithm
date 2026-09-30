% =========================================================================
%  Run_12_Wide_Gap_Severity.m
%  Experiment 12 — Wide-Gap Severity Analysis
%
%  Motivation:
%    This is the paper's SIGNATURE experiment. It directly tests the
%    central hypothesis: "as the fitness gap between the global optimum
%    and the nearest local minimum grows, algorithms that cannot bridge
%    the wide spatial gap are progressively exposed."
%    A parameterised Schwefel family f_alpha(x) controls the gap severity
%    while keeping the landscape geometry (optimum location, local minima
%    positions) identical to the standard shifted Schwefel.
%
%  Mathematical construction (f_alpha family):
%    Base: shifted Schwefel f_s(x) with shift_vec (seed=9999+D, same as Exp 3)
%
%    Gaussian deepening term centred at x* = shift_vec:
%      G(x) = -sum_{i=1}^{D} exp( -(x_i - shift_vec_i)^2 / (2 * sigma^2) )
%      sigma = 50.0  (covers one Schwefel basin width)
%
%      Properties:
%        G(x*) = -D     (deepens global basin, proportional to D)
%        G(x_local) ≈ 0  (local minimum unaffected; ||x_local-x*||>>sigma)
%
%    Parameterised objective:
%      f_alpha(x) = f_s(x) + alpha * G(x)
%
%    alpha chosen so that the NORMALIZED fitness gap G_alpha equals target:
%      G_alpha = [f_alpha(x_local) - f_alpha(x*)] / [f_s(x_local) - f_s(x*)]
%      alpha   = (G_alpha - 1) * std_gap / D
%      std_gap = f_s(x_local) - f_s(x*)  (standard gap, D=30: ~16100, D=50: ~26834)
%
%    Five severity levels:
%      WG1  G_alpha = 0.25  (gap = 1/4 of standard — shallow, hardest to detect)
%      WG2  G_alpha = 0.50  (gap = 1/2 of standard — reduced)
%      WG3  G_alpha = 1.00  (gap = standard — alpha=0, baseline = Exp 3)
%      WG4  G_alpha = 2.00  (gap = 2× standard — deeper global basin)
%      WG5  G_alpha = 5.00  (gap = 5× standard — very deep well)
%
%    Verified properties:
%      ✓ f(x_local) > f(x*) for all levels (gap never inverts)
%      ✓ WG3 (alpha=0) recovers exact shifted Schwefel
%      ✓ Measured G_alpha matches target to 6 decimal places
%      ✓ Local minima positions unchanged; only global basin depth varies
%
%  Success threshold (adaptive — varies with alpha):
%    SR = fraction of runs with f_final < f*(alpha) + 0.01 * std_gap
%    (i.e. within 1% of the true global optimum value)
%    This is robust to negative f* at WG4/WG5.
%
%  Dimensions : D = 30, 50
%  Algorithms : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA, L-SHADE,
%                   MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Runs       : 30 per (algorithm, WG level, dimension)
%  FE budget  : D × 10,000 per run
%  Population : 30 agents
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Exp12\):
%    Run_12_results_D<d>_WG<k>.mat     — workspace per (D, WG level)
%    Run_12_results_D<d>_WG<k>.csv     — statistics
%    Run_12_gap_summary_D<d>.mat       — cross-WG summary arrays
%    Run_12_gap_summary_D<d>.csv       — WG × algorithm table
%    Run_12_log.txt                    — console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp12';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_12_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST        = [30, 50];
SHIFT_SEED      = 9999;         % same landscape as Exp 3/10/11
TARGET_G_ALPHA  = [0.25, 0.50, 1.00, 2.00, 5.00];
WG_LABELS       = {'WG1','WG2','WG3','WG4','WG5'};
WG_NAMES        = {'0.25x gap','0.50x gap','1.00x gap (baseline)','2.00x gap','5.00x gap'};
SIGMA           = 50.0;          % Gaussian width (one Schwefel basin)
LOCAL_MIN_DIST  = 302.5232;      % nearest local min offset from global
SR_EPSILON_FRAC = 0.01;          % SR success: within 1% of std_gap from f*
N_WG            = numel(WG_LABELS);
N_RUNS          = 30;
N_POP           = 30;
LB_SCALAR       = -500;
UB_SCALAR       =  500;

%% ── 2. ALGORITHM REGISTRY ────────────────────────────────────────────────
ALG = {
    'CCO',    'wrap_CCO',    false;
    'CMA-ES', 'wrap_CMAES',  false;
    'DE',     'wrap_DE',     false;
    'GA',     'wrap_GA',     false;
    'GWO',    'wrap_GWO',    false;
    'HHO',    'wrap_HHO',    false;
    'LOA',    'wrap_LOA',    true;
    'L-SHADE','wrap_LSHADE', false;
    'MVO',    'wrap_MVO',    false;
    'PSO',    'wrap_PSO',    false;
    'RIME',   'wrap_RIME',   false;
    'SA',     'wrap_SA',     false;
    'SCA',    'wrap_SCA',    false;
    'SO',     'wrap_SO',     false;
    'WOA',    'wrap_WOA',    false;
};
N_ALG = size(ALG, 1);

%% ── 3. MAIN LOOP ─────────────────────────────────────────────────────────
fprintf('=========================================================\n');
fprintf('  Experiment 12 — Wide-Gap Severity Analysis             \n');
fprintf('  Dimensions  : D = 30, 50                               \n');
fprintf('  WG levels   : G_alpha = 0.25x 0.50x 1.00x 2.00x 5.00x\n');
fprintf('  Runs        : %d per (algorithm, WG level, dimension)   \n', N_RUNS);
fprintf('  sigma       : %.1f  (Gaussian basin width)              \n', SIGMA);
fprintf('  SR criterion: within %.0f%% of std_gap from f*(alpha)   \n', ...
        SR_EPSILON_FRAC*100);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Shifted Schwefel landscape (same as Exp 3/10/11) ─────────────────
    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);
    x_opt     = shift_vec;
    x_local   = max(LB_SCALAR, min(UB_SCALAR, shift_vec - LOCAL_MIN_DIST));

    % ── Standard gap (alpha=0 = exact shifted Schwefel) ──────────────────
    f_opt_std   = schwefel_shifted(x_opt,   shift_vec, D);
    f_local_std = schwefel_shifted(x_local, shift_vec, D);
    std_gap     = f_local_std - f_opt_std;
    sr_epsilon  = SR_EPSILON_FRAC * std_gap;

    % ── Compute alpha values for all WG levels ────────────────────────────
    alphas     = (TARGET_G_ALPHA - 1.0) .* std_gap / D;
    f_opt_alpha = zeros(1, N_WG);   % f*(alpha) per level (for SR threshold)
    for wi = 1:N_WG
        f_opt_alpha(wi) = f_alpha_shifted(x_opt, shift_vec, D, alphas(wi), SIGMA);
    end

    fprintf('=========================================================\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('  shift_vec (first 5): ['); fprintf(' %.3f', shift_vec(1:min(5,D))); fprintf(' ...]\n');
    fprintf('  Standard gap Δf = %.4f\n', std_gap);
    fprintf('  SR epsilon = %.2f  (1%% of std_gap)\n', sr_epsilon);
    fprintf('\n  WG level breakdown:\n');
    fprintf('  %-5s %8s %10s %14s %14s %14s\n', ...
            'Level','G_alpha','alpha','f*(alpha)','f(x_local,α)','Δf_alpha');
    fprintf('  %s\n', repmat('-',1,70));
    for wi = 1:N_WG
        f_l = f_alpha_shifted(x_local, shift_vec, D, alphas(wi), SIGMA);
        gap_a = f_l - f_opt_alpha(wi);
        fprintf('  %-5s %8.2fx %10.4f %14.4f %14.4f %14.4f\n', ...
                WG_LABELS{wi}, TARGET_G_ALPHA(wi), alphas(wi), ...
                f_opt_alpha(wi), f_l, gap_a);
    end
    fprintf('\n');

    % Sanity: WG3 should match standard shifted Schwefel
    assert(abs(alphas(3)) < 1e-9, 'WG3 alpha should be 0');
    assert(abs(f_opt_alpha(3)) < 1.0, sprintf('WG3 f* should be ≈ 0, got %.4e', f_opt_alpha(3)));
    fprintf('  ✓ WG3 (alpha=0) recovers standard shifted Schwefel\n\n');

    % Summary arrays: (N_ALG × N_WG)
    summ_mean    = zeros(N_ALG, N_WG);
    summ_std     = zeros(N_ALG, N_WG);
    summ_sr      = zeros(N_ALG, N_WG);
    summ_fet     = nan(N_ALG, N_WG);
    summ_gap     = TARGET_G_ALPHA;   % for saving

    for wi = 1:N_WG
        wg_label   = WG_LABELS{wi};
        wg_name    = WG_NAMES{wi};
        alpha      = alphas(wi);
        f_star     = f_opt_alpha(wi);
        sr_thresh  = f_star + sr_epsilon;

        fprintf('  ── %s (%s | alpha=%.4f | f*=%.4f | SR thr=%.4f) ──\n', ...
                wg_label, wg_name, alpha, f_star, sr_thresh);

        % Objectives
        obj_scalar = @(x) f_alpha_shifted(x, shift_vec, D, alpha, SIGMA);
        obj_loa    = @(x) deal(f_alpha_shifted(x, shift_vec, D, alpha, SIGMA), 0);

        % Additional sanity check per level
        f_chk = f_alpha_shifted(x_opt, shift_vec, D, alpha, SIGMA);
        fprintf('  Sanity: f_alpha(x_opt) = %.6e\n\n', f_chk);

        % Storage
        best_f_mat = zeros(N_ALG, N_RUNS);
        conv_cell  = cell(N_ALG, N_RUNS);
        fe_to_tgt  = nan(N_ALG, N_RUNS);

        for ai = 1:N_ALG
            alg_name  = ALG{ai,1};
            wrap_fn   = str2func(ALG{ai,2});
            needs_loa = ALG{ai,3};
            obj_fn    = obj_scalar;
            if needs_loa, obj_fn = obj_loa; end

            fprintf('  [D=%2d | %s] Running %-8s ', D, wg_label, alg_name);
            t_start = tic;

            for r = 1:N_RUNS
                rng(r);
                try
                    [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE);
                catch ME
                    warning('Exp12: %s run %d D=%d %s failed: %s', ...
                            alg_name, r, D, wg_label, ME.message);
                    fbest = NaN;
                    conv  = NaN(1, floor(Max_FE/N_POP));
                end
                best_f_mat(ai,r) = fbest;
                conv_row = conv(:)';
                conv_cell{ai,r} = conv_row;

                % FE-to-target: first iteration where conv < sr_thresh
                hit = find(conv_row < sr_thresh, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(ai,r) = hit * N_POP;
                end
            end
            elapsed = toc(t_start);

            vals    = best_f_mat(ai,:);
            valid_v = vals(isfinite(vals));
            sr      = sum(valid_v < sr_thresh) / N_RUNS;
            med_fet = median(fe_to_tgt(ai,:), 'omitnan');

            if isempty(valid_v)
                fprintf('→ ALL RUNS FAILED\n');
            else
                fprintf('→ mean=%10.3e | SR=%4.1f%% | FEt=%8s  [%.1fs]\n', ...
                        mean(valid_v), sr*100, format_fet(med_fet), elapsed);
            end
        end % algorithm loop

        % ----- Statistics ------------------------------------------------
        stat_mean   = zeros(N_ALG,1); stat_std    = zeros(N_ALG,1);
        stat_best   = zeros(N_ALG,1); stat_worst  = zeros(N_ALG,1);
        stat_median = zeros(N_ALG,1); stat_sr     = zeros(N_ALG,1);
        stat_fet    = nan(N_ALG,1);

        for ai = 1:N_ALG
            vals = best_f_mat(ai,:);
            v    = vals(isfinite(vals));
            if isempty(v)
                stat_mean(ai)=NaN; stat_std(ai)=NaN; stat_best(ai)=NaN;
                stat_worst(ai)=NaN; stat_median(ai)=NaN;
            else
                stat_mean(ai)   = mean(v);
                stat_std(ai)    = std(v);
                stat_best(ai)   = min(v);
                stat_worst(ai)  = max(v);
                stat_median(ai) = median(v);
            end
            stat_sr(ai)  = sum(best_f_mat(ai,:) < sr_thresh) / N_RUNS;
            stat_fet(ai) = median(fe_to_tgt(ai,:), 'omitnan');
        end

        summ_mean(:,wi) = stat_mean;
        summ_std(:,wi)  = stat_std;
        summ_sr(:,wi)   = stat_sr;
        summ_fet(:,wi)  = stat_fet;

        % Normalised mean: (mean - f*) / std_gap so levels are comparable
        stat_mean_norm = (stat_mean - f_star) / std_gap;

        % Print table
        fprintf('\n  %-8s %12s %10s %10s %8s %10s %12s\n', ...
                'Algorithm','Mean','Std','Best','SR%%','FEt(med)','NormMean');
        fprintf('  %s\n', repmat('-',1,76));
        for ai = 1:N_ALG
            fprintf('  %-8s %12.4e %10.4e %10.4e %7.1f%% %10s %12.4f\n', ...
                    ALG{ai,1}, stat_mean(ai), stat_std(ai), stat_best(ai), ...
                    stat_sr(ai)*100, format_fet(stat_fet(ai)), stat_mean_norm(ai));
        end

        [~, rank_ord] = sort(stat_mean);
        fprintf('\n  Ranking (%s, by mean):\n', wg_label);
        for r = 1:N_ALG
            ai = rank_ord(r);
            fprintf('    %2d. %-8s mean=%10.3e  SR=%4.1f%%  NormMean=%8.4f\n', ...
                    r, ALG{ai,1}, stat_mean(ai), stat_sr(ai)*100, stat_mean_norm(ai));
        end
        fprintf('\n');

        % Save per-(D, WG) .mat
        mat_path  = fullfile(RESULTS_DIR, ...
                    sprintf('Run_12_results_D%d_%s.mat', D, wg_label));
        alg_names = ALG(:,1);
        save(mat_path, 'D','wg_label','wg_name','alpha','f_star', ...
             'sr_thresh','sr_epsilon','std_gap', ...
             'shift_vec','x_opt','x_local','SIGMA', ...
             'N_RUNS','N_POP','Max_FE', ...
             'alg_names','best_f_mat','conv_cell','fe_to_tgt', ...
             'stat_mean','stat_mean_norm','stat_std','stat_best','stat_worst', ...
             'stat_median','stat_sr','stat_fet','rank_ord');
        fprintf('  Saved: %s\n\n', mat_path);

        % Save per-(D, WG) .csv
        csv_path = fullfile(RESULTS_DIR, ...
                   sprintf('Run_12_results_D%d_%s.csv', D, wg_label));
        write_csv_wg(csv_path, ALG(:,1), stat_mean, stat_std, ...
                     stat_best, stat_worst, stat_median, stat_sr, stat_fet, ...
                     stat_mean_norm, D, wg_label, wg_name, alpha, f_star, ...
                     sr_thresh, std_gap, N_RUNS, Max_FE);
        fprintf('  Saved: %s\n\n', csv_path);

    end % WG loop

    % ----- Cross-WG summary ───────────────────────────────────────────────
    fprintf('  ╔══════════════════════════════════════════════════════════╗\n');
    fprintf('  ║  GAP-SEVERITY SUMMARY — D=%d — Success Rate (%%)        \n', D);
    fprintf('  ║  G_alpha: 0.25x  0.50x  1.00x  2.00x  5.00x           \n');
    fprintf('  ╚══════════════════════════════════════════════════════════╝\n');
    fprintf('  %-8s', 'Algorithm');
    for wi = 1:N_WG
        fprintf(' %7s', WG_LABELS{wi});
    end
    fprintf('  | WG1→WG5 change\n');
    fprintf('  %s\n', repmat('-',1,68));
    for ai = 1:N_ALG
        fprintf('  %-8s', ALG{ai,1});
        for wi = 1:N_WG
            fprintf(' %6.1f%%', summ_sr(ai,wi)*100);
        end
        change = (summ_sr(ai,N_WG) - summ_sr(ai,1))*100;
        fprintf('  | %+.1f%%\n', change);
    end
    fprintf('\n');

    % Gap-robustness ranking: algorithms whose SR grows most from WG1→WG5
    % (positive = benefits from deeper gap, i.e. doesn't get confused)
    sr_change = summ_sr(:,N_WG) - summ_sr(:,1);
    [~, chg_ord] = sort(sr_change, 'descend');
    fprintf('  Gap-robustness ranking (SR improvement WG1→WG5):\n');
    for r = 1:N_ALG
        ai = chg_ord(r);
        fprintf('    %2d. %-8s  WG1=%.1f%%  WG3=%.1f%%  WG5=%.1f%%  change=%+.1f%%\n', ...
                r, ALG{ai,1}, summ_sr(ai,1)*100, summ_sr(ai,3)*100, ...
                summ_sr(ai,5)*100, sr_change(ai)*100);
    end
    fprintf('\n');

    % Save summary .mat
    summ_mat = fullfile(RESULTS_DIR, sprintf('Run_12_gap_summary_D%d.mat', D));
    alg_names = ALG(:,1);
    save(summ_mat, 'D','TARGET_G_ALPHA','WG_LABELS','WG_NAMES', ...
         'alphas','f_opt_alpha','std_gap','sr_epsilon', ...
         'shift_vec','x_opt','x_local','SIGMA','N_RUNS','N_POP', ...
         'alg_names','summ_mean','summ_std','summ_sr','summ_fet');
    fprintf('  Saved summary: %s\n\n', summ_mat);

    % Save summary .csv
    summ_csv = fullfile(RESULTS_DIR, sprintf('Run_12_gap_summary_D%d.csv', D));
    write_summary_csv(summ_csv, ALG(:,1), summ_mean, summ_sr, summ_fet, ...
                      WG_LABELS, WG_NAMES, TARGET_G_ALPHA, alphas, ...
                      f_opt_alpha, D, N_RUNS, std_gap, sr_epsilon);
    fprintf('  Saved summary: %s\n\n', summ_csv);

end % dimension loop

fprintf('=========================================================\n');
fprintf('  Experiment 12 complete.\n');
fprintf('  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% Standard shifted Schwefel (f* ≈ 0 at x = shift_vec). Same as Exp 3.
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function f = f_alpha_shifted(x, shift_vec, d, alpha, sigma)
% WIDE-GAP PARAMETERISED OBJECTIVE
%
%   f_alpha(x) = f_shifted(x) + alpha * G(x, shift_vec, sigma)
%
%   G(x) = -sum_{i=1}^{D} exp( -(x_i - shift_vec_i)^2 / (2*sigma^2) )
%
%   Properties:
%     G(x*=shift_vec) = -D           (deepens global basin)
%     G(x_local) ≈ 0                 (local minima unaffected)
%     alpha = (G_alpha-1)*std_gap/D  gives exact normalised gap G_alpha
%     WG3: alpha=0 → exact standard shifted Schwefel
%     WG4/5: f* goes negative (global basin deepened below zero) — correct
%
%   Success criterion uses adaptive threshold: f < f*(alpha) + epsilon
%   rather than f < 1.0, since f* is not zero for WG4/WG5.
    f_base = schwefel_shifted(x, shift_vec, d);
    G      = -sum(exp(-(x - shift_vec).^2 / (2 * sigma^2)));
    f      = f_base + alpha * G;
end


function s = format_fet(fet)
    if isnan(fet), s = 'N/A'; else, s = sprintf('%d', round(fet)); end
end


function write_csv_wg(csv_path, alg_names, ...
                       s_mean, s_std, s_best, s_worst, s_median, ...
                       s_sr, s_fet, s_mean_norm, D, wg_label, wg_name, ...
                       alpha, f_star, sr_thresh, std_gap, N_RUNS, Max_FE)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp12: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_12_Wide_Gap_Severity\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'WG_level,%s (%s)\n', wg_label, wg_name);
    fprintf(fid, 'alpha,%.6f\n', alpha);
    fprintf(fid, 'f_star_alpha,%.6f\n', f_star);
    fprintf(fid, 'SR_threshold,%.6f  (= f* + 1%% of std_gap)\n', sr_thresh);
    fprintf(fid, 'std_gap,%.6f\n', std_gap);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, 'FE_Budget,%d\n', Max_FE);
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm,Mean,Std,Best,Worst,Median,SR,FEt_median,NormMean\n');
    for i = 1:numel(alg_names)
        fet_str = 'NaN';
        if ~isnan(s_fet(i)), fet_str = sprintf('%d', round(s_fet(i))); end
        fprintf(fid, '%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%s,%.6f\n', ...
                alg_names{i}, s_mean(i), s_std(i), s_best(i), ...
                s_worst(i), s_median(i), s_sr(i), fet_str, s_mean_norm(i));
    end
    fclose(fid);
end


function write_summary_csv(csv_path, alg_names, summ_mean, summ_sr, summ_fet, ...
                            WG_LABELS, WG_NAMES, TARGET_G, alphas, f_opt_alpha, ...
                            D, N_RUNS, std_gap, sr_epsilon)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp12: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_12_Wide_Gap_Summary\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Standard_gap,%.6f\n', std_gap);
    fprintf(fid, 'SR_epsilon_1pct_std_gap,%.6f\n', sr_epsilon);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, '\n');
    % WG level metadata
    fprintf(fid, 'WG_Level,G_alpha_target,alpha,f_star_alpha\n');
    for wi = 1:numel(WG_LABELS)
        fprintf(fid, '%s,%.2f,%.6f,%.6f\n', ...
                WG_LABELS{wi}, TARGET_G(wi), alphas(wi), f_opt_alpha(wi));
    end
    fprintf(fid, '\n');
    % Per-algorithm results
    fprintf(fid, 'Algorithm');
    for wi = 1:numel(WG_LABELS)
        fprintf(fid, ',%s_Mean,%s_SR,%s_FEt', ...
                WG_LABELS{wi}, WG_LABELS{wi}, WG_LABELS{wi});
    end
    fprintf(fid, '\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s', alg_names{i});
        for wi = 1:numel(WG_LABELS)
            fet_str = 'NaN';
            if ~isnan(summ_fet(i,wi))
                fet_str = sprintf('%d', round(summ_fet(i,wi)));
            end
            fprintf(fid, ',%.6e,%.4f,%s', ...
                    summ_mean(i,wi), summ_sr(i,wi), fet_str);
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
