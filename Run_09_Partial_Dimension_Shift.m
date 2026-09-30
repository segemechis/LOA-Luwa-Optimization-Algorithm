% =========================================================================
%  Run_09_Partial_Dimension_Shift.m
%  Experiment 9 — Partial-Dimension Shift
%
%  Motivation:
%    Experiments 3 and 8 shifted ALL dimensions simultaneously. In many
%    real-world problems, only a subset of variables are displaced from
%    their canonical operating range — the rest behave as expected. This
%    experiment tests whether algorithms can exploit the free (unshifted)
%    dimensions while navigating the shifted ones, and how performance
%    degrades as the fraction of shifted dimensions grows from 25% to 100%.
%
%  Shift construction (CORRECTED unified formula):
%    A full shift vector s ∈ R^D is drawn once (seed=PART_SEED=7777):
%      Free dims (i > n_shifted):    s_i = 420.9687
%      Shifted dims (i <= n_shifted): s_i ~ U[-200, 200]
%
%    Objective (unified formula, valid for ALL dims):
%      y = x - s + 420.9687
%      ỹ = clamp(y, -500, 500)
%      f(x) = 418.9829*D - sum_i( ỹ_i * sin(sqrt(|ỹ_i|)) )
%
%    Why s_i=420.9687 for free dims:
%      y_i = x_i - 420.9687 + 420.9687 = x_i  → raw Schwefel → opt at x_i=420.9687 ✓
%    Why s_i=random for shifted dims:
%      y_i = x_i - s_i + 420.9687             → opt at x_i=s_i ✓
%
%    Global optimum:
%      x*_i = s_i  (shifted dims)   → x*_i = random U[-200,200]
%      x*_i = 420.9687 (free dims)  → standard Schwefel optimum
%    f* ≈ 0 at x* (verified by sanity check before each fraction level).
%
%  Fraction levels tested:
%    P25  25% of dims shifted   (D=30: 8 shifted,  22 free)
%    P50  50% of dims shifted   (D=30: 15 shifted, 15 free)
%    P75  75% of dims shifted   (D=30: 22 shifted,  8 free)
%    P100 100% of dims shifted  (D=30: 30 shifted,  0 free) ← same as Exp3
%
%  Metrics: Mean, Std, Best, Worst, Median, SR (f<1.0), FE-to-target
%
%  Dimensions : D = 30, 50
%  Algorithms : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA, L-SHADE,
%                   MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Runs       : 30 per (algorithm, fraction, dimension)
%  FE budget  : D × 10,000 per run
%  Population : 30 agents
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Exp09\):
%    Run_09_results_D<d>_P<pct>.mat    — workspace per (D, fraction)
%    Run_09_results_D<d>_P<pct>.csv    — statistics per (D, fraction)
%    Run_09_fraction_summary_D<d>.mat  — cross-fraction arrays
%    Run_09_fraction_summary_D<d>.csv  — fraction × algorithm table
%    Run_09_log.txt                    — console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp09';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_09_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST       = [30, 50];
FRACTIONS      = [0.25, 0.50, 0.75, 1.00];
FRAC_LABELS    = {'P25','P50','P75','P100'};
FRAC_PCT       = [25, 50, 75, 100];
PART_SEED      = 7777;     % seed for per-dimension shift values
N_RUNS         = 30;
N_POP          = 30;
LB_SCALAR      = -500;
UB_SCALAR      =  500;
SUCCESS_THRESH = 1.0;

N_FRAC = numel(FRACTIONS);

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
fprintf('  Experiment 9 — Partial-Dimension Shift                 \n');
fprintf('  Dimensions  : D = 30, 50                               \n');
fprintf('  Fractions   : 25%% 50%% 75%% 100%% of dims shifted     \n');
fprintf('  Runs        : %d per (algorithm, fraction, dimension)   \n', N_RUNS);
fprintf('  Shift seed  : %d (per-dim values, fixed across fractions)\n', PART_SEED);
fprintf('  Success thr : f < %.1f                                  \n', SUCCESS_THRESH);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Draw full per-dimension shift values once for this D ─────────────
    % shift_full(i) is the displacement used when dimension i is "shifted".
    % Free dims instead use 420.9687 in the unified shift vector.
    rng(PART_SEED);
    shift_full = -200 + 400 * rand(1, D);   % U[-200, 200]^D

    fprintf('=========================================================\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('  shift_full(1:5) = [%.2f %.2f %.2f %.2f %.2f ...]\n', ...
            shift_full(1), shift_full(2), shift_full(3), ...
            shift_full(4), shift_full(5));
    fprintf('=========================================================\n\n');

    % Summary arrays: (N_ALG × N_FRAC)
    summ_mean = zeros(N_ALG, N_FRAC);
    summ_std  = zeros(N_ALG, N_FRAC);
    summ_sr   = zeros(N_ALG, N_FRAC);
    summ_fet  = nan(N_ALG, N_FRAC);

    for fi = 1:N_FRAC
        frac       = FRACTIONS(fi);
        frac_label = FRAC_LABELS{fi};
        n_shifted  = round(frac * D);
        n_free     = D - n_shifted;

        % ── Build unified shift vector for this fraction ─────────────────
        % Free dims:    shift_vec(i) = 420.9687  → y_i = x_i → opt at x_i=420.9687
        % Shifted dims: shift_vec(i) = shift_full(i)          → opt at x_i=shift_full(i)
        shift_vec          = 420.9687 * ones(1, D);
        shift_vec(1:n_shifted) = shift_full(1:n_shifted);

        % ── Build optimum vector ─────────────────────────────────────────
        x_opt          = 420.9687 * ones(1, D);
        x_opt(1:n_shifted) = shift_full(1:n_shifted);

        fprintf('  ── %s  (%d%% shifted: %d dims shifted, %d free) ────────\n', ...
                frac_label, FRAC_PCT(fi), n_shifted, n_free);

        % Sanity check
        f_check = schwefel_shifted(x_opt, shift_vec, D);
        fprintf('  Sanity: f(x_opt) = %.6e  (should be ≈ 0)\n', f_check);
        if f_check > 1.0
            error('Exp09: sanity check failed D=%d %s. Abort.', D, frac_label);
        end
        f_rand_check = schwefel_shifted(lb + rand(1,D).*(ub-lb), shift_vec, D);
        fprintf('  Sanity: f(random) = %.2f  (should be >> 0)\n\n', f_rand_check);

        % Objectives
        obj_scalar = @(x) schwefel_shifted(x, shift_vec, D);
        obj_loa    = @(x) deal(schwefel_shifted(x, shift_vec, D), 0);

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

            fprintf('  [D=%2d | %s] Running %-8s ', D, frac_label, alg_name);
            t_start = tic;

            for r = 1:N_RUNS
                rng(r);
                try
                    [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE);
                catch ME
                    warning('Exp09: %s run %d D=%d %s failed: %s', ...
                            alg_name, r, D, frac_label, ME.message);
                    fbest = NaN;
                    conv  = NaN(1, floor(Max_FE/N_POP));
                end
                best_f_mat(ai,r) = fbest;
                conv_row = conv(:)';
                conv_cell{ai,r} = conv_row;

                hit = find(conv_row < SUCCESS_THRESH, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(ai,r) = hit * N_POP;
                end
            end
            elapsed = toc(t_start);

            vals      = best_f_mat(ai,:);
            valid_v   = vals(isfinite(vals));
            sr        = sum(valid_v < SUCCESS_THRESH) / N_RUNS;
            med_fet   = median(fe_to_tgt(ai,:), 'omitnan');

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
            stat_sr(ai)  = sum(best_f_mat(ai,:) < SUCCESS_THRESH) / N_RUNS;
            stat_fet(ai) = median(fe_to_tgt(ai,:), 'omitnan');
        end

        summ_mean(:,fi) = stat_mean;
        summ_std(:,fi)  = stat_std;
        summ_sr(:,fi)   = stat_sr;
        summ_fet(:,fi)  = stat_fet;

        % Print table
        fprintf('\n  %-8s %12s %10s %10s %8s %10s\n', ...
                'Algorithm','Mean','Std','Best','SR%%','FEt(med)');
        fprintf('  %s\n', repmat('-',1,64));
        for ai = 1:N_ALG
            fprintf('  %-8s %12.4e %10.4e %10.4e %7.1f%% %10s\n', ...
                    ALG{ai,1}, stat_mean(ai), stat_std(ai), stat_best(ai), ...
                    stat_sr(ai)*100, format_fet(stat_fet(ai)));
        end

        [~, rank_ord] = sort(stat_mean);
        fprintf('\n  Ranking (%s, by mean):\n', frac_label);
        for r = 1:N_ALG
            ai = rank_ord(r);
            fprintf('    %2d. %-8s mean=%10.3e  SR=%4.1f%%\n', ...
                    r, ALG{ai,1}, stat_mean(ai), stat_sr(ai)*100);
        end
        fprintf('\n');

        % Save per-(D, fraction) .mat
        mat_path  = fullfile(RESULTS_DIR, ...
                    sprintf('Run_09_results_D%d_%s.mat', D, frac_label));
        alg_names = ALG(:,1);
        save(mat_path, 'D','frac','frac_label','n_shifted','n_free', ...
             'shift_vec','x_opt','N_RUNS','N_POP','Max_FE','SUCCESS_THRESH', ...
             'alg_names','best_f_mat','conv_cell','fe_to_tgt', ...
             'stat_mean','stat_std','stat_best','stat_worst', ...
             'stat_median','stat_sr','stat_fet','rank_ord');
        fprintf('  Saved: %s\n\n', mat_path);

        % Save per-(D, fraction) .csv
        csv_path = fullfile(RESULTS_DIR, ...
                   sprintf('Run_09_results_D%d_%s.csv', D, frac_label));
        write_csv_fraction(csv_path, ALG(:,1), stat_mean, stat_std, ...
                           stat_best, stat_worst, stat_median, stat_sr, stat_fet, ...
                           D, frac, n_shifted, n_free, N_RUNS, Max_FE, SUCCESS_THRESH);
        fprintf('  Saved: %s\n\n', csv_path);

    end % fraction loop

    % ----- Cross-fraction summary ─────────────────────────────────────────
    fprintf('  ╔══════════════════════════════════════════════════════════╗\n');
    fprintf('  ║  FRACTION SUMMARY — D=%d — Success Rate (%%)            \n', D);
    fprintf('  ╚══════════════════════════════════════════════════════════╝\n');
    fprintf('  %-8s', 'Algorithm');
    for fi = 1:N_FRAC
        fprintf(' %7s', FRAC_LABELS{fi});
    end
    fprintf('  | P25→P100 drop\n');
    fprintf('  %s\n', repmat('-',1,65));
    for ai = 1:N_ALG
        fprintf('  %-8s', ALG{ai,1});
        for fi = 1:N_FRAC
            fprintf(' %6.1f%%', summ_sr(ai,fi)*100);
        end
        drop = (summ_sr(ai,1) - summ_sr(ai,N_FRAC)) * 100;
        fprintf('  | %+.1f%%\n', -drop);
    end
    fprintf('\n');

    % Robustness ranking: smallest SR drop P25→P100
    sr_drop = summ_sr(:,1) - summ_sr(:,N_FRAC);
    [~, drop_ord] = sort(sr_drop);
    fprintf('  Robustness ranking (smallest SR drop P25→P100):\n');
    for r = 1:N_ALG
        ai = drop_ord(r);
        fprintf('    %2d. %-8s  SR: P25=%.1f%% P50=%.1f%% P75=%.1f%% P100=%.1f%%\n', ...
                r, ALG{ai,1}, summ_sr(ai,1)*100, summ_sr(ai,2)*100, ...
                summ_sr(ai,3)*100, summ_sr(ai,4)*100);
    end
    fprintf('\n');

    % Save summary .mat
    summ_mat = fullfile(RESULTS_DIR, ...
               sprintf('Run_09_fraction_summary_D%d.mat', D));
    alg_names = ALG(:,1);
    save(summ_mat, 'D','FRACTIONS','FRAC_LABELS','FRAC_PCT', ...
         'N_RUNS','N_POP','SUCCESS_THRESH','shift_full', ...
         'alg_names','summ_mean','summ_std','summ_sr','summ_fet');
    fprintf('  Saved summary: %s\n\n', summ_mat);

    % Save summary .csv
    summ_csv = fullfile(RESULTS_DIR, ...
               sprintf('Run_09_fraction_summary_D%d.csv', D));
    write_summary_csv(summ_csv, ALG(:,1), summ_mean, summ_sr, summ_fet, ...
                      FRACTIONS, FRAC_LABELS, FRAC_PCT, D, N_RUNS, SUCCESS_THRESH);
    fprintf('  Saved summary: %s\n\n', summ_csv);

end % dimension loop

fprintf('=========================================================\n');
fprintf('  Experiment 9 complete.\n');
fprintf('  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% SCHWEFEL_SHIFTED  Partial-dimension shifted Schwefel (f* ≈ 0 at x = x_opt).
%
%  Unified formula (works for both free and shifted dims):
%    y = x - shift_vec + 420.9687
%    ỹ = clamp(y, -500, 500)
%    f = 418.9829*d - sum(ỹ.*sin(sqrt(|ỹ|)))
%
%  Free dims:    shift_vec(i) = 420.9687 → y_i = x_i → opt at x_i = 420.9687
%  Shifted dims: shift_vec(i) = s_i      → y_i = x_i - s_i + 420.9687
%                                                → opt at x_i = s_i
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function s = format_fet(fet)
    if isnan(fet), s = 'N/A'; else, s = sprintf('%d', round(fet)); end
end


function write_csv_fraction(csv_path, alg_names, ...
                             s_mean, s_std, s_best, s_worst, s_median, ...
                             s_sr, s_fet, D, frac, n_shifted, n_free, ...
                             N_RUNS, Max_FE, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp09: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_09_Partial_Dimension_Shift\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Shifted_fraction,%.2f\n', frac);
    fprintf(fid, 'N_shifted_dims,%d\n', n_shifted);
    fprintf(fid, 'N_free_dims,%d\n', n_free);
    fprintf(fid, 'Success_threshold,%.1f\n', thresh);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, 'FE_Budget,%d\n', Max_FE);
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm,Mean,Std,Best,Worst,Median,SR,FEt_median\n');
    for i = 1:numel(alg_names)
        fet_str = 'NaN';
        if ~isnan(s_fet(i)), fet_str = sprintf('%d', round(s_fet(i))); end
        fprintf(fid, '%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%s\n', ...
                alg_names{i}, s_mean(i), s_std(i), s_best(i), ...
                s_worst(i), s_median(i), s_sr(i), fet_str);
    end
    fclose(fid);
end


function write_summary_csv(csv_path, alg_names, summ_mean, summ_sr, summ_fet, ...
                            FRACTIONS, FRAC_LABELS, FRAC_PCT, D, N_RUNS, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp09: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_09_Partial_Dimension_Shift_Summary\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Success_threshold,%.1f\n', thresh);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm');
    for fi = 1:numel(FRACTIONS)
        fprintf(fid, ',%s_Mean,%s_SR,%s_FEt', ...
                FRAC_LABELS{fi}, FRAC_LABELS{fi}, FRAC_LABELS{fi});
    end
    fprintf(fid, '\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s', alg_names{i});
        for fi = 1:numel(FRACTIONS)
            fet_str = 'NaN';
            if ~isnan(summ_fet(i,fi))
                fet_str = sprintf('%d', round(summ_fet(i,fi)));
            end
            fprintf(fid, ',%.6e,%.4f,%s', ...
                    summ_mean(i,fi), summ_sr(i,fi), fet_str);
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
