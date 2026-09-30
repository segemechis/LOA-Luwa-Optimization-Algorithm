% =========================================================================
%  Run_08_Progressive_Shift_Severity.m
%  Experiment 8 — Progressive Shift Severity
%
%  Motivation:
%    Experiment 3 (Shifted Schwefel) used a single fixed shift magnitude.
%    This experiment asks: how does each algorithm's performance degrade
%    as the global optimum is progressively displaced further from the
%    canonical search region? The shift direction is fixed; only the
%    magnitude scales from 0% (standard Schwefel) to 100% (maximum safe
%    displacement). This isolates spatial displacement as a single
%    controlled variable and produces a performance–severity curve per
%    algorithm — the key figure for the paper's narrative.
%
%  Shift construction:
%    A fixed unit-direction vector v is drawn once (seed=42), shared
%    across all severity levels, dimensions, and algorithms.
%    shift_vec(sev) = sev * MAX_SHIFT * v,   sev ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
%    MAX_SHIFT = 200  (keeps optimum safely inside [-500,500]^D at all levels)
%
%    Objective (identical to Exp 3):
%      y = x - shift_vec + 420.9687
%      ỹ = clamp(y, -500, 500)
%      f(x) = 418.9829*D - sum_i( ỹ_i * sin(sqrt(|ỹ_i|)) )
%    Global optimum: f* ≈ 0 at x* = shift_vec
%    S0 (sev=0): shift=0 → standard Schwefel (x*=420.9687)
%
%  Severity levels:
%    S0  0%  → shift = 0                  (standard Schwefel baseline)
%    S1  20% → ||shift|| = 40  per D^0.5
%    S2  40% → ||shift|| = 80
%    S3  60% → ||shift|| = 120
%    S4  80% → ||shift|| = 160
%    S5  100%→ ||shift|| = 200
%
%  Metrics reported per (algorithm, severity, dimension):
%    • Mean final fitness (30 runs)
%    • Std, Best, Worst, Median
%    • Success Rate (SR): fraction of runs with f_final < SUCCESS_THRESH
%    • FE to target (FE_t): median FEs to first reach f < SUCCESS_THRESH
%      (NaN if the algorithm never reaches threshold in a run)
%
%  Dimensions : D = 30, 50
%  Algorithms : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA, L-SHADE,
%                   MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Runs       : 30 per (algorithm, severity, dimension)
%  FE budget  : D × 10,000 per run
%  Population : 30 agents (N_g=6 per grade for LOA → 30 total)
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Exp08\):
%    Run_08_results_D<d>_S<k>.mat    — full workspace per (D, severity)
%    Run_08_results_D<d>_S<k>.csv    — statistics per (D, severity)
%    Run_08_severity_summary_D<d>.mat — cross-severity arrays for plotting
%    Run_08_severity_summary_D<d>.csv — severity × algorithm table
%    Run_08_log.txt                   — console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp08';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_08_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST       = [30, 50];
SEVERITIES     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];   % shift as fraction of MAX_SHIFT
SEV_LABELS     = {'S0','S1','S2','S3','S4','S5'};
MAX_SHIFT      = 200.0;    % max safe displacement per component (U[-200,200] range)
DIR_SEED       = 42;       % seed for fixed shift direction
N_RUNS         = 30;
N_POP          = 30;
LB_SCALAR      = -500;
UB_SCALAR      =  500;
SUCCESS_THRESH = 1.0;      % f < 1.0 counts as success (near-optimal)

N_SEV = numel(SEVERITIES);

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
fprintf('  Experiment 8 — Progressive Shift Severity              \n');
fprintf('  Dimensions  : D = 30, 50                               \n');
fprintf('  Severities  : 0%% 20%% 40%% 60%% 80%% 100%%           \n');
fprintf('  Runs        : %d per (algorithm, severity, dimension)   \n', N_RUNS);
fprintf('  Max shift   : %.0f units per component (direction fixed)\n', MAX_SHIFT);
fprintf('  Success thr : f < %.1f                                  \n', SUCCESS_THRESH);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Fixed unit-direction vector for this dimension ─────────────────
    rng(DIR_SEED);
    raw_dir   = randn(1, D);
    direction = raw_dir / norm(raw_dir);   % unit vector

    fprintf('=========================================================\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('=========================================================\n\n');

    % Summary arrays: (N_ALG × N_SEV) for post-experiment plotting
    summ_mean = zeros(N_ALG, N_SEV);
    summ_std  = zeros(N_ALG, N_SEV);
    summ_sr   = zeros(N_ALG, N_SEV);   % success rate [0,1]
    summ_fet  = nan(N_ALG, N_SEV);     % median FE to target

    for si = 1:N_SEV
        sev       = SEVERITIES(si);
        sev_label = SEV_LABELS{si};
        mag       = sev * MAX_SHIFT;
        shift_vec = mag * direction;    % 1×D shift vector

        fprintf('  ── %s  (%.0f%% severity, ||shift||=%.2f) ─────────────────\n', ...
                sev_label, sev*100, norm(shift_vec));

        % Sanity check: f at true optimum should be ≈ 0
        f_check = schwefel_shifted(shift_vec, shift_vec, D);
        fprintf('  Sanity: f(x*=shift_vec) = %.6e  (should be ≈ 0)\n', f_check);
        if f_check > 1.0
            error('Exp08: sanity check failed at D=%d %s. Abort.', D, sev_label);
        end

        % Objective functions
        obj_scalar = @(x) schwefel_shifted(x, shift_vec, D);
        obj_loa    = @(x) deal(schwefel_shifted(x, shift_vec, D), 0);

        % Storage for this (D, severity)
        best_f_mat = zeros(N_ALG, N_RUNS);
        conv_cell  = cell(N_ALG, N_RUNS);
        fe_to_tgt  = nan(N_ALG, N_RUNS);   % FE at which threshold first crossed

        for ai = 1:N_ALG
            alg_name  = ALG{ai,1};
            wrap_fn   = str2func(ALG{ai,2});
            needs_loa = ALG{ai,3};
            obj_fn    = obj_scalar;
            if needs_loa, obj_fn = obj_loa; end

            fprintf('  [D=%2d | %s] Running %-8s ', D, sev_label, alg_name);
            t_start = tic;

            for r = 1:N_RUNS
                rng(r);
                try
                    [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE);
                catch ME
                    warning('Exp08: %s run %d D=%d %s failed: %s', ...
                            alg_name, r, D, sev_label, ME.message);
                    fbest = NaN;
                    conv  = NaN(1, floor(Max_FE/N_POP));
                end
                best_f_mat(ai,r) = fbest;
                conv_row         = conv(:)';
                conv_cell{ai,r}  = conv_row;

                % FE to target: find first iteration index where f < threshold
                % Each iteration ≈ N_POP FEs
                hit = find(conv_row < SUCCESS_THRESH, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(ai,r) = hit * N_POP;   % approximate FE count
                end
            end
            elapsed = toc(t_start);

            vals       = best_f_mat(ai,:);
            valid_vals = vals(isfinite(vals));
            n_success  = sum(valid_vals < SUCCESS_THRESH);
            sr         = n_success / N_RUNS;
            med_fet    = median(fe_to_tgt(ai,:), 'omitnan');

            if isempty(valid_vals)
                fprintf('→ ALL RUNS FAILED\n');
            else
                fprintf('→ mean=%10.3e | SR=%4.1f%% | FEt=%8s  [%.1fs]\n', ...
                        mean(valid_vals), sr*100, ...
                        format_fet(med_fet), elapsed);
            end
        end % algorithm loop

        % ----- Per-(D,severity) statistics --------------------------------
        stat_mean   = zeros(N_ALG,1); stat_std    = zeros(N_ALG,1);
        stat_best   = zeros(N_ALG,1); stat_worst  = zeros(N_ALG,1);
        stat_median = zeros(N_ALG,1); stat_sr     = zeros(N_ALG,1);
        stat_fet    = nan(N_ALG,1);

        for ai = 1:N_ALG
            vals       = best_f_mat(ai,:);
            valid_vals = vals(isfinite(vals));
            if isempty(valid_vals)
                stat_mean(ai)=NaN; stat_std(ai)=NaN; stat_best(ai)=NaN;
                stat_worst(ai)=NaN; stat_median(ai)=NaN;
            else
                stat_mean(ai)   = mean(valid_vals);
                stat_std(ai)    = std(valid_vals);
                stat_best(ai)   = min(valid_vals);
                stat_worst(ai)  = max(valid_vals);
                stat_median(ai) = median(valid_vals);
            end
            stat_sr(ai)  = sum(best_f_mat(ai,:) < SUCCESS_THRESH) / N_RUNS;
            stat_fet(ai) = median(fe_to_tgt(ai,:), 'omitnan');
        end

        % Store in summary arrays
        summ_mean(:,si) = stat_mean;
        summ_std(:,si)  = stat_std;
        summ_sr(:,si)   = stat_sr;
        summ_fet(:,si)  = stat_fet;

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
        fprintf('\n  Ranking (%s, by mean):\n', sev_label);
        for r = 1:N_ALG
            ai = rank_ord(r);
            fprintf('    %2d. %-8s mean=%10.3e  SR=%4.1f%%\n', ...
                    r, ALG{ai,1}, stat_mean(ai), stat_sr(ai)*100);
        end
        fprintf('\n');

        % Save per-(D,severity) .mat
        mat_path  = fullfile(RESULTS_DIR, ...
                    sprintf('Run_08_results_D%d_%s.mat', D, sev_label));
        alg_names = ALG(:,1);
        save(mat_path, 'D','sev','sev_label','mag','shift_vec', ...
             'N_RUNS','N_POP','Max_FE','SUCCESS_THRESH', ...
             'alg_names','best_f_mat','conv_cell','fe_to_tgt', ...
             'stat_mean','stat_std','stat_best','stat_worst', ...
             'stat_median','stat_sr','stat_fet','rank_ord');
        fprintf('  Saved: %s\n\n', mat_path);

        % Save per-(D,severity) .csv
        csv_path = fullfile(RESULTS_DIR, ...
                   sprintf('Run_08_results_D%d_%s.csv', D, sev_label));
        write_csv_severity(csv_path, ALG(:,1), stat_mean, stat_std, ...
                           stat_best, stat_worst, stat_median, stat_sr, stat_fet, ...
                           D, sev, mag, N_RUNS, Max_FE, SUCCESS_THRESH);
        fprintf('  Saved: %s\n\n', csv_path);

    end % severity loop

    % ----- Cross-severity summary for this D ─────────────────────────────
    fprintf('  ╔══════════════════════════════════════════════════════════╗\n');
    fprintf('  ║  SEVERITY SUMMARY — D=%d — Success Rate (%%)            \n', D);
    fprintf('  ╚══════════════════════════════════════════════════════════╝\n');
    fprintf('  %-8s', 'Algorithm');
    for si = 1:N_SEV
        fprintf(' %6s', SEV_LABELS{si});
    end
    fprintf('  | Trend\n');
    fprintf('  %s\n', repmat('-',1,70));
    for ai = 1:N_ALG
        fprintf('  %-8s', ALG{ai,1});
        for si = 1:N_SEV
            fprintf(' %5.1f%%', summ_sr(ai,si)*100);
        end
        % Trend: SR at S5 vs S0
        trend_str = '→';
        if summ_sr(ai,N_SEV) > summ_sr(ai,1) + 0.1
            trend_str = '↑';
        elseif summ_sr(ai,N_SEV) < summ_sr(ai,1) - 0.1
            trend_str = '↓';
        end
        fprintf('  | %s\n', trend_str);
    end
    fprintf('\n');

    % Overall robustness: algorithms that maintain SR across all severities
    sr_drop = summ_sr(:,1) - summ_sr(:,N_SEV);   % drop from S0 to S5
    [sr_drop_sorted, drop_ord] = sort(sr_drop);
    fprintf('  Robustness ranking (smallest SR drop S0→S5 = most robust):\n');
    for r = 1:N_ALG
        ai = drop_ord(r);
        fprintf('    %2d. %-8s  SR drop = %+.1f%%  (S0=%.1f%% → S5=%.1f%%)\n', ...
                r, ALG{ai,1}, -sr_drop_sorted(r)*100, ...
                summ_sr(ai,1)*100, summ_sr(ai,N_SEV)*100);
    end
    fprintf('\n');

    % Save summary .mat
    summ_mat = fullfile(RESULTS_DIR, ...
               sprintf('Run_08_severity_summary_D%d.mat', D));
    alg_names = ALG(:,1);
    save(summ_mat, 'D','SEVERITIES','SEV_LABELS','N_RUNS','N_POP', ...
         'MAX_SHIFT','direction','SUCCESS_THRESH', ...
         'alg_names','summ_mean','summ_std','summ_sr','summ_fet');
    fprintf('  Saved summary: %s\n\n', summ_mat);

    % Save summary .csv
    summ_csv = fullfile(RESULTS_DIR, ...
               sprintf('Run_08_severity_summary_D%d.csv', D));
    write_summary_csv(summ_csv, ALG(:,1), summ_mean, summ_sr, summ_fet, ...
                      SEVERITIES, SEV_LABELS, D, N_RUNS, MAX_SHIFT, SUCCESS_THRESH);
    fprintf('  Saved summary: %s\n\n', summ_csv);

end % dimension loop

fprintf('=========================================================\n');
fprintf('  Experiment 8 complete.\n');
fprintf('  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% SCHWEFEL_SHIFTED  Shifted Schwefel (f* ≈ 0 at x = shift_vec).
%   y = x - shift_vec + 420.9687 → ỹ = clamp(y,-500,500)
%   f = 418.9829*d - sum(ỹ.*sin(sqrt(|ỹ|)))
%   Identical formula to Experiment 3.
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function s = format_fet(fet)
% Format FE-to-target for display: NaN → 'N/A', else integer string.
    if isnan(fet)
        s = 'N/A';
    else
        s = sprintf('%d', round(fet));
    end
end


function write_csv_severity(csv_path, alg_names, ...
                             s_mean, s_std, s_best, s_worst, s_median, ...
                             s_sr, s_fet, D, sev, mag, N_RUNS, Max_FE, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp08: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_08_Progressive_Shift_Severity\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Severity_fraction,%.2f\n', sev);
    fprintf(fid, 'Shift_magnitude,%.4f\n', mag);
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
                            SEVERITIES, SEV_LABELS, D, N_RUNS, MAX_SHIFT, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp08: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_08_Severity_Summary\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Max_shift,%.1f\n', MAX_SHIFT);
    fprintf(fid, 'Success_threshold,%.1f\n', thresh);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, '\n');

    % Header
    fprintf(fid, 'Algorithm');
    for si = 1:numel(SEVERITIES)
        fprintf(fid, ',%s_Mean,%s_SR,%s_FEt', ...
                SEV_LABELS{si}, SEV_LABELS{si}, SEV_LABELS{si});
    end
    fprintf(fid, '\n');

    % Data
    for i = 1:numel(alg_names)
        fprintf(fid, '%s', alg_names{i});
        for si = 1:numel(SEVERITIES)
            fet_str = 'NaN';
            if ~isnan(summ_fet(i,si))
                fet_str = sprintf('%d', round(summ_fet(i,si)));
            end
            fprintf(fid, ',%.6e,%.4f,%s', ...
                    summ_mean(i,si), summ_sr(i,si), fet_str);
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
