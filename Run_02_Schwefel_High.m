% =========================================================================
%  Run_02_Schwefel_High.m
%  Experiment 2 — Standard Schwefel Function: High Dimensionality
%  Dimensions tested : D = 50  and  D = 100
%  Algorithms        : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA,
%                          L-SHADE, MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Independent runs  : 30
%  FE budget         : D × 10,000  per run
%  Population size   : 30 agents   (N_g = 6 per grade for LOA → 30 total)
%  Bounds            : [-500, 500]^D
%  Global optimum    : f* = 0
%
%  Outputs (saved to C:\LOA — Luwa Optimization Algorithm\Results\Exp02\):
%    Run_02_results_D<d>.mat  — full workspace per dimension
%    Run_02_results_D<d>.csv  — summary statistics table
%    Run_02_log.txt           — console mirror (all printed output)
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp02';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_02_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST  = [50, 100];    % dimensions to test
N_RUNS    = 30;
N_POP     = 30;
LB_SCALAR = -500;
UB_SCALAR =  500;

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
fprintf('  Experiment 2 — Standard Schwefel (High Dimensionality) \n');
fprintf('  Dimensions : D = 50, 100                               \n');
fprintf('  Runs       : %d per algorithm per dimension            \n', N_RUNS);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    obj_scalar = @(x) schwefel_standard(x, D);
    obj_loa    = @(x) deal(schwefel_standard(x, D), 0);

    fprintf('─────────────────────────────────────────────────────\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('─────────────────────────────────────────────────────\n');

    best_f_mat = zeros(N_ALG, N_RUNS);
    conv_cell  = cell(N_ALG, N_RUNS);

    for ai = 1:N_ALG
        alg_name  = ALG{ai, 1};
        wrap_name = ALG{ai, 2};
        needs_loa = ALG{ai, 3};
        wrap_fn   = str2func(wrap_name);
        obj_fn    = obj_scalar;
        if needs_loa, obj_fn = obj_loa; end

        fprintf('  [D=%3d] Running %-8s ', D, alg_name);
        t_start = tic;

        for r = 1:N_RUNS
            rng(r);
            try
                [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE);
            catch ME
                warning('Run_02: %s run %d D=%d failed: %s', ...
                        alg_name, r, D, ME.message);
                fbest = NaN;
                conv  = NaN(1, floor(Max_FE / N_POP));
            end
            best_f_mat(ai, r) = fbest;
            conv_cell{ai, r}  = conv(:)';
        end
        elapsed = toc(t_start);

        vals       = best_f_mat(ai, :);
        valid_vals = vals(isfinite(vals));
        if isempty(valid_vals)
            fprintf('→ ALL RUNS FAILED\n');
        else
            fprintf('→ mean = %12.4e | std = %10.4e | best = %10.4e | worst = %10.4e  [%.1fs]\n', ...
                    mean(valid_vals), std(valid_vals), ...
                    min(valid_vals),  max(valid_vals), elapsed);
        end
    end

    % ----- Statistics ---------------------------------------------------
    stat_mean   = zeros(N_ALG, 1);
    stat_std    = zeros(N_ALG, 1);
    stat_best   = zeros(N_ALG, 1);
    stat_worst  = zeros(N_ALG, 1);
    stat_median = zeros(N_ALG, 1);

    for ai = 1:N_ALG
        vals       = best_f_mat(ai, :);
        valid_vals = vals(isfinite(vals));
        if isempty(valid_vals)
            stat_mean(ai) = NaN; stat_std(ai)    = NaN;
            stat_best(ai) = NaN; stat_worst(ai)  = NaN;
            stat_median(ai) = NaN;
        else
            stat_mean(ai)   = mean(valid_vals);
            stat_std(ai)    = std(valid_vals);
            stat_best(ai)   = min(valid_vals);
            stat_worst(ai)  = max(valid_vals);
            stat_median(ai) = median(valid_vals);
        end
    end

    fprintf('\n  %-8s  %12s  %12s  %12s  %12s  %12s\n', ...
            'Algorithm','Mean','Std','Best','Worst','Median');
    fprintf('  %s\n', repmat('-', 1, 74));
    for ai = 1:N_ALG
        fprintf('  %-8s  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n', ...
                ALG{ai,1}, stat_mean(ai), stat_std(ai), ...
                stat_best(ai), stat_worst(ai), stat_median(ai));
    end
    fprintf('\n');

    [~, rank_order] = sort(stat_mean);
    fprintf('  Ranking (by mean fitness, lower is better):\n');
    for r = 1:N_ALG
        ai = rank_order(r);
        fprintf('    %2d. %-8s  mean = %12.4e\n', r, ALG{ai,1}, stat_mean(ai));
    end
    fprintf('\n');

    % ----- Save .mat ----------------------------------------------------
    mat_path  = fullfile(RESULTS_DIR, sprintf('Run_02_results_D%d.mat', D));
    alg_names = ALG(:, 1);
    save(mat_path, ...
         'D', 'N_RUNS', 'N_POP', 'Max_FE', 'LB_SCALAR', 'UB_SCALAR', ...
         'alg_names', 'best_f_mat', 'conv_cell', ...
         'stat_mean', 'stat_std', 'stat_best', 'stat_worst', 'stat_median', ...
         'rank_order');
    fprintf('  Saved: %s\n\n', mat_path);

    % ----- Save .csv ----------------------------------------------------
    csv_path = fullfile(RESULTS_DIR, sprintf('Run_02_results_D%d.csv', D));
    write_csv_summary(csv_path, ALG(:,1), ...
                      stat_mean, stat_std, stat_best, stat_worst, stat_median, ...
                      D, N_RUNS, Max_FE);
    fprintf('  Saved: %s\n\n', csv_path);
end

fprintf('=========================================================\n');
fprintf('  Experiment 2 complete.\n');
fprintf('  Results folder: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_standard(x, d)
% SCHWEFEL_STANDARD  Standard Schwefel function (minimisation, f* = 0).
%
%   f(x) = 418.9829 * d  -  sum_i( x_i * sin(sqrt(|x_i|)) )
%
%   Global minimum: f* = 0 at x* = [420.9687, ..., 420.9687]
%   Search domain : x_i ∈ [-500, 500]
    f = 418.9829 * d - sum(x .* sin(sqrt(abs(x))));
end


function write_csv_summary(csv_path, alg_names, ...
                            s_mean, s_std, s_best, s_worst, s_median, ...
                            D, N_RUNS, Max_FE)
    fid = fopen(csv_path, 'w');
    if fid == -1
        warning('Run_02: cannot open %s for writing', csv_path);
        return;
    end
    fprintf(fid, 'Experiment,Run_02_Schwefel_High\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, 'FE_Budget,%d\n', Max_FE);
    fprintf(fid, 'Bounds,[-500; 500]\n');
    fprintf(fid, 'Global_Optimum,0\n');
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm,Mean,Std,Best,Worst,Median\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s,%.6e,%.6e,%.6e,%.6e,%.6e\n', ...
                alg_names{i}, s_mean(i), s_std(i), ...
                s_best(i), s_worst(i), s_median(i));
    end
    fclose(fid);
end
