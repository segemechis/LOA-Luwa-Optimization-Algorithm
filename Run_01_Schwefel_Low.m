% =========================================================================
%  Run_01_Schwefel_Low.m
%  Experiment 1 — Standard Schwefel Function: Low Dimensionality
%  Dimensions tested : D = 10  and  D = 30
%  Algorithms        : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA,
%                          L-SHADE, MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Independent runs  : 30
%  FE budget         : D × 10,000  per run
%  Population size   : 30 agents   (N_g = 30 for LOA → 150 total)
%  Bounds            : [-500, 500]^D
%  Global optimum    : f* ≈ 0  (shifted to 0; raw Schwefel ≈ 418.9829·D)
%
%  Outputs (saved to C:\LOA — Luwa Optimization Algorithm\Results\Exp01\):
%    Run_01_results_D<d>.mat  — full workspace per dimension
%    Run_01_results_D<d>.csv  — summary statistics table
%    Run_01_log.txt           — console mirror (all printed output)
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp01';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

% Mirror all fprintf output to a log file
log_path = fullfile(RESULTS_DIR, 'Run_01_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST  = [10, 30];     % dimensions to test
N_RUNS    = 30;           % independent runs per algorithm per dimension
N_POP     = 30;           % population size  (N_g per grade for LOA)
LB_SCALAR = -500;         % Schwefel search-space lower bound (scalar)
UB_SCALAR =  500;         % Schwefel search-space upper bound (scalar)

%% ── 2. ALGORITHM REGISTRY ────────────────────────────────────────────────
% Each row: { display_name,  wrapper_function_handle_name,  needs_loa_obj }
%   needs_loa_obj = true  → objective must return [f, cv]  (LOA only)
%   needs_loa_obj = false → objective returns scalar f
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

%% ── 3. SCHWEFEL FUNCTION DEFINITIONS ────────────────────────────────────
%  Standard Schwefel (minimisation form, global minimum = 0):
%    f(x) = 418.9829 * d - sum_i( x_i * sin(sqrt(|x_i|)) )
%
%  obj_scalar(x) returns a single scalar  (used by all algorithms except LOA)
%  obj_loa(x)    returns [f, cv=0]        (used by LOA via wrap_LOA)
%
%  Both are created fresh inside the dimension loop so that 'd' is captured
%  correctly in the closure for each dimension.

%% ── 4. MAIN LOOP ─────────────────────────────────────────────────────────
fprintf('=========================================================\n');
fprintf('  Experiment 1 — Standard Schwefel (Low Dimensionality)  \n');
fprintf('  Dimensions : D = 10, 30                                \n');
fprintf('  Runs       : %d per algorithm per dimension            \n', N_RUNS);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;        % CEC-standard FE budget
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ----- Objective functions for this dimension -----------------------
    obj_scalar = @(x) schwefel_standard(x, D);
    obj_loa    = @(x) deal(schwefel_standard(x, D), 0);  % [f, cv=0]

    fprintf('─────────────────────────────────────────────────────\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('─────────────────────────────────────────────────────\n');

    % ----- Storage matrices for this dimension -------------------------
    % best_f_mat(alg, run) — best fitness of each run
    best_f_mat  = zeros(N_ALG, N_RUNS);
    % conv_cell{alg, run}  — convergence curve (row vector)
    conv_cell   = cell(N_ALG, N_RUNS);

    % ----- Run all algorithms ------------------------------------------
    for ai = 1:N_ALG
        alg_name   = ALG{ai, 1};
        wrap_name  = ALG{ai, 2};
        needs_loa  = ALG{ai, 3};
        wrap_fn    = str2func(wrap_name);

        % Select objective
        if needs_loa
            obj_fn = obj_loa;
        else
            obj_fn = obj_scalar;
        end

        fprintf('  [D=%3d] Running %-8s ', D, alg_name);

        t_start = tic;
        for r = 1:N_RUNS
            rng(r);   % deterministic seed per run for reproducibility
            try
                [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE);
            catch ME
                warning('Run_01: %s run %d D=%d failed: %s', ...
                        alg_name, r, D, ME.message);
                fbest = NaN;
                conv  = NaN(1, floor(Max_FE / N_POP));
            end
            best_f_mat(ai, r)  = fbest;
            conv_cell{ai, r}   = conv(:)';   % ensure row vector
        end
        elapsed = toc(t_start);

        % Per-algorithm summary line
        vals        = best_f_mat(ai, :);
        valid_vals  = vals(isfinite(vals));
        if isempty(valid_vals)
            fprintf('→ ALL RUNS FAILED\n');
        else
            fprintf('→ mean = %12.4e | std = %10.4e | best = %10.4e | worst = %10.4e  [%.1fs]\n', ...
                    mean(valid_vals), std(valid_vals), ...
                    min(valid_vals),  max(valid_vals), elapsed);
        end
    end

    % ----- Statistical summary table -----------------------------------
    fprintf('\n  %-8s  %12s  %12s  %12s  %12s  %12s\n', ...
            'Algorithm','Mean','Std','Best','Worst','Median');
    fprintf('  %s\n', repmat('-', 1, 74));

    stat_mean   = zeros(N_ALG, 1);
    stat_std    = zeros(N_ALG, 1);
    stat_best   = zeros(N_ALG, 1);
    stat_worst  = zeros(N_ALG, 1);
    stat_median = zeros(N_ALG, 1);

    for ai = 1:N_ALG
        vals = best_f_mat(ai, :);
        valid_vals = vals(isfinite(vals));
        if isempty(valid_vals)
            stat_mean(ai)   = NaN; stat_std(ai)  = NaN;
            stat_best(ai)   = NaN; stat_worst(ai)= NaN;
            stat_median(ai) = NaN;
        else
            stat_mean(ai)   = mean(valid_vals);
            stat_std(ai)    = std(valid_vals);
            stat_best(ai)   = min(valid_vals);
            stat_worst(ai)  = max(valid_vals);
            stat_median(ai) = median(valid_vals);
        end
        fprintf('  %-8s  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n', ...
                ALG{ai,1}, stat_mean(ai), stat_std(ai), ...
                stat_best(ai), stat_worst(ai), stat_median(ai));
    end
    fprintf('\n');

    % ----- Ranking by mean --------------------------------------------
    [~, rank_order] = sort(stat_mean);
    fprintf('  Ranking (by mean fitness, lower is better):\n');
    for r = 1:N_ALG
        ai = rank_order(r);
        fprintf('    %2d. %-8s  mean = %12.4e\n', r, ALG{ai,1}, stat_mean(ai));
    end
    fprintf('\n');

    % ----- Save .mat --------------------------------------------------
    mat_path = fullfile(RESULTS_DIR, sprintf('Run_01_results_D%d.mat', D));
    alg_names    = ALG(:, 1);
    save(mat_path, ...
         'D', 'N_RUNS', 'N_POP', 'Max_FE', 'LB_SCALAR', 'UB_SCALAR', ...
         'alg_names', 'best_f_mat', 'conv_cell', ...
         'stat_mean', 'stat_std', 'stat_best', 'stat_worst', 'stat_median', ...
         'rank_order');
    fprintf('  Saved: %s\n\n', mat_path);

    % ----- Save .csv --------------------------------------------------
    csv_path = fullfile(RESULTS_DIR, sprintf('Run_01_results_D%d.csv', D));
    write_csv_summary(csv_path, ALG(:,1), ...
                      stat_mean, stat_std, stat_best, stat_worst, stat_median, ...
                      D, N_RUNS, Max_FE);
    fprintf('  Saved: %s\n\n', csv_path);
end

fprintf('=========================================================\n');
fprintf('  Experiment 1 complete.\n');
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
%
%   Input  : x — 1×d row vector
%   Output : f — scalar fitness value
    f = 418.9829 * d - sum(x .* sin(sqrt(abs(x))));
end


function write_csv_summary(csv_path, alg_names, ...
                            s_mean, s_std, s_best, s_worst, s_median, ...
                            D, N_RUNS, Max_FE)
% WRITE_CSV_SUMMARY  Write per-algorithm statistics to a CSV file.
    fid = fopen(csv_path, 'w');
    if fid == -1
        warning('Run_01: cannot open %s for writing', csv_path);
        return;
    end

    % Header block
    fprintf(fid, 'Experiment,Run_01_Schwefel_Low\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, 'FE_Budget,%d\n', Max_FE);
    fprintf(fid, 'Bounds,[-500; 500]\n');
    fprintf(fid, 'Global_Optimum,0\n');
    fprintf(fid, '\n');

    % Column headers
    fprintf(fid, 'Algorithm,Mean,Std,Best,Worst,Median\n');

    n = numel(alg_names);
    for i = 1:n
        fprintf(fid, '%s,%.6e,%.6e,%.6e,%.6e,%.6e\n', ...
                alg_names{i}, s_mean(i), s_std(i), ...
                s_best(i), s_worst(i), s_median(i));
    end

    fclose(fid);
end
