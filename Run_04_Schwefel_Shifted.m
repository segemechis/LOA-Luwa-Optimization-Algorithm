% =========================================================================
%  Run_04_Schwefel_Shifted.m
%  Experiment 4 — Shifted Schwefel Function
%
%  Motivation:
%    The standard Schwefel function has its global optimum at exactly
%    x* = [420.9687, ..., 420.9687] — the same value in every dimension.
%    Critics argue that some algorithms exploit this symmetric structure
%    rather than genuinely navigating the landscape (e.g. GA's SBX
%    crossover finds the per-dimension optimum value independently and
%    recombines, giving it an artificial advantage on the standard form).
%    This experiment eliminates that artifact by shifting the optimum to
%    a completely random vector drawn once per dimension and fixed across
%    all 30 runs and all 15 algorithms.
%
%  Shifted Schwefel construction (CORRECT formulation):
%    Let o (shift_vec) be drawn once: o ~ U[-200, 200]^D
%    Define: y = x - o + 420.9687
%    f_shifted(x) = 418.9829*D - sum_i( ỹ_i * sin(sqrt(|ỹ_i|)) )
%    where ỹ = clamp(y, -500, 500)
%
%    When x = o: y = 420.9687  for every dim → f* = 0. ✓
%    The landscape shape is IDENTICAL to standard Schwefel — only the
%    location of the global optimum moves from 420.9687 to shift_vec.
%
%    Shift range ±200 (conservative): ensures y stays comfortably inside
%    [-500, 500] for most of the search space even before clamping.
%    Clamping ỹ handles edge cases near the search boundary.
%
%  Why the original y = x - shift_vec formula was wrong:
%    That formula sets f*=0 at y=0, i.e. at x=shift_vec in y-space,
%    but the raw Schwefel's minimum is at y=420.9687, not y=0.
%    Result: f(shift_vec) = 418.9829*D ≠ 0, and algorithms that found
%    x≈420.9687 (the standard optimum in x-space) got y ≈ 420.9687-shift,
%    which pushed the Schwefel sum above 418.9829*D, giving negative fitness.
%
%  Dimensions tested : D = 10, 30, 50
%  Algorithms        : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA,
%                          L-SHADE, MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Independent runs  : 30
%  FE budget         : D × 10,000  per run
%  Population size   : 30 agents   (N_g = 6 per grade for LOA → 30 total)
%  Bounds            : x_i ∈ [-500, 500]^D
%
%  Outputs (saved to C:\LOA — Luwa Optimization Algorithm\Results\Exp04\):
%    Run_04_results_D<d>.mat  — full workspace including shift vector
%    Run_04_results_D<d>.csv  — summary statistics table
%    Run_04_log.txt           — console mirror (all printed output)
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp04';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_04_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST   = [10, 30, 50];    % D=100 excluded per experimental plan
N_RUNS     = 30;
N_POP      = 30;
LB_SCALAR  = -500;
UB_SCALAR  =  500;
SHIFT_SEED = 9999;

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
fprintf('  Experiment 4 — Shifted Schwefel Function               \n');
fprintf('  Dimensions : D = 10, 30, 50                            \n');
fprintf('  Runs       : %d per algorithm per dimension             \n', N_RUNS);
fprintf('  Shift seed : %d (fixed, same for all algorithms)        \n', SHIFT_SEED);
fprintf('  Formula    : y = x - shift + 420.9687, f* = 0 at x=shift\n');
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Generate shift vector ────────────────────────────────────────────
    % Range ±200: conservative choice so y = x - shift + 420.9687 stays
    % close to [-500,500] for most x, with clamping handling edge cases.
    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);   % o ~ U[-200, 200]^D

    fprintf('─────────────────────────────────────────────────────\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('  Shift vector (first 5 dims): [');
    fprintf(' %.3f', shift_vec(1:min(5,D)));
    fprintf(' ...]\n');
    fprintf('  True optimum x* = shift_vec  (f* = 0)\n');
    fprintf('─────────────────────────────────────────────────────\n');

    % ── Objective functions ──────────────────────────────────────────────
    obj_scalar = @(x) schwefel_shifted(x, shift_vec, D);
    obj_loa    = @(x) deal(schwefel_shifted(x, shift_vec, D), 0);

    % Sanity check: f at true optimum must equal ≈ 0
    f_check = obj_scalar(shift_vec);
    fprintf('  Sanity check — f(shift_vec) = %.6e  (should be ≈ 0)\n', f_check);
    if f_check > 1.0
        error('Run_04: Sanity check FAILED at D=%d. f(shift_vec)=%.4e. Abort.', D, f_check);
    end

    % Additional check: f at a random point should be large positive
    rng(12345);
    f_random = obj_scalar(lb + rand(1,D).*(ub-lb));
    fprintf('  Sanity check — f(random x)  = %.6e  (should be >> 0)\n\n', f_random);

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
                warning('Run_04: %s run %d D=%d failed: %s', ...
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
            stat_mean(ai)   = NaN; stat_std(ai)   = NaN;
            stat_best(ai)   = NaN; stat_worst(ai) = NaN;
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
    fprintf('  %s\n', repmat('-',1,74));
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
    mat_path  = fullfile(RESULTS_DIR, sprintf('Run_04_results_D%d.mat', D));
    alg_names = ALG(:,1);
    save(mat_path, ...
         'D', 'N_RUNS', 'N_POP', 'Max_FE', 'LB_SCALAR', 'UB_SCALAR', ...
         'shift_vec', 'SHIFT_SEED', ...
         'alg_names', 'best_f_mat', 'conv_cell', ...
         'stat_mean', 'stat_std', 'stat_best', 'stat_worst', 'stat_median', ...
         'rank_order');
    fprintf('  Saved: %s\n\n', mat_path);

    % ----- Save .csv ----------------------------------------------------
    csv_path = fullfile(RESULTS_DIR, sprintf('Run_04_results_D%d.csv', D));
    write_csv_summary(csv_path, ALG(:,1), ...
                      stat_mean, stat_std, stat_best, stat_worst, stat_median, ...
                      D, N_RUNS, Max_FE, shift_vec);
    fprintf('  Saved: %s\n\n', csv_path);
end

fprintf('=========================================================\n');
fprintf('  Experiment 4 complete.\n');
fprintf('  Results folder: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% SCHWEFEL_SHIFTED  Shifted Schwefel (minimisation, f* = 0 at x = shift_vec).
%
%  Correct CEC-style shift formulation:
%    y  = x - shift_vec + 420.9687
%    ỹ  = clamp(y, -500, 500)
%    f  = 418.9829 * d  -  sum_i( ỹ_i * sin(sqrt(|ỹ_i|)) )
%
%  When x = shift_vec: y = 420.9687 (global min of raw Schwefel) → f = 0. ✓
%  The landscape shape is identical to standard Schwefel in every respect.
%  Clamping handles boundary cases when x is far from the optimum region.
%
%  Common mistake (WRONG): y = x - shift_vec
%    That makes f* = 0 at y = 0 (x = shift_vec in y-space), but the raw
%    Schwefel minimum is at y = 420.9687, not y = 0. So f(shift_vec) ≠ 0,
%    and algorithms near x = 420.9687 get negative fitness values.
%
%  Inputs:
%    x         — 1×d row vector
%    shift_vec — 1×d row vector, o ~ U[-200, 200]^D
%    d         — problem dimension
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));        % clamp to Schwefel's domain
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function write_csv_summary(csv_path, alg_names, ...
                            s_mean, s_std, s_best, s_worst, s_median, ...
                            D, N_RUNS, Max_FE, shift_vec)
    fid = fopen(csv_path, 'w');
    if fid == -1
        warning('Run_04: cannot open %s for writing', csv_path);
        return;
    end
    fprintf(fid, 'Experiment,Run_04_Schwefel_Shifted\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, 'FE_Budget,%d\n', Max_FE);
    fprintf(fid, 'Bounds,[-500; 500]\n');
    fprintf(fid, 'Formula,y = x - shift + 420.9687  then clamp to [-500,500]\n');
    fprintf(fid, 'Global_Optimum,0  (at x = shift_vec)\n');
    fprintf(fid, 'Shift_range,U[-200;200]\n');
    fprintf(fid, 'Shift_seed,%d+D\n', 9999);
    n_show = min(10, D);
    sv_str = sprintf('%.4f ', shift_vec(1:n_show));
    if D > 10, sv_str = [sv_str, '...']; end
    fprintf(fid, 'Shift_vector_first10,[%s]\n', strtrim(sv_str));
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm,Mean,Std,Best,Worst,Median\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s,%.6e,%.6e,%.6e,%.6e,%.6e\n', ...
                alg_names{i}, s_mean(i), s_std(i), ...
                s_best(i), s_worst(i), s_median(i));
    end
    fclose(fid);
end
