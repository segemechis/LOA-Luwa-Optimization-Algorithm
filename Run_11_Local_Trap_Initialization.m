% =========================================================================
%  Run_11_Local_Trap_Initialization.m
%  Experiment 11 — Local-Trap Initialization
%
%  Motivation:
%    This is the most direct test of the paper's central hypothesis:
%    "Can an algorithm escape a spatially separated local optimum trap
%    when the global optimum is far away?"
%    The entire population is deliberately initialised inside the nearest
%    inferior local basin of the shifted Schwefel function. The algorithm
%    must detect that it is trapped and self-organise an escape — without
%    any external help — across a large spatial gap to reach the true
%    global optimum.
%
%  Trap construction:
%    The shifted Schwefel has its global optimum at x* = shift_vec.
%    Its nearest inferior local minimum lies at:
%      x_trap(i) = shift_vec(i) - 302.5232   for all i
%    (derived from the standard Schwefel local min at y_i ≈ -302.52,
%     mapped back through the shift formula y = x - shift + 420.9687)
%    x_trap is clipped to [-500, 500]^D for boundary safety.
%
%    Fitness gap between trap and global:
%      Δf = f(x_trap) - f(x*) ≈ 118.44 * D
%      D=30: Δf ≈ 16100  |  D=50: Δf ≈ 26834
%    Spatial gap: ||x* - x_trap|| = 302.52 * sqrt(D)
%      D=30: ≈ 1657  |  D=50: ≈ 2139  (Euclidean, in 1000-unit search space)
%
%  Three trap configurations (progressively harder):
%    T1  Tight trap   half_width = 25   very tightly clustered
%    T2  Standard     half_width = 50   one Schwefel basin width
%    T3  Wide trap    half_width = 150  agents spread across local basin
%
%    lb_init = clip(x_trap - half_width, -500, 500)
%    ub_init = clip(x_trap + half_width, -500, 500)
%    Search space: [-500, 500]^D (always full — escape is geometrically possible)
%
%  Primary research question:
%    Can LOA reliably escape the local trap (SR > 0) while competitors
%    stagnate? How does escape probability change with trap tightness?
%
%  Metrics: Mean, Std, Best, Worst, Median, SR (f < 1.0), FE-to-target
%
%  Dimensions : D = 30, 50
%  Algorithms : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA, L-SHADE,
%                   MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Runs       : 30 per (algorithm, trap config, dimension)
%  FE budget  : D × 10,000 per run
%  Population : 30 agents
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Exp11\):
%    Run_11_results_D<d>_<trap>.mat   — workspace per (D, trap config)
%    Run_11_results_D<d>_<trap>.csv   — statistics
%    Run_11_trap_summary_D<d>.mat     — cross-trap summary arrays
%    Run_11_trap_summary_D<d>.csv     — config × algorithm table
%    Run_11_log.txt                   — console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp11';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_11_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST       = [30, 50];
SHIFT_SEED     = 9999;        % same landscape as Exp 3 / Exp 10
LOCAL_MIN_DIST = 302.5232;    % Schwefel nearest local min offset from global
TRAP_LABELS    = {'T1','T2','T3'};
TRAP_NAMES     = {'Tight (hw=25)','Standard (hw=50)','Wide (hw=150)'};
TRAP_HALFWIDTHS = [25, 50, 150];
N_TRAPS        = numel(TRAP_LABELS);
N_RUNS         = 30;
N_POP          = 30;
LB_SCALAR      = -500;
UB_SCALAR      =  500;
SUCCESS_THRESH = 1.0;         % f < 1.0 = escaped trap, found global opt

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
fprintf('  Experiment 11 — Local-Trap Initialization              \n');
fprintf('  Dimensions  : D = 30, 50                               \n');
fprintf('  Trap configs: T1(hw=25) T2(hw=50) T3(hw=150)          \n');
fprintf('  Runs        : %d per (algorithm, trap, dimension)       \n', N_RUNS);
fprintf('  Trap centre : x_trap = shift_vec - 302.5232            \n');
fprintf('  Search space: [-500,500]^D  (escape is possible)       \n');
fprintf('  Success thr : f < %.1f  (global basin reached)          \n', SUCCESS_THRESH);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Shifted Schwefel landscape (same as Exp 3 / Exp 10) ──────────────
    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);
    x_opt     = shift_vec;

    % ── Trap centre: nearest inferior local minimum ───────────────────────
    x_trap      = shift_vec - LOCAL_MIN_DIST;
    x_trap      = max(LB_SCALAR, min(UB_SCALAR, x_trap));   % clip to bounds
    n_oob       = sum(abs(shift_vec - LOCAL_MIN_DIST) > UB_SCALAR | ...
                      abs(shift_vec - LOCAL_MIN_DIST) < LB_SCALAR);

    spatial_gap = norm(x_opt - x_trap);
    f_trap      = schwefel_shifted(x_trap, shift_vec, D);
    f_opt       = schwefel_shifted(x_opt,  shift_vec, D);
    fitness_gap = f_trap - f_opt;

    fprintf('=========================================================\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('  shift_vec (first 5): ['); fprintf(' %.3f', shift_vec(1:min(5,D))); fprintf(' ...]\n');
    fprintf('  x_opt     (first 5): ['); fprintf(' %.3f', x_opt(1:min(5,D)));     fprintf(' ...]\n');
    fprintf('  x_trap    (first 5): ['); fprintf(' %.3f', x_trap(1:min(5,D)));    fprintf(' ...]\n');
    fprintf('  Boundary clips: %d/%d dims\n', n_oob, D);
    fprintf('  Spatial gap ||x_opt - x_trap|| = %.2f\n', spatial_gap);
    fprintf('  Fitness gap f(x_trap) - f(x_opt) = %.2f\n', fitness_gap);
    fprintf('=========================================================\n\n');

    % Sanity checks
    if f_opt > 1.0
        error('Exp11: f(x_opt)=%.4e at D=%d. Check shift formula.', f_opt, D);
    end
    fprintf('  Sanity: f(x_opt)  = %.6e  (should be ≈ 0)  ✓\n', f_opt);
    fprintf('  Sanity: f(x_trap) = %.4e  (should be >> 0) ✓\n\n', f_trap);

    % Objectives (fixed for all trap configs this D)
    obj_scalar = @(x) schwefel_shifted(x, shift_vec, D);
    obj_loa    = @(x) deal(schwefel_shifted(x, shift_vec, D), 0);

    % Summary arrays: (N_ALG × N_TRAPS)
    summ_mean = zeros(N_ALG, N_TRAPS);
    summ_std  = zeros(N_ALG, N_TRAPS);
    summ_sr   = zeros(N_ALG, N_TRAPS);
    summ_fet  = nan(N_ALG, N_TRAPS);

    for ti = 1:N_TRAPS
        trap_label    = TRAP_LABELS{ti};
        trap_name     = TRAP_NAMES{ti};
        half_w        = TRAP_HALFWIDTHS(ti);

        % Build init box around trap centre
        lb_init = max(LB_SCALAR, x_trap - half_w);
        ub_init = min(UB_SCALAR, x_trap + half_w);
        box_vol = prod(ub_init - lb_init);   % proportional to init volume

        fprintf('  ── %s (%s) ───────────────────────────────────────────\n', ...
                trap_label, trap_name);
        fprintf('  lb_init(1:3)=[%.1f %.1f %.1f...]  ub_init(1:3)=[%.1f %.1f %.1f...]\n', ...
                lb_init(1),lb_init(2),lb_init(3), ...
                ub_init(1),ub_init(2),ub_init(3));
        fprintf('  Min box width: %.1f\n\n', min(ub_init - lb_init));

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

            fprintf('  [D=%2d | %s] Running %-8s ', D, trap_label, alg_name);
            t_start = tic;

            for r = 1:N_RUNS
                rng(r);
                try
                    [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE, ...
                                               lb_init, ub_init);
                catch ME
                    warning('Exp11: %s run %d D=%d %s failed: %s', ...
                            alg_name, r, D, trap_label, ME.message);
                    fbest = NaN;
                    conv  = NaN(1, floor(Max_FE/N_POP));
                end
                best_f_mat(ai,r) = fbest;
                conv_row = conv(:)';
                conv_cell{ai,r} = conv_row;

                % FE-to-target: first iteration where conv < SUCCESS_THRESH
                hit = find(conv_row < SUCCESS_THRESH, 1, 'first');
                if ~isempty(hit)
                    fe_to_tgt(ai,r) = hit * N_POP;
                end
            end
            elapsed = toc(t_start);

            vals    = best_f_mat(ai,:);
            valid_v = vals(isfinite(vals));
            sr      = sum(valid_v < SUCCESS_THRESH) / N_RUNS;
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
            stat_sr(ai)  = sum(best_f_mat(ai,:) < SUCCESS_THRESH) / N_RUNS;
            stat_fet(ai) = median(fe_to_tgt(ai,:), 'omitnan');
        end

        summ_mean(:,ti) = stat_mean;
        summ_std(:,ti)  = stat_std;
        summ_sr(:,ti)   = stat_sr;
        summ_fet(:,ti)  = stat_fet;

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
        fprintf('\n  Ranking (%s, by mean):\n', trap_label);
        for r = 1:N_ALG
            ai = rank_ord(r);
            fprintf('    %2d. %-8s mean=%10.3e  SR=%4.1f%%\n', ...
                    r, ALG{ai,1}, stat_mean(ai), stat_sr(ai)*100);
        end
        fprintf('\n');

        % Save per-(D, trap) .mat
        mat_path  = fullfile(RESULTS_DIR, ...
                    sprintf('Run_11_results_D%d_%s.mat', D, trap_label));
        alg_names = ALG(:,1);
        save(mat_path, 'D','trap_label','trap_name','half_w', ...
             'shift_vec','x_opt','x_trap','lb_init','ub_init', ...
             'spatial_gap','fitness_gap','f_trap','f_opt', ...
             'N_RUNS','N_POP','Max_FE','SUCCESS_THRESH', ...
             'alg_names','best_f_mat','conv_cell','fe_to_tgt', ...
             'stat_mean','stat_std','stat_best','stat_worst', ...
             'stat_median','stat_sr','stat_fet','rank_ord');
        fprintf('  Saved: %s\n\n', mat_path);

        % Save per-(D, trap) .csv
        csv_path = fullfile(RESULTS_DIR, ...
                   sprintf('Run_11_results_D%d_%s.csv', D, trap_label));
        write_csv_trap(csv_path, ALG(:,1), stat_mean, stat_std, ...
                       stat_best, stat_worst, stat_median, stat_sr, stat_fet, ...
                       D, trap_label, trap_name, half_w, ...
                       spatial_gap, fitness_gap, N_RUNS, Max_FE, SUCCESS_THRESH);
        fprintf('  Saved: %s\n\n', csv_path);

    end % trap loop

    % ----- Cross-trap summary ─────────────────────────────────────────────
    fprintf('  ╔══════════════════════════════════════════════════════════╗\n');
    fprintf('  ║  TRAP SUMMARY — D=%d — Success Rate (%%)                \n', D);
    fprintf('  ║  (SR > 0%% means algorithm escaped the trap)            \n');
    fprintf('  ╚══════════════════════════════════════════════════════════╝\n');
    fprintf('  %-8s', 'Algorithm');
    for ti = 1:N_TRAPS
        fprintf(' %8s', TRAP_LABELS{ti});
    end
    fprintf('  | Escaped any trap?\n');
    fprintf('  %s\n', repmat('-',1,55));
    for ai = 1:N_ALG
        fprintf('  %-8s', ALG{ai,1});
        any_escape = false;
        for ti = 1:N_TRAPS
            fprintf(' %7.1f%%', summ_sr(ai,ti)*100);
            if summ_sr(ai,ti) > 0, any_escape = true; end
        end
        fprintf('  | %s\n', ternary(any_escape, 'YES', 'NO'));
    end
    fprintf('\n');

    % Escape ranking: by mean SR across all trap configs
    mean_sr = mean(summ_sr, 2);
    [~, esc_ord] = sort(mean_sr, 'descend');
    fprintf('  Escape ability ranking (mean SR across T1/T2/T3):\n');
    for r = 1:N_ALG
        ai = esc_ord(r);
        fprintf('    %2d. %-8s  mean SR=%.1f%%  | T1=%.1f%%  T2=%.1f%%  T3=%.1f%%\n', ...
                r, ALG{ai,1}, mean_sr(ai)*100, ...
                summ_sr(ai,1)*100, summ_sr(ai,2)*100, summ_sr(ai,3)*100);
    end
    fprintf('\n');

    % Save summary .mat
    summ_mat = fullfile(RESULTS_DIR, sprintf('Run_11_trap_summary_D%d.mat', D));
    alg_names = ALG(:,1);
    save(summ_mat, 'D','TRAP_LABELS','TRAP_NAMES','TRAP_HALFWIDTHS', ...
         'shift_vec','x_opt','x_trap','spatial_gap','fitness_gap', ...
         'N_RUNS','N_POP','SUCCESS_THRESH', ...
         'alg_names','summ_mean','summ_std','summ_sr','summ_fet','mean_sr');
    fprintf('  Saved summary: %s\n\n', summ_mat);

    % Save summary .csv
    summ_csv = fullfile(RESULTS_DIR, sprintf('Run_11_trap_summary_D%d.csv', D));
    write_summary_csv(summ_csv, ALG(:,1), summ_mean, summ_sr, summ_fet, ...
                      TRAP_LABELS, TRAP_NAMES, TRAP_HALFWIDTHS, D, ...
                      N_RUNS, spatial_gap, fitness_gap, SUCCESS_THRESH);
    fprintf('  Saved summary: %s\n\n', summ_csv);

end % dimension loop

fprintf('=========================================================\n');
fprintf('  Experiment 11 complete.\n');
fprintf('  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% Shifted Schwefel — f* ≈ 0 at x = shift_vec. Identical to Exp 3.
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function s = format_fet(fet)
    if isnan(fet), s = 'N/A'; else, s = sprintf('%d', round(fet)); end
end


function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end


function write_csv_trap(csv_path, alg_names, ...
                         s_mean, s_std, s_best, s_worst, s_median, ...
                         s_sr, s_fet, D, trap_label, trap_name, half_w, ...
                         spatial_gap, fitness_gap, N_RUNS, Max_FE, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp11: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_11_Local_Trap_Initialization\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Trap_config,%s (%s)\n', trap_label, trap_name);
    fprintf(fid, 'Trap_half_width,%d\n', half_w);
    fprintf(fid, 'Trap_centre,shift_vec - 302.5232 (nearest Schwefel local min)\n');
    fprintf(fid, 'Spatial_gap_opt_to_trap,%.4f\n', spatial_gap);
    fprintf(fid, 'Fitness_gap_trap_minus_opt,%.4f\n', fitness_gap);
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
                            TRAP_LABELS, TRAP_NAMES, TRAP_HW, D, N_RUNS, ...
                            spatial_gap, fitness_gap, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp11: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_11_Trap_Summary\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Spatial_gap,%.4f\n', spatial_gap);
    fprintf(fid, 'Fitness_gap,%.4f\n', fitness_gap);
    fprintf(fid, 'Success_threshold,%.1f\n', thresh);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm');
    for ti = 1:numel(TRAP_LABELS)
        fprintf(fid, ',%s_Mean,%s_SR,%s_FEt', ...
                TRAP_LABELS{ti}, TRAP_LABELS{ti}, TRAP_LABELS{ti});
    end
    fprintf(fid, ',Mean_SR_across_traps\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s', alg_names{i});
        for ti = 1:numel(TRAP_LABELS)
            fet_str = 'NaN';
            if ~isnan(summ_fet(i,ti))
                fet_str = sprintf('%d', round(summ_fet(i,ti)));
            end
            fprintf(fid, ',%.6e,%.4f,%s', ...
                    summ_mean(i,ti), summ_sr(i,ti), fet_str);
        end
        fprintf(fid, ',%.4f\n', mean(summ_sr(i,:)));
    end
    fclose(fid);
end
