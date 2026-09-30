% =========================================================================
%  Run_10_Distant_Initialization.m
%  Experiment 10 — Distant-Initialization Schwefel
%
%  Motivation:
%    In real-world optimisation the initial population is often poorly
%    positioned relative to the true optimum — engineers may start from
%    a known but suboptimal operating point, or the search must begin
%    in a constrained commissioning region far from the global solution.
%    This experiment tests whether each algorithm can RECOVER from an
%    increasingly distant initial placement and still find the globally
%    shifted Schwefel optimum.
%
%  Setup:
%    The objective is the same shifted Schwefel as Experiment 3:
%      shift_vec drawn at seed (SHIFT_SEED + D), o ~ U[-200, 200]^D
%      f(x) = 418.9829*D - sum( ỹ.*sin(sqrt(|ỹ|)) ),  ỹ=clamp(x-shift+420.9687)
%      Global optimum: f* ≈ 0 at x* = shift_vec
%    Search space: [-500, 500]^D (always the full space — not restricted).
%
%  Initialization regions:
%    Five cases defined by how far the initial population centre is from
%    the true optimum x*. Centre moves linearly from x* toward the
%    diametrically opposite corner of the search space:
%
%    Case  Label     Centre dist from x*   Init box (centre ± 50)
%    I1    Near      0%  (at x*)           [x*-50, x*+50] ∩ [-500,500]
%    I2    25%       25% of max_dist       centre + 25%*(x_far-x*)
%    I3    50%       50% of max_dist       centre + 50%*(x_far-x*)
%    I4    75%       75% of max_dist       centre + 75%*(x_far-x*)
%    I5    Extreme   100% (opposite corner) x_far ± 50
%
%    where x_far(i) = -500 if x*(i) >= 0,  else +500
%    (the corner farthest from x* in each dimension)
%
%    All algorithms receive the same lb_init/ub_init for each case via
%    the updated wrapper interface. The full search space lb/ub =
%    [-500,500]^D remains active throughout — init is the ONLY variable.
%
%  Metrics: Mean, Std, Best, Worst, Median, SR (f < 1.0), FE-to-target
%
%  Dimensions : D = 30, 50
%  Algorithms : 15 (CCO, CMA-ES, DE, GA, GWO, HHO, LOA, L-SHADE,
%                   MVO, PSO, RIME, SA, SCA, SO, WOA)
%  Runs       : 30 per (algorithm, case, dimension)
%  FE budget  : D × 10,000 per run
%  Population : 30 agents
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Exp10\):
%    Run_10_results_D<d>_<case>.mat   — workspace per (D, init case)
%    Run_10_results_D<d>_<case>.csv   — statistics
%    Run_10_init_summary_D<d>.mat     — cross-case summary arrays
%    Run_10_init_summary_D<d>.csv     — case × algorithm table
%    Run_10_log.txt                   — console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
CODE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\LOA MATLAB Code';
RESULTS_DIR = 'C:\LOA — Luwa Optimization Algorithm\Results\Exp10';

addpath(CODE_DIR);
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

log_path = fullfile(RESULTS_DIR, 'Run_10_log.txt');
diary(log_path);
diary on;

%% ── 1. EXPERIMENT SETTINGS ───────────────────────────────────────────────
DIM_LIST       = [30, 50];
SHIFT_SEED     = 9999;          % same as Exp 3 — identical shifted landscape
HALF_WIDTH     = 50;            % half-width of init box per dimension
CASE_LABELS    = {'I1','I2','I3','I4','I5'};
CASE_NAMES     = {'Near','25%','50%','75%','Extreme'};
CASE_FRACS     = [0.00, 0.25, 0.50, 0.75, 1.00];  % fraction toward far corner
N_CASES        = numel(CASE_LABELS);
N_RUNS         = 30;
N_POP          = 30;
LB_SCALAR      = -500;
UB_SCALAR      =  500;
SUCCESS_THRESH = 1.0;

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
fprintf('  Experiment 10 — Distant-Initialization Schwefel        \n');
fprintf('  Dimensions  : D = 30, 50                               \n');
fprintf('  Init cases  : I1(Near) I2(25%%) I3(50%%) I4(75%%) I5(Extreme)\n');
fprintf('  Runs        : %d per (algorithm, case, dimension)       \n', N_RUNS);
fprintf('  Shift seed  : %d (same landscape as Exp 3)             \n', SHIFT_SEED);
fprintf('  Search space: [-500,500]^D always (init only differs)  \n');
fprintf('  Success thr : f < %.1f                                  \n', SUCCESS_THRESH);
fprintf('=========================================================\n\n');

for di = 1:numel(DIM_LIST)
    D      = DIM_LIST(di);
    Max_FE = D * 10000;
    lb     = LB_SCALAR * ones(1, D);
    ub     = UB_SCALAR * ones(1, D);

    % ── Reconstruct shifted Schwefel for this D (same as Exp 3) ──────────
    rng(SHIFT_SEED + D);
    shift_vec = -200 + 400 * rand(1, D);   % o ~ U[-200,200]^D
    x_opt     = shift_vec;                  % global optimum

    % Farthest corner from x_opt
    x_far = zeros(1, D);
    x_far(x_opt >= 0) = LB_SCALAR;    % if opt is positive, far corner is -500
    x_far(x_opt <  0) = UB_SCALAR;    % if opt is negative, far corner is +500

    max_dist = norm(x_far - x_opt);

    fprintf('=========================================================\n');
    fprintf('  D = %d   |   FE budget = %d\n', D, Max_FE);
    fprintf('  x_opt (shift_vec) first 5: [');
    fprintf(' %.3f', x_opt(1:min(5,D)));
    fprintf(' ...]\n');
    fprintf('  Max distance (x_opt → far corner): %.2f\n', max_dist);
    fprintf('  Init box half-width: %.0f per dimension\n', HALF_WIDTH);
    fprintf('=========================================================\n\n');

    % Verify optimum
    f_check = schwefel_shifted(x_opt, shift_vec, D);
    fprintf('  Sanity: f(x_opt) = %.6e  (should be ≈ 0)\n\n', f_check);
    if f_check > 1.0
        error('Exp10: f(x_opt)=%.4e at D=%d. Abort.', f_check, D);
    end

    % Objectives (fixed for all cases this D)
    obj_scalar = @(x) schwefel_shifted(x, shift_vec, D);
    obj_loa    = @(x) deal(schwefel_shifted(x, shift_vec, D), 0);

    % Summary arrays: (N_ALG × N_CASES)
    summ_mean = zeros(N_ALG, N_CASES);
    summ_std  = zeros(N_ALG, N_CASES);
    summ_sr   = zeros(N_ALG, N_CASES);
    summ_fet  = nan(N_ALG, N_CASES);

    for ci = 1:N_CASES
        case_label = CASE_LABELS{ci};
        case_name  = CASE_NAMES{ci};
        frac       = CASE_FRACS(ci);

        % ── Build init box for this case ──────────────────────────────────
        % Centre: move from x_opt toward x_far by frac of max_dist
        center  = x_opt + frac * (x_far - x_opt);
        lb_init = max(LB_SCALAR, center - HALF_WIDTH);
        ub_init = min(UB_SCALAR, center + HALF_WIDTH);

        % Actual centre distance from x_opt
        actual_dist = norm(center - x_opt);
        box_width   = min(ub_init - lb_init);

        fprintf('  ── %s (%s) ──────────────────────────────────────────\n', ...
                case_label, case_name);
        fprintf('  Centre dist from x_opt: %.2f  (%.1f%% of max_dist=%.0f)\n', ...
                actual_dist, 100*actual_dist/max_dist, max_dist);
        fprintf('  lb_init(1:3)=[%.2f %.2f %.2f...]  ub_init(1:3)=[%.2f %.2f %.2f...]\n', ...
                lb_init(1),lb_init(2),lb_init(3), ...
                ub_init(1),ub_init(2),ub_init(3));
        fprintf('  Min box width: %.2f\n\n', box_width);

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

            fprintf('  [D=%2d | %s] Running %-8s ', D, case_label, alg_name);
            t_start = tic;

            for r = 1:N_RUNS
                rng(r);
                try
                    % Pass lb_init/ub_init as 6th/7th args to trigger
                    % constrained initialisation in the updated wrappers.
                    % Full search bounds lb/ub remain [-500,500]^D.
                    [~, fbest, conv] = wrap_fn(obj_fn, lb, ub, N_POP, Max_FE, ...
                                               lb_init, ub_init);
                catch ME
                    warning('Exp10: %s run %d D=%d %s failed: %s', ...
                            alg_name, r, D, case_label, ME.message);
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

        summ_mean(:,ci) = stat_mean;
        summ_std(:,ci)  = stat_std;
        summ_sr(:,ci)   = stat_sr;
        summ_fet(:,ci)  = stat_fet;

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
        fprintf('\n  Ranking (%s, by mean):\n', case_label);
        for r = 1:N_ALG
            ai = rank_ord(r);
            fprintf('    %2d. %-8s mean=%10.3e  SR=%4.1f%%\n', ...
                    r, ALG{ai,1}, stat_mean(ai), stat_sr(ai)*100);
        end
        fprintf('\n');

        % Save per-(D, case) .mat
        mat_path  = fullfile(RESULTS_DIR, ...
                    sprintf('Run_10_results_D%d_%s.mat', D, case_label));
        alg_names = ALG(:,1);
        save(mat_path, 'D','case_label','case_name','frac', ...
             'center','lb_init','ub_init','actual_dist','max_dist', ...
             'shift_vec','x_opt','N_RUNS','N_POP','Max_FE','SUCCESS_THRESH', ...
             'alg_names','best_f_mat','conv_cell','fe_to_tgt', ...
             'stat_mean','stat_std','stat_best','stat_worst', ...
             'stat_median','stat_sr','stat_fet','rank_ord');
        fprintf('  Saved: %s\n\n', mat_path);

        % Save per-(D, case) .csv
        csv_path = fullfile(RESULTS_DIR, ...
                   sprintf('Run_10_results_D%d_%s.csv', D, case_label));
        write_csv_case(csv_path, ALG(:,1), stat_mean, stat_std, ...
                       stat_best, stat_worst, stat_median, stat_sr, stat_fet, ...
                       D, case_label, case_name, actual_dist, max_dist, ...
                       N_RUNS, Max_FE, SUCCESS_THRESH);
        fprintf('  Saved: %s\n\n', csv_path);

    end % case loop

    % ----- Cross-case summary ─────────────────────────────────────────────
    fprintf('  ╔══════════════════════════════════════════════════════════╗\n');
    fprintf('  ║  INIT-DISTANCE SUMMARY — D=%d — Success Rate (%%)       \n', D);
    fprintf('  ╚══════════════════════════════════════════════════════════╝\n');
    fprintf('  %-8s', 'Algorithm');
    for ci = 1:N_CASES
        fprintf(' %7s', CASE_LABELS{ci});
    end
    fprintf('  | I1→I5 drop\n');
    fprintf('  %s\n', repmat('-',1,67));
    for ai = 1:N_ALG
        fprintf('  %-8s', ALG{ai,1});
        for ci = 1:N_CASES
            fprintf(' %6.1f%%', summ_sr(ai,ci)*100);
        end
        drop = (summ_sr(ai,1) - summ_sr(ai,N_CASES))*100;
        fprintf('  | %+.1f%%\n', -drop);
    end
    fprintf('\n');

    % Robustness: smallest SR drop I1→I5
    sr_drop = summ_sr(:,1) - summ_sr(:,N_CASES);
    [~, drop_ord] = sort(sr_drop);
    fprintf('  Recovery robustness ranking (smallest SR drop I1→I5):\n');
    for r = 1:N_ALG
        ai = drop_ord(r);
        fprintf('    %2d. %-8s  I1=%4.1f%%  I3=%4.1f%%  I5=%4.1f%%\n', ...
                r, ALG{ai,1}, summ_sr(ai,1)*100, ...
                summ_sr(ai,3)*100, summ_sr(ai,5)*100);
    end
    fprintf('\n');

    % Save summary .mat
    summ_mat = fullfile(RESULTS_DIR, ...
               sprintf('Run_10_init_summary_D%d.mat', D));
    alg_names = ALG(:,1);
    save(summ_mat, 'D','CASE_LABELS','CASE_NAMES','CASE_FRACS', ...
         'shift_vec','x_opt','max_dist','HALF_WIDTH', ...
         'N_RUNS','N_POP','SUCCESS_THRESH', ...
         'alg_names','summ_mean','summ_std','summ_sr','summ_fet');
    fprintf('  Saved summary: %s\n\n', summ_mat);

    % Save summary .csv
    summ_csv = fullfile(RESULTS_DIR, ...
               sprintf('Run_10_init_summary_D%d.csv', D));
    write_summary_csv(summ_csv, ALG(:,1), summ_mean, summ_sr, summ_fet, ...
                      CASE_LABELS, CASE_NAMES, CASE_FRACS, D, N_RUNS, ...
                      max_dist, SUCCESS_THRESH);
    fprintf('  Saved summary: %s\n\n', summ_csv);

end % dimension loop

fprintf('=========================================================\n');
fprintf('  Experiment 10 complete.\n');
fprintf('  Results: %s\n', RESULTS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = schwefel_shifted(x, shift_vec, d)
% Shifted Schwefel — identical to Experiment 3.
% f* ≈ 0 at x = shift_vec.
    y = x - shift_vec + 420.9687;
    y = max(-500, min(500, y));
    f = 418.9829 * d - sum(y .* sin(sqrt(abs(y))));
end


function s = format_fet(fet)
    if isnan(fet), s = 'N/A'; else, s = sprintf('%d', round(fet)); end
end


function write_csv_case(csv_path, alg_names, ...
                         s_mean, s_std, s_best, s_worst, s_median, ...
                         s_sr, s_fet, D, case_label, case_name, ...
                         actual_dist, max_dist, N_RUNS, Max_FE, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp10: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_10_Distant_Initialization\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Init_case,%s (%s)\n', case_label, case_name);
    fprintf(fid, 'Centre_dist_from_optimum,%.4f\n', actual_dist);
    fprintf(fid, 'Max_possible_dist,%.4f\n', max_dist);
    fprintf(fid, 'Normalised_dist,%.4f\n', actual_dist/max_dist);
    fprintf(fid, 'Init_box_half_width,50\n');
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
                            CASE_LABELS, CASE_NAMES, CASE_FRACS, D, N_RUNS, ...
                            max_dist, thresh)
    fid = fopen(csv_path, 'w');
    if fid == -1, warning('Exp10: cannot open %s', csv_path); return; end
    fprintf(fid, 'Experiment,Run_10_Distant_Init_Summary\n');
    fprintf(fid, 'Dimension,%d\n', D);
    fprintf(fid, 'Max_dist_search_space,%.4f\n', max_dist);
    fprintf(fid, 'Success_threshold,%.1f\n', thresh);
    fprintf(fid, 'Independent_Runs,%d\n', N_RUNS);
    fprintf(fid, '\n');
    fprintf(fid, 'Algorithm');
    for ci = 1:numel(CASE_LABELS)
        fprintf(fid, ',%s_Mean,%s_SR,%s_FEt', ...
                CASE_LABELS{ci}, CASE_LABELS{ci}, CASE_LABELS{ci});
    end
    fprintf(fid, '\n');
    for i = 1:numel(alg_names)
        fprintf(fid, '%s', alg_names{i});
        for ci = 1:numel(CASE_LABELS)
            fet_str = 'NaN';
            if ~isnan(summ_fet(i,ci))
                fet_str = sprintf('%d', round(summ_fet(i,ci)));
            end
            fprintf(fid, ',%.6e,%.4f,%s', ...
                    summ_mean(i,ci), summ_sr(i,ci), fet_str);
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
