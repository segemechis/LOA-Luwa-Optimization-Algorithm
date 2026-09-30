% =========================================================================
%  Run_Statistical_Analysis.m
%  Statistical Analysis — Friedman Rank Test + Wilcoxon Signed-Rank Test
%
%  Covers all 53 sub-experiments across 7 experiment groups:
%    Group 1 — Standard Schwefel  (Exp01/02): D=10,30,50,100
%    Group 2 — Shifted Schwefel   (Exp04):    D=10,30,50
%    Group 3 — Progressive Shift  (Exp08):    D=30,50 × S0-S5
%    Group 4 — Partial-Dim Shift  (Exp09):    D=30,50 × P25-P100
%    Group 5 — Distant Init       (Exp10):    D=30,50 × I1-I5
%    Group 6 — Local Trap         (Exp11):    D=30,50 × T1-T3
%    Group 7 — Wide-Gap Severity  (Exp12):    D=30,50 × WG1-WG5
%
%  Tests:
%    1. Friedman test — overall significance across all sub-experiments
%       Reports: average rank per algorithm, chi-squared statistic, p-value
%
%    2. Wilcoxon signed-rank test — LOA vs each competitor, per sub-exp
%       Paired on run seeds (same rng(r) → same initialisation → paired)
%       Reports: W statistic, Z approximation, p-value, effect size r
%       Significance threshold: alpha = 0.05 (two-tailed)
%
%    3. Post-hoc Holm correction — controls family-wise error rate
%       Applied across the 14 Wilcoxon tests per sub-experiment
%
%  Outputs (C:\LOA — Luwa Optimization Algorithm\Results\Stats\):
%    Table_Friedman.csv     — Friedman ranks per group + overall
%    Table_Wilcoxon.csv     — All pairwise LOA tests (p-values, effect sizes)
%    Table_AvgRank.csv      — Overall average rank per algorithm
%    Table_WilcoxonHolm.csv — Holm-corrected significance flags
%    Statistical_Analysis.mat — Full workspace
%    Stats_log.txt           — Console mirror
%
%  Author: Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

clc; clear; close all;

%% ── 0. PATHS ─────────────────────────────────────────────────────────────
BASE_DIR    = 'C:\LOA — Luwa Optimization Algorithm\Results';
STATS_DIR   = fullfile(BASE_DIR, 'Stats');
if ~exist(STATS_DIR, 'dir'), mkdir(STATS_DIR); end

diary(fullfile(STATS_DIR, 'Stats_log.txt'));
diary on;

%% ── 1. ALGORITHM LIST ────────────────────────────────────────────────────
ALG_NAMES = {'CCO','CMA-ES','DE','GA','GWO','HHO','LOA', ...
             'L-SHADE','MVO','PSO','RIME','SA','SCA','SO','WOA'};
N_ALG  = numel(ALG_NAMES);
LOA_IDX = find(strcmp(ALG_NAMES, 'LOA'));
N_RUNS = 30;
ALPHA  = 0.05;

fprintf('=========================================================\n');
fprintf('  Statistical Analysis — LOA Paper                       \n');
fprintf('  Algorithms: %d | Runs per test: %d | Alpha: %.2f      \n', ...
        N_ALG, N_RUNS, ALPHA);
fprintf('=========================================================\n\n');

%% ── 2. DEFINE ALL SUB-EXPERIMENTS ────────────────────────────────────────
% Each entry: {group_name, mat_path, mat_var, label}
% mat_var = name of the N_ALG x N_RUNS matrix inside the .mat file
subexp = {};

% Group 1 — Standard Schwefel
for D = [10, 30]
    subexp{end+1} = {'G1_Standard', ...
        fullfile(BASE_DIR,'Exp01',sprintf('Run_01_results_D%d.mat',D)), ...
        'best_f_mat', sprintf('Std_D%d',D)};
end
for D = [50, 100]
    subexp{end+1} = {'G1_Standard', ...
        fullfile(BASE_DIR,'Exp02',sprintf('Run_02_results_D%d.mat',D)), ...
        'best_f_mat', sprintf('Std_D%d',D)};
end

% Group 2 — Shifted Schwefel
for D = [10, 30, 50]
    subexp{end+1} = {'G2_Shifted', ...
        fullfile(BASE_DIR,'Exp04',sprintf('Run_04_results_D%d.mat',D)), ...
        'best_f_mat', sprintf('Sft_D%d',D)};
end

% Group 3 — Progressive Shift Severity
SEV_LABELS = {'S0','S1','S2','S3','S4','S5'};
for D = [30, 50]
    for si = 1:6
        subexp{end+1} = {'G3_ProgShift', ...
            fullfile(BASE_DIR,'Exp08', ...
                sprintf('Run_08_results_D%d_%s.mat',D,SEV_LABELS{si})), ...
            'best_f_mat', sprintf('PS_D%d_%s',D,SEV_LABELS{si})};
    end
end

% Group 4 — Partial-Dimension Shift
FRAC_LABELS = {'P25','P50','P75','P100'};
for D = [30, 50]
    for fi = 1:4
        subexp{end+1} = {'G4_PartShift', ...
            fullfile(BASE_DIR,'Exp09', ...
                sprintf('Run_09_results_D%d_%s.mat',D,FRAC_LABELS{fi})), ...
            'best_f_mat', sprintf('PD_D%d_%s',D,FRAC_LABELS{fi})};
    end
end

% Group 5 — Distant Initialization
CASE_LABELS = {'I1','I2','I3','I4','I5'};
for D = [30, 50]
    for ci = 1:5
        subexp{end+1} = {'G5_DistInit', ...
            fullfile(BASE_DIR,'Exp10', ...
                sprintf('Run_10_results_D%d_%s.mat',D,CASE_LABELS{ci})), ...
            'best_f_mat', sprintf('DI_D%d_%s',D,CASE_LABELS{ci})};
    end
end

% Group 6 — Local Trap
TRAP_LABELS = {'T1','T2','T3'};
for D = [30, 50]
    for ti = 1:3
        subexp{end+1} = {'G6_LocalTrap', ...
            fullfile(BASE_DIR,'Exp11', ...
                sprintf('Run_11_results_D%d_%s.mat',D,TRAP_LABELS{ti})), ...
            'best_f_mat', sprintf('LT_D%d_%s',D,TRAP_LABELS{ti})};
    end
end

% Group 7 — Wide-Gap Severity
WG_LABELS = {'WG1','WG2','WG3','WG4','WG5'};
for D = [30, 50]
    for wi = 1:5
        subexp{end+1} = {'G7_WideGap', ...
            fullfile(BASE_DIR,'Exp12', ...
                sprintf('Run_12_results_D%d_%s.mat',D,WG_LABELS{wi})), ...
            'best_f_mat', sprintf('WG_D%d_%s',D,WG_LABELS{wi})};
    end
end

N_SUBEXP = numel(subexp);
fprintf('Total sub-experiments: %d\n\n', N_SUBEXP);

%% ── 3. LOAD ALL DATA ─────────────────────────────────────────────────────
% all_data{se} = N_ALG x N_RUNS matrix of final fitness values
all_data   = cell(N_SUBEXP, 1);
all_labels = cell(N_SUBEXP, 1);
all_groups = cell(N_SUBEXP, 1);
load_ok    = false(N_SUBEXP, 1);

for se = 1:N_SUBEXP
    group   = subexp{se}{1};
    matpath = subexp{se}{2};
    matvar  = subexp{se}{3};
    label   = subexp{se}{4};

    all_labels{se} = label;
    all_groups{se} = group;

    if ~exist(matpath, 'file')
        fprintf('  MISSING: %s\n', matpath);
        continue;
    end

    try
        S = load(matpath, matvar);
        mat = S.(matvar);
        if size(mat,1) ~= N_ALG || size(mat,2) ~= N_RUNS
            fprintf('  SIZE MISMATCH at %s: got [%d x %d]\n', ...
                    label, size(mat,1), size(mat,2));
            continue;
        end
        all_data{se} = mat;
        load_ok(se)  = true;
    catch ME
        fprintf('  ERROR loading %s: %s\n', label, ME.message);
    end
end

n_loaded = sum(load_ok);
fprintf('Loaded %d / %d sub-experiments successfully.\n\n', n_loaded, N_SUBEXP);

% Filter to loaded only
valid_idx  = find(load_ok);
data_valid = all_data(valid_idx);
labs_valid = all_labels(valid_idx);
grps_valid = all_groups(valid_idx);
N_VALID    = numel(valid_idx);

%% ── 4. FRIEDMAN RANK TEST ────────────────────────────────────────────────
% For each sub-experiment, rank the N_ALG algorithms by their median
% final fitness (lower rank = better performance).
% Then aggregate ranks across sub-experiments using the Friedman test.

fprintf('=========================================================\n');
fprintf('  FRIEDMAN RANK TEST\n');
fprintf('=========================================================\n\n');

% rank_matrix(se, ai) = rank of algorithm ai in sub-experiment se
rank_matrix = zeros(N_VALID, N_ALG);

for se = 1:N_VALID
    mat = data_valid{se};   % N_ALG x N_RUNS
    medians = zeros(N_ALG, 1);
    for ai = 1:N_ALG
        vals = mat(ai,:);
        valid_v = vals(isfinite(vals));
        if isempty(valid_v)
            medians(ai) = Inf;
        else
            medians(ai) = median(valid_v);
        end
    end
    % Rank: 1 = best (lowest median)
    [~, sort_idx] = sort(medians);
    ranks = zeros(N_ALG, 1);
    ranks(sort_idx) = 1:N_ALG;
    rank_matrix(se, :) = ranks';
end

% Average ranks across all sub-experiments
avg_ranks = mean(rank_matrix, 1);
[sorted_ranks, rank_ord] = sort(avg_ranks);

fprintf('  Overall Average Friedman Rank (lower = better):\n');
fprintf('  %s\n', repmat('-',1,45));
for r = 1:N_ALG
    ai = rank_ord(r);
    fprintf('  %2d. %-8s  avg rank = %.4f\n', r, ALG_NAMES{ai}, avg_ranks(ai));
end
fprintf('\n');

% Friedman chi-squared statistic
% chi2_F = 12*N/(k*(k+1)) * sum(R_j^2) - 3*N*(k+1)
% N = number of sub-experiments, k = number of algorithms
N_se = N_VALID;
k    = N_ALG;
chi2_F = (12 * N_se) / (k * (k+1)) * sum(avg_ranks.^2) - 3 * N_se * (k+1);
df_F   = k - 1;
% p-value from chi-squared distribution
p_friedman = 1 - chi2cdf(chi2_F, df_F);

fprintf('  Friedman test: chi2(%.0f) = %.4f,  p = %.6f\n', ...
        df_F, chi2_F, p_friedman);
if p_friedman < ALPHA
    fprintf('  Result: SIGNIFICANT — algorithms differ (p < %.2f)\n\n', ALPHA);
else
    fprintf('  Result: NOT SIGNIFICANT (p >= %.2f)\n\n', ALPHA);
end

% Per-group average ranks
groups_unique = unique(grps_valid);
fprintf('  Per-group average ranks (LOA highlighted):\n');
for g = 1:numel(groups_unique)
    grp = groups_unique{g};
    grp_idx = strcmp(grps_valid, grp);
    grp_ranks = rank_matrix(grp_idx, :);
    grp_avg = mean(grp_ranks, 1);
    [~, grp_ord] = sort(grp_avg);
    loa_rank = grp_avg(LOA_IDX);
    loa_pos  = find(grp_ord == LOA_IDX);
    fprintf('  %s (n=%d sub-exp): LOA avg rank=%.2f (pos=%d/%d)\n', ...
            grp, sum(grp_idx), loa_rank, loa_pos, N_ALG);
end
fprintf('\n');

%% ── 5. WILCOXON SIGNED-RANK TEST ─────────────────────────────────────────
% For each sub-experiment: LOA vs each of 14 competitors
% Paired on run index (rng(r) ensures same initialisation)

fprintf('=========================================================\n');
fprintf('  WILCOXON SIGNED-RANK TEST  (LOA vs each competitor)    \n');
fprintf('=========================================================\n\n');

% Store results: p_wilcox(se, ci) = p-value for LOA vs competitor ci
% ci = 1..14 (skipping LOA itself)
comp_idx = setdiff(1:N_ALG, LOA_IDX);   % 14 competitor indices
N_COMP   = numel(comp_idx);

p_wilcox  = ones(N_VALID, N_COMP);    % p-values
z_wilcox  = zeros(N_VALID, N_COMP);   % Z statistic
r_effect  = zeros(N_VALID, N_COMP);   % effect size r = |Z|/sqrt(N)
loa_wins  = zeros(N_VALID, N_COMP);   % 1 if LOA significantly better
loa_loses = zeros(N_VALID, N_COMP);   % 1 if LOA significantly worse

for se = 1:N_VALID
    mat      = data_valid{se};
    loa_vals = mat(LOA_IDX, :)';   % 30 x 1

    for ci_idx = 1:N_COMP
        ai = comp_idx(ci_idx);
        comp_vals = mat(ai, :)';   % 30 x 1

        % Handle NaN: replace with large penalty value
        loa_clean  = loa_vals;
        comp_clean = comp_vals;
        max_val = max([loa_clean(isfinite(loa_clean)); ...
                       comp_clean(isfinite(comp_clean))]) * 2;
        if isempty(max_val) || max_val <= 0, max_val = 1e10; end
        loa_clean(~isfinite(loa_clean))   = max_val;
        comp_clean(~isfinite(comp_clean)) = max_val;

        % Differences: positive = LOA better (lower cost)
        d = comp_clean - loa_clean;

        % Wilcoxon signed-rank test
        [p, ~, stats] = signrank(loa_clean, comp_clean, 'tail', 'both');
        p_wilcox(se, ci_idx) = p;

        % Z statistic (from normal approximation for N=30)
        if isfield(stats, 'zval')
            z = stats.zval;
        else
            % Manual normal approximation
            W     = stats.signedrank;
            n_eff = sum(d ~= 0);
            mu_W  = n_eff * (n_eff+1) / 4;
            sig_W = sqrt(n_eff * (n_eff+1) * (2*n_eff+1) / 24);
            z     = (W - mu_W) / (sig_W + eps);
        end
        z_wilcox(se, ci_idx)  = z;
        r_effect(se, ci_idx)  = abs(z) / sqrt(N_RUNS);

        loa_wins(se, ci_idx)  = (p < ALPHA) && (mean(loa_clean) < mean(comp_clean));
        loa_loses(se, ci_idx) = (p < ALPHA) && (mean(loa_clean) > mean(comp_clean));
    end
end

% Summary: wins and losses per competitor
fprintf('  LOA Win/Loss Summary across %d sub-experiments:\n', N_VALID);
fprintf('  %-10s %8s %8s %8s %10s\n', 'Competitor','Wins','Losses','Ties','Win%%');
fprintf('  %s\n', repmat('-',1,50));
for ci_idx = 1:N_COMP
    ai       = comp_idx(ci_idx);
    wins     = sum(loa_wins(:, ci_idx));
    losses   = sum(loa_loses(:, ci_idx));
    ties     = N_VALID - wins - losses;
    win_pct  = wins / N_VALID * 100;
    fprintf('  %-10s %8d %8d %8d %9.1f%%\n', ...
            ALG_NAMES{ai}, wins, losses, ties, win_pct);
end
fprintf('\n');

% LOA overall
total_wins   = sum(loa_wins(:));
total_losses = sum(loa_loses(:));
total_tests  = N_VALID * N_COMP;
fprintf('  LOA TOTAL: %d wins / %d losses / %d ties out of %d tests\n', ...
        total_wins, total_losses, total_tests-total_wins-total_losses, total_tests);
fprintf('  Overall LOA win rate: %.1f%%\n\n', total_wins/total_tests*100);

%% ── 6. HOLM CORRECTION ───────────────────────────────────────────────────
% Apply Holm step-down correction per sub-experiment (14 tests each)

fprintf('=========================================================\n');
fprintf('  HOLM-CORRECTED SIGNIFICANCE (alpha=%.2f)               \n', ALPHA);
fprintf('=========================================================\n\n');

sig_holm = zeros(N_VALID, N_COMP);   % 1 = significant after Holm

for se = 1:N_VALID
    p_row = p_wilcox(se, :);
    [p_sorted, sort_idx] = sort(p_row);
    m = N_COMP;
    reject = false(1, m);
    for k = 1:m
        threshold = ALPHA / (m - k + 1);
        if p_sorted(k) <= threshold
            reject(k) = true;
        else
            break;  % Holm: stop at first non-rejection
        end
    end
    for k = 1:m
        orig_idx = sort_idx(k);
        if reject(k)
            sig_holm(se, orig_idx) = 1;
        end
    end
end

% Summary of Holm-significant wins
fprintf('  Holm-significant LOA wins per competitor:\n');
fprintf('  %-10s %10s %10s\n','Competitor','Sig Wins','of %d');
fprintf('  %s\n', repmat('-',1,36));
for ci_idx = 1:N_COMP
    ai = comp_idx(ci_idx);
    sig_wins = sum(sig_holm(:,ci_idx) & loa_wins(:,ci_idx));
    fprintf('  %-10s %10d %10d\n', ALG_NAMES{ai}, sig_wins, N_VALID);
end
fprintf('\n');

%% ── 7. SAVE ALL RESULTS ──────────────────────────────────────────────────

% Save .mat
save(fullfile(STATS_DIR, 'Statistical_Analysis.mat'), ...
     'ALG_NAMES','N_ALG','LOA_IDX','ALPHA', ...
     'subexp','N_SUBEXP','N_VALID','load_ok', ...
     'labs_valid','grps_valid', ...
     'rank_matrix','avg_ranks','rank_ord','sorted_ranks', ...
     'chi2_F','df_F','p_friedman', ...
     'comp_idx','p_wilcox','z_wilcox','r_effect', ...
     'loa_wins','loa_loses','sig_holm');
fprintf('Saved: Statistical_Analysis.mat\n');

% Table_AvgRank.csv
fid = fopen(fullfile(STATS_DIR,'Table_AvgRank.csv'),'w');
fprintf(fid,'Rank,Algorithm,AvgFriedmanRank,Friedman_chi2,df,p_value\n');
for r = 1:N_ALG
    ai = rank_ord(r);
    fprintf(fid,'%d,%s,%.4f,%.4f,%d,%.6f\n', ...
            r, ALG_NAMES{ai}, avg_ranks(ai), chi2_F, df_F, p_friedman);
end
fclose(fid);
fprintf('Saved: Table_AvgRank.csv\n');

% Table_Friedman.csv — per sub-experiment ranks
fid = fopen(fullfile(STATS_DIR,'Table_Friedman.csv'),'w');
fprintf(fid,'SubExp,Group');
for ai = 1:N_ALG
    fprintf(fid,',%s', ALG_NAMES{ai});
end
fprintf(fid,'\n');
for se = 1:N_VALID
    fprintf(fid,'%s,%s', labs_valid{se}, grps_valid{se});
    for ai = 1:N_ALG
        fprintf(fid,',%d', rank_matrix(se,ai));
    end
    fprintf(fid,'\n');
end
% Average row
fprintf(fid,'AVERAGE,ALL');
for ai = 1:N_ALG
    fprintf(fid,',%.4f', avg_ranks(ai));
end
fprintf(fid,'\n');
fclose(fid);
fprintf('Saved: Table_Friedman.csv\n');

% Table_Wilcoxon.csv — p-values and effect sizes
fid = fopen(fullfile(STATS_DIR,'Table_Wilcoxon.csv'),'w');
fprintf(fid,'SubExp,Group,Competitor,p_value,Z,EffectSize_r,');
fprintf(fid,'LOA_wins,Sig_after_Holm\n');
for se = 1:N_VALID
    for ci_idx = 1:N_COMP
        ai = comp_idx(ci_idx);
        fprintf(fid,'%s,%s,%s,%.6f,%.4f,%.4f,%d,%d\n', ...
                labs_valid{se}, grps_valid{se}, ALG_NAMES{ai}, ...
                p_wilcox(se,ci_idx), z_wilcox(se,ci_idx), ...
                r_effect(se,ci_idx), ...
                loa_wins(se,ci_idx), sig_holm(se,ci_idx));
    end
end
fclose(fid);
fprintf('Saved: Table_Wilcoxon.csv\n');

% Table_WilcoxonHolm.csv — compact significance matrix
fid = fopen(fullfile(STATS_DIR,'Table_WilcoxonHolm.csv'),'w');
fprintf(fid,'SubExp,Group');
for ci_idx = 1:N_COMP
    fprintf(fid,',%s', ALG_NAMES{comp_idx(ci_idx)});
end
fprintf(fid,'\n');
for se = 1:N_VALID
    fprintf(fid,'%s,%s', labs_valid{se}, grps_valid{se});
    for ci_idx = 1:N_COMP
        % +1 = LOA wins sig, -1 = LOA loses sig, 0 = tie/ns
        val = sig_holm(se,ci_idx) * (loa_wins(se,ci_idx) - loa_loses(se,ci_idx));
        fprintf(fid,',%d', val);
    end
    fprintf(fid,'\n');
end
fclose(fid);
fprintf('Saved: Table_WilcoxonHolm.csv\n\n');

fprintf('=========================================================\n');
fprintf('  Statistical analysis complete.\n');
fprintf('  Results: %s\n', STATS_DIR);
fprintf('=========================================================\n');

diary off;


% =========================================================================
%  LOCAL FUNCTION
% =========================================================================

function p = chi2cdf(x, k)
% Chi-squared CDF using regularised incomplete gamma function.
    p = gammainc(x/2, k/2, 'lower');
end
