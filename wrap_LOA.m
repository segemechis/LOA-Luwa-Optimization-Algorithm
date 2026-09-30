function [x_best, f_best, conv] = wrap_LOA(obj_loa, lb, ub, N_pop, Max_FE, lb_init, ub_init)
% wrap_LOA  Run LOA on the Schwefel benchmark (LOA-compatible objective).
%
% LOA SIGNATURE:
%   [x_best, f_best, convergence, convergence_cv, FE_count] = ...
%       LOA(obj_func, lb, ub, N_g, FE_max, options)
%
% KEY NOTES:
%   • obj_loa MUST return [f, cv] — wrap as @(x) deal(f(x), 0).
%
%   FAIRNESS FIX — equal total population across all algorithms:
%   • LOA runs 5 grades of N_g agents each → total = 5 * N_g agents.
%   • All competitor algorithms use N_pop agents directly.
%   • To make LOA comparable, N_g is set to floor(N_pop / 5) so that
%       total LOA agents = 5 * floor(N_pop / 5) ≈ N_pop.
%   • Example: N_pop = 30 → N_g = 6 → total = 30 agents. ✓
%   • N_g is enforced to be at least 4 (LOA internal minimum for
%     meaningful grade diversity and crossover parent selection).
%
%   CONVERGENCE RESAMPLING:
%   • LOA records f_best every N_g FEs → convergence length ≈ FE_max/N_g.
%   • This is resampled to Max_iter = floor(Max_FE / N_pop) points
%     (using the original N_pop, not N_g) so the convergence vector
%     length matches every other algorithm for fair plotting.
%
% INPUTS:
%   obj_loa — handle returning [f, cv] = obj_loa(x)
%   lb, ub  — bounds [1 x d]
%   N_pop   — nominal population size passed by the benchmark runner
%             (same value sent to all 15 algorithms)
%   Max_FE  — total function evaluation budget
%
% OUTPUTS:
%   x_best — best solution found [1 x d]
%   f_best — best (minimised) objective value
%   conv   — row vector of best fitness, resampled to
%            Max_iter = floor(Max_FE / N_pop) points
%
% Author: Sena Gemechis File, Jimma Institute of Technology

    % ── Fair population conversion ────────────────────────────────────
    N_GRADES = 5;
    N_g      = max(4, floor(N_pop / N_GRADES));
    % N_g >= 4 : hard floor so LOA always has enough agents per grade
    % for meaningful DE crossover parent selection (needs at least 3
    % distinct pool members) and Encampment clustering.

    options      = struct();
    options.seed = [];   % seed set externally by benchmark runner (rng)

    % ── Optional constrained initialisation ──────────────────────────
    % X0 must be (N_g*5) x d — one block of N_g rows per grade.
    if nargin >= 7 && ~isempty(lb_init) && ~isempty(ub_init)
        d_loc  = numel(lb);
        N_total = N_g * 5;
        options.X0 = lb_init + rand(N_total, d_loc) .* (ub_init - lb_init);
    end

    [x_best, f_best, convergence, ~, ~] = ...
        LOA(obj_loa, lb, ub, N_g, Max_FE, options);

    % ── Resample convergence to Max_iter points ───────────────────────
    % Max_iter is computed from N_pop (not N_g) so the output vector
    % length matches all other algorithms' wrappers exactly.
    Max_iter = floor(Max_FE / N_pop);
    if length(convergence) >= 2
        src_idx = linspace(1, length(convergence), Max_iter);
        conv    = interp1(1:length(convergence), convergence, src_idx, 'previous');
    else
        conv = repmat(f_best, 1, Max_iter);
    end
    conv = conv(:)';   % ensure row vector
end
