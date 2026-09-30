function [x_best, f_best, conv] = wrap_SA(obj_scalar, lb, ub, N_pop, Max_FE, lb_init, ub_init)
% wrap_SA  Run SA on the Schwefel benchmark.
%
% SA SIGNATURE:
%   [x_best, f_best, convergence] = SA(obj_func, lb, ub, N, T_max)
%
% KEY NOTES:
%   • SA (parallel-chain Simulated Annealing) expects a scalar-returning
%     objective: f = obj_func(x).
%   • N parallel chains are run simultaneously; each chain generates one
%     candidate per iteration, so FE budget = N_pop * T_max.
%   • T_max is the number of annealing steps; FE budget = N_pop * T_max.
%   • Cooling: geometric schedule, T0=1.0 → T_final=1e-6 over T_max steps.
%   • Neighbour: Gaussian perturbation with linearly decaying sigma_max.
%   • convergence is a column vector of length T_max.
%
% INPUTS:
%   obj_scalar — handle returning f = obj_scalar(x)
%   lb, ub     — bounds [1 x d]
%   N_pop      — number of parallel SA chains
%   Max_FE     — total function evaluation budget
%
% OUTPUTS:
%   x_best — best solution found [1 x d]
%   f_best — best (minimised) objective value
%   conv   — row vector of best fitness per iteration (length Max_iter)
%
% Author: Sena Gemechis File, Jimma Institute of Technology

    % ── Optional constrained initialisation ─────────────────────────────
    % When lb_init/ub_init are supplied the algorithm starts its population
    % inside that sub-region; the full search space lb/ub remains active.
    options = struct();
    if nargin >= 7 && ~isempty(lb_init) && ~isempty(ub_init)
        d   = numel(lb);
        X0  = lb_init + rand(N_pop, d) .* (ub_init - lb_init);
        options.X0 = X0;
    end

    Max_iter = floor(Max_FE / N_pop);


    [x_best, f_best, convergence] = SA(obj_scalar, lb, ub, N_pop, Max_iter, options);

    conv = standardize_conv(convergence, f_best, Max_iter);
end


function conv = standardize_conv(conv_raw, f_best, Max_iter)
% Ensure convergence vector is a row vector of length Max_iter.
    if isempty(conv_raw) || all(~isfinite(conv_raw))
        conv = repmat(f_best, 1, Max_iter);
        return;
    end
    conv_raw = conv_raw(:)';
    if length(conv_raw) == Max_iter
        conv = conv_raw;
    elseif length(conv_raw) > Max_iter
        idx  = round(linspace(1, length(conv_raw), Max_iter));
        conv = conv_raw(idx);
    else
        conv = [conv_raw, repmat(conv_raw(end), 1, Max_iter - length(conv_raw))];
    end
end
