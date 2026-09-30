function [x_best, f_best, conv] = wrap_CMAES(obj_scalar, lb, ub, N_pop, Max_FE, lb_init, ub_init)
% wrap_CMAES  Run CMA-ES on the PS-MPPT benchmark.
%
% CMA-ES SIGNATURE:
%   [x_best, f_best, convergence] = CMAES(obj_func, lb, ub, N, T_max)
%
% KEY NOTES:
%   • CMA-ES is a generation-based algorithm; T_max here is iterations
%     (generations). FE budget = N_pop * T_max.
%   • N_pop controls the lambda (population / offspring) size.
%   • CMA-ES is self-adaptive and generally performs best with its default
%     lambda = floor(4 + 3*log(d)); for the 1D MPPT problem d=1 this gives
%     lambda ≈ 4. However N_pop is passed through for fair FE-budget parity.
%   • Bounds are enforced by mirroring within CMAES.m (reflect strategy).
%   • convergence is a column vector of length T_max (one value per generation).
%
% INPUTS:
%   obj_scalar — handle returning f = obj_scalar(x)
%   lb, ub     — scalar bounds on D
%   N_pop      — population size (lambda)
%   Max_FE     — total function evaluation budget
%
% OUTPUTS:
%   x_best — best duty ratio found
%   f_best — best (minimized) objective value
%   conv   — row vector of best fitness per iteration (length Max_iter)
%
% Author: Sena Gemechis File, Jimma Institute of Technology

    % ── Optional constrained initialisation ─────────────────────────────
    options = struct();
    if nargin >= 7 && ~isempty(lb_init) && ~isempty(ub_init)
        d   = numel(lb);
        X0  = lb_init + rand(N_pop, d) .* (ub_init - lb_init);
        options.X0 = X0;
    end

    Max_iter = floor(Max_FE / N_pop);

    [x_best, f_best, convergence] = CMAES(obj_scalar, lb, ub, N_pop, Max_iter, options);

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
