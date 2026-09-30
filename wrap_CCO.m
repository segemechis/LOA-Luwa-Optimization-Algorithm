function [x_best, f_best, conv] = wrap_CCO(obj_scalar, lb, ub, N_pop, Max_FE, lb_init, ub_init)
% wrap_CCO  Run CCO on the PS-MPPT benchmark.
%
% CCO SIGNATURE:
%   [x_best, f_best, convergence] = CCO(obj_func, lb, ub, N, T_max)
%
% KEY NOTES:
%   • CCO (Cuckoo Catfish Optimizer) expects a scalar-returning objective.
%   • T_max is the number of iterations; FE budget ≈ N_pop * T_max.
%   • CCO may perform extra evaluations via parasitism-rate redraws;
%     actual FE count may marginally exceed N_pop * T_max.
%   • convergence is a column vector of length T_max.
%
% INPUTS:
%   obj_scalar — handle returning f = obj_scalar(x)
%   lb, ub     — scalar bounds on D
%   N_pop      — population size
%   Max_FE     — total function evaluation budget
%
% OUTPUTS:
%   x_best — best duty ratio found
%   f_best — best (minimized) objective value
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


    [x_best, f_best, convergence] = CCO(obj_scalar, lb, ub, N_pop, Max_iter, options);

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
