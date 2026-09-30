function [x_best, f_best, conv] = wrap_LSHADE(obj_scalar, lb, ub, N_pop, Max_FE, lb_init, ub_init)
% wrap_LSHADE  Run L-SHADE on the PS-MPPT benchmark.
%
% LSHADE SIGNATURE:
%   [x_best, f_best, convergence] = LSHADE(obj_func, lb, ub, N_init, T_max)
%
% KEY NOTES:
%   • CRITICAL DIFFERENCE: In LSHADE.m, T_max is the TOTAL FUNCTION
%     EVALUATION BUDGET, not the number of iterations. This is consistent
%     with L-SHADE's adaptive population reduction scheme — the number of
%     iterations is not fixed in advance.
%   • Max_FE is therefore passed directly as T_max to LSHADE.
%   • LSHADE returns convergence as a column vector of length Max_FE
%     (one f_best value recorded per function evaluation).
%   • N_pop is passed as N_init (initial population size). LSHADE uses
%     N_min=4 and linearly reduces population during the run.
%   • The convergence vector is resampled to Max_iter = floor(Max_FE/N_pop)
%     points for consistent plotting against other algorithms.
%
% INPUTS:
%   obj_scalar — handle returning f = obj_scalar(x)
%   lb, ub     — scalar bounds on D
%   N_pop      — initial population size (N_init)
%   Max_FE     — total function evaluation budget (passed directly as T_max)
%
% OUTPUTS:
%   x_best — best duty ratio found
%   f_best — best (minimized) objective value
%   conv   — row vector of best fitness, resampled to Max_iter points
%
% Author: Sena Gemechis File, Jimma Institute of Technology

    % ── Optional constrained initialisation ─────────────────────────────
    options = struct();
    if nargin >= 7 && ~isempty(lb_init) && ~isempty(ub_init)
        d   = numel(lb);
        X0  = lb_init + rand(N_pop, d) .* (ub_init - lb_init);
        options.X0 = X0;
    end

    % LSHADE treats T_max as total FE budget — pass Max_FE directly.
    [x_best, f_best, convergence] = LSHADE(obj_scalar, lb, ub, N_pop, Max_FE, options);

    % Resample to Max_iter = floor(Max_FE/N_pop) points for consistent
    % comparison with iteration-based algorithms.
    Max_iter = floor(Max_FE / N_pop);
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
