function [x_best, f_best, convergence] = SA(obj_func, lb, ub, N, T_max, options)
% SA  Simulated Annealing (continuous-domain, Boltzmann acceptance).
%
%  A standard continuous-space SA following Kirkpatrick et al. (1983).
%  The algorithm runs N independent restarts in parallel and accepts or
%  rejects each candidate according to the Metropolis criterion.  At the
%  end of every iteration the global best is updated from all chains.
%
%  Cooling schedule: geometric,  T_ann(t) = T0 * alpha^t
%    T0    = 1.0   (initial temperature, normalised to fitness scale)
%    alpha = (T_final/T0)^(1/T_max)  with  T_final = 1e-6
%
%  Neighbour generation: Gaussian perturbation with adaptive step size
%    sigma(t) = sigma_max * (1 - t/T_max)  (linear decay)
%    sigma_max = 0.1 * (ub - lb)
%
%  Reference: Kirkpatrick S, Gelatt C D, Vecchi M P. Optimization by
%  Simulated Annealing. Science, 1983, 220(4598):671-680.
%
%  INPUTS:
%    obj_func - function handle: f = obj_func(x), x is [1 x d] row vector
%    lb, ub   - bounds [1 x d]
%    N        - number of parallel chains (population-like interface)
%    T_max    - total number of annealing steps (iterations)
%
%  OUTPUTS:
%    x_best     - best position found [1 x d]
%    f_best     - best objective value
%    convergence - convergence curve [T_max x 1]

    d       = numel(lb);
    T0      = 1.0;
    T_final = 1e-6;
    alpha   = (T_final / T0)^(1 / T_max);  % cooling rate

    sigma_max = 0.1 * (ub - lb);           % max Gaussian step width
    if nargin < 6 || ~isstruct(options), options = struct(); end

    % ── Initialisation ──────────────────────────────────────────────────
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X     = lb + rand(N, d) .* (ub - lb);
    end
    F_val = zeros(N, 1);
    for i = 1:N
        F_val(i) = obj_func(X(i,:));
    end

    [f_best, idx] = min(F_val);
    x_best        = X(idx,:);

    T_ann       = T0;
    convergence = zeros(T_max, 1);

    % ── Main loop ───────────────────────────────────────────────────────
    for t = 1:T_max

        % Adaptive step size (linearly decaying)
        sigma = sigma_max * (1 - t / T_max);
        sigma = max(sigma, 1e-8 * (ub - lb));  % floor to avoid zero step

        for i = 1:N
            % Generate neighbour via Gaussian perturbation
            x_new = X(i,:) + sigma .* randn(1, d);
            x_new = max(lb, min(ub, x_new));   % reflect/clamp to bounds

            f_new = obj_func(x_new);

            % Metropolis acceptance criterion
            delta = f_new - F_val(i);
            if delta < 0 || rand < exp(-delta / T_ann)
                X(i,:)   = x_new;
                F_val(i) = f_new;
            end

            % Update global best
            if F_val(i) < f_best
                f_best = F_val(i);
                x_best = X(i,:);
            end
        end

        % Geometric cooling
        T_ann = T_ann * alpha;

        convergence(t) = f_best;
    end
end
