function [x_best, f_best, convergence] = SCA(obj_func, lb, ub, N, T_max, options)
% SCA  Sine Cosine Algorithm.
%
%  Implements the original SCA exactly as described in Mirjalili (2016).
%  Position update equations:
%
%    x_i(t+1) = x_i(t) + r1 * sin(r2) * |r3 * x_dest - x_i(t)|   if r4 < 0.5
%    x_i(t+1) = x_i(t) + r1 * cos(r2) * |r3 * x_dest - x_i(t)|   otherwise
%
%  where
%    r1 = a - t*(a/T_max)    (linearly decreasing, a = 2)
%    r2 ~ U[0, 2*pi]
%    r3 ~ U[0, 2]
%    r4 ~ U[0, 1]
%    x_dest = current global best position
%
%  Reference: Mirjalili S. SCA: A Sine Cosine Algorithm for Solving
%  Optimization Problems. Knowledge-Based Systems, 2016, 96:120-133.
%
%  INPUTS:
%    obj_func - function handle: f = obj_func(x), x is [1 x d] row vector
%    lb, ub   - bounds [1 x d]
%    N        - population size
%    T_max    - maximum iterations
%
%  OUTPUTS:
%    x_best     - best position found [1 x d]
%    f_best     - best objective value
%    convergence - convergence curve [T_max x 1]

    d = numel(lb);
    a = 2;   % control parameter for r1 (Mirjalili 2016, Eq. 3.4)
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

    convergence = zeros(T_max, 1);

    % ── Main loop ───────────────────────────────────────────────────────
    for t = 1:T_max

        % Adaptive parameter r1 (Eq. 3.4 in Mirjalili 2016)
        r1 = a - t * (a / T_max);

        for i = 1:N
            r2 = 2 * pi * rand(1, d);   % random angle
            r3 = 2 * rand(1, d);        % weight on destination
            r4 = rand(1, d);            % sine/cosine switch

            % Position update (vectorised over dimensions)
            diff = abs(r3 .* x_best - X(i,:));

            X(i,:) = X(i,:) + r1 * ( ...
                sin(r2) .* diff .* (r4 <  0.5) + ...
                cos(r2) .* diff .* (r4 >= 0.5) );

            % Boundary: clamping
            X(i,:) = max(lb, min(ub, X(i,:)));

            % Evaluate
            fi = obj_func(X(i,:));
            F_val(i) = fi;

            % Update global best
            if fi < f_best
                f_best = fi;
                x_best = X(i,:);
            end
        end

        convergence(t) = f_best;
    end
end
