function [x_best, f_best, convergence] = PSO(obj_func, lb, ub, N, T_max, options)
% PSO  Particle Swarm Optimization (standard inertia-weight variant).
%
%  Parameters follow the canonical PSO (Shi & Eberhart, 1998):
%    w  = inertia weight, linearly decreasing 0.9 -> 0.4
%    c1 = cognitive coefficient = 2.0
%    c2 = social coefficient    = 2.0
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

    d  = numel(lb);
    w_max = 0.9;  w_min = 0.4;
    c1 = 2.0;     c2 = 2.0;
    v_max = 0.2 * (ub - lb);   % velocity clamping
    if nargin < 6 || ~isstruct(options), options = struct(); end

    % Initialization
    if isfield(options,'X0') && ~isempty(options.X0)
        X  = options.X0;
    else
        X  = lb + rand(N, d) .* (ub - lb);   % positions
    end
    V  = zeros(N, d);                     % velocities
    F  = zeros(N, 1);
    for i = 1:N, F(i) = obj_func(X(i,:)); end

    pbest   = X;          % personal best positions
    pbest_f = F;          % personal best fitness

    [f_best, idx] = min(F);
    x_best = X(idx,:);

    convergence = zeros(T_max, 1);

    for t = 1:T_max
        w = w_max - (w_max - w_min) * t / T_max;

        for i = 1:N
            r1 = rand(1,d);  r2 = rand(1,d);
            V(i,:) = w*V(i,:) ...
                   + c1*r1.*(pbest(i,:) - X(i,:)) ...
                   + c2*r2.*(x_best     - X(i,:));

            % Velocity clamping
            V(i,:) = max(-v_max, min(v_max, V(i,:)));

            X(i,:) = X(i,:) + V(i,:);

            % Boundary: reflection
            below = X(i,:) < lb;
            above = X(i,:) > ub;
            X(i,below) = 2*lb(below) - X(i,below);
            X(i,above) = 2*ub(above) - X(i,above);
            X(i,:) = max(lb, min(ub, X(i,:)));

            fi = obj_func(X(i,:));

            % Update personal best
            if fi < pbest_f(i)
                pbest(i,:) = X(i,:);
                pbest_f(i) = fi;
            end

            % Update global best
            if fi < f_best
                f_best = fi;
                x_best = X(i,:);
            end
        end

        convergence(t) = f_best;
    end
end
