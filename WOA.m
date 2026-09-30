function [x_best, f_best, convergence] = WOA(obj_func, lb, ub, N, T_max, options)
% WOA  Whale Optimization Algorithm (Mirjalili & Lewis, 2016).
%
%  Implements shrinking encircling, spiral updating, and random search.
%  b = 1 (spiral shape constant), standard implementation.
%
%  Reference: Mirjalili S, Lewis A. The Whale Optimization Algorithm.
%  Advances in Engineering Software, 2016, 95:51-67.

    d = numel(lb);
    b = 1;   % spiral shape constant
    if nargin < 6 || ~isstruct(options), options = struct(); end

    % Initialization
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X = lb + rand(N, d) .* (ub - lb);
    end
    F = zeros(N, 1);
    for i = 1:N, F(i) = obj_func(X(i,:)); end

    [f_best, idx] = min(F);
    x_best = X(idx,:);

    convergence = zeros(T_max, 1);

    for t = 1:T_max
        a  = 2 - 2*t/T_max;    % decreases from 2 to 0
        a2 = -1 - t/T_max;     % decreases from -1 to -2

        for i = 1:N
            p  = rand;
            r1 = rand(1,d);
            r2 = rand(1,d);
            A  = 2*a*r1 - a;
            C  = 2*r2;
            l  = (a2 - 1)*rand + 1;   % random in [a2, 1]

            if p < 0.5
                if abs(A(1)) < 1
                    % Shrinking encircling
                    D = abs(C .* x_best - X(i,:));
                    X(i,:) = x_best - A .* D;
                else
                    % Random search (exploration)
                    rand_idx = randi(N);
                    X_rand   = X(rand_idx,:);
                    D = abs(C .* X_rand - X(i,:));
                    X(i,:) = X_rand - A .* D;
                end
            else
                % Spiral updating
                D = abs(x_best - X(i,:));
                X(i,:) = D .* exp(b*l) .* cos(2*pi*l) + x_best;
            end

            % Boundary
            X(i,:) = max(lb, min(ub, X(i,:)));

            fi = obj_func(X(i,:));
            if fi < f_best
                f_best = fi;
                x_best = X(i,:);
            end
        end

        convergence(t) = f_best;
    end
end
