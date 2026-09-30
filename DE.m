function [x_best, f_best, convergence] = DE(obj_func, lb, ub, N, T_max, options)
% DE  Differential Evolution (DE/rand/1/bin) -- standard variant.
%
%  Strategy: DE/rand/1/bin
%    F_scale = 0.8  (mutation scale factor)
%    CR      = 0.9  (crossover probability)
%
%  Reference: Storn R, Price K. Differential Evolution -- A Simple and
%  Efficient Heuristic for Global Optimization over Continuous Spaces.
%  Journal of Global Optimization, 1997, 11(4):341-359.

    d       = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end
    F_scale = 0.8;
    CR      = 0.9;

    % Initialization
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X = lb + rand(N, d) .* (ub - lb);
    end
    F_val = zeros(N, 1);
    for i = 1:N, F_val(i) = obj_func(X(i,:)); end

    [f_best, idx] = min(F_val);
    x_best = X(idx,:);

    convergence = zeros(T_max, 1);

    for t = 1:T_max
        for i = 1:N
            % Mutation: select three distinct random indices != i
            candidates = setdiff(1:N, i);
            r = candidates(randperm(numel(candidates), 3));
            r1 = r(1); r2 = r(2); r3 = r(3);

            % Mutant vector
            V_mut = X(r1,:) + F_scale * (X(r2,:) - X(r3,:));

            % Boundary clamping on mutant
            V_mut = max(lb, min(ub, V_mut));

            % Crossover (binomial)
            mask      = rand(1,d) < CR;
            j_rand    = randi(d);
            mask(j_rand) = true;   % ensure at least one component crosses

            U = X(i,:);
            U(mask) = V_mut(mask);

            % Boundary on trial vector
            U = max(lb, min(ub, U));

            % Selection
            f_trial = obj_func(U);
            if f_trial <= F_val(i)
                X(i,:)   = U;
                F_val(i) = f_trial;
                if f_trial < f_best
                    f_best = f_trial;
                    x_best = U;
                end
            end
        end

        convergence(t) = f_best;
    end
end
