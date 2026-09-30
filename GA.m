function [x_best, f_best, convergence] = GA(obj_func, lb, ub, N, T_max, options)
% GA  Genetic Algorithm (real-valued, tournament selection, SBX + polynomial mutation).
%
%  A canonical real-coded GA following Deb (2001):
%    Selection  : binary tournament (deterministic, pressure = 2)
%    Crossover  : Simulated Binary Crossover (SBX), eta_c = 20
%    Mutation   : Polynomial Mutation, eta_m = 20, p_m = 1/d
%    Elitism    : best individual always survives (mu + 1 elitism)
%
%  Reference: Deb K. Multi-Objective Optimization using Evolutionary
%  Algorithms. Wiley, 2001.
%
%  INPUTS:
%    obj_func - function handle: f = obj_func(x), x is [1 x d] row vector
%    lb, ub   - bounds [1 x d]
%    N        - population size (even number recommended)
%    T_max    - maximum iterations (generations)
%
%  OUTPUTS:
%    x_best     - best position found [1 x d]
%    f_best     - best objective value
%    convergence - convergence curve [T_max x 1]

    d     = numel(lb);
    eta_c = 20;          % SBX distribution index
    eta_m = 20;          % polynomial mutation distribution index
    p_m   = 1 / d;       % per-gene mutation probability
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

        % --- Binary tournament selection (N parents) -------------------
        parents = zeros(N, d);
        for k = 1:N
            a = randi(N);  b = randi(N);
            if F_val(a) <= F_val(b)
                parents(k,:) = X(a,:);
            else
                parents(k,:) = X(b,:);
            end
        end

        % --- SBX crossover + polynomial mutation → offspring -----------
        offspring   = zeros(N, d);
        offspring_f = zeros(N, 1);

        i = 1;
        while i <= N
            p1 = parents(i,:);
            if i < N
                p2 = parents(i+1,:);
            else
                p2 = parents(1,:);   % wrap-around for odd N
            end

            % SBX crossover
            c1 = zeros(1,d);  c2 = zeros(1,d);
            for j = 1:d
                if rand < 0.5 && abs(p1(j) - p2(j)) > 1e-10
                    x_lo = min(p1(j), p2(j));
                    x_hi = max(p1(j), p2(j));
                    beta_lo = 1 + 2*(x_lo - lb(j)) / (x_hi - x_lo);
                    beta_hi = 1 + 2*(ub(j) - x_hi) / (x_hi - x_lo);
                    alpha1  = 2 - beta_lo^(-(eta_c+1));
                    alpha2  = 2 - beta_hi^(-(eta_c+1));
                    u = rand;
                    if u <= 1/alpha1
                        beta_q1 = (u * alpha1)^(1/(eta_c+1));
                    else
                        beta_q1 = (1/(2 - u*alpha1))^(1/(eta_c+1));
                    end
                    u = rand;
                    if u <= 1/alpha2
                        beta_q2 = (u * alpha2)^(1/(eta_c+1));
                    else
                        beta_q2 = (1/(2 - u*alpha2))^(1/(eta_c+1));
                    end
                    c1(j) = 0.5*((p1(j)+p2(j)) - beta_q1*abs(p2(j)-p1(j)));
                    c2(j) = 0.5*((p1(j)+p2(j)) + beta_q2*abs(p2(j)-p1(j)));
                else
                    c1(j) = p1(j);
                    c2(j) = p2(j);
                end
            end

            % Polynomial mutation on child 1
            for j = 1:d
                if rand < p_m
                    delta_max = min(c1(j)-lb(j), ub(j)-c1(j)) / (ub(j)-lb(j));
                    u = rand;
                    if u < 0.5
                        delta_q = (2*u + (1-2*u)*(1-delta_max)^(eta_m+1))^(1/(eta_m+1)) - 1;
                    else
                        delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta_max)^(eta_m+1))^(1/(eta_m+1));
                    end
                    c1(j) = c1(j) + delta_q * (ub(j) - lb(j));
                end
            end
            c1 = max(lb, min(ub, c1));
            offspring(i,:)   = c1;
            offspring_f(i)   = obj_func(c1);

            if i < N
                % Polynomial mutation on child 2
                for j = 1:d
                    if rand < p_m
                        delta_max = min(c2(j)-lb(j), ub(j)-c2(j)) / (ub(j)-lb(j));
                        u = rand;
                        if u < 0.5
                            delta_q = (2*u + (1-2*u)*(1-delta_max)^(eta_m+1))^(1/(eta_m+1)) - 1;
                        else
                            delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta_max)^(eta_m+1))^(1/(eta_m+1));
                        end
                        c2(j) = c2(j) + delta_q * (ub(j) - lb(j));
                    end
                end
                c2 = max(lb, min(ub, c2));
                offspring(i+1,:)   = c2;
                offspring_f(i+1)   = obj_func(c2);
            end

            i = i + 2;
        end

        % --- Elitist replacement: keep best from {X ∪ offspring} ------
        combined_X = [X; offspring];
        combined_F = [F_val; offspring_f];
        [combined_F_sorted, sort_idx] = sort(combined_F);
        X     = combined_X(sort_idx(1:N), :);
        F_val = combined_F_sorted(1:N);

        % Update global best
        if F_val(1) < f_best
            f_best = F_val(1);
            x_best = X(1,:);
        end

        convergence(t) = f_best;
    end
end
