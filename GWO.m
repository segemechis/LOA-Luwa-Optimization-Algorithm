function [x_best, f_best, convergence] = GWO(obj_func, lb, ub, N, T_max, options)
% GWO  Grey Wolf Optimizer (Mirjalili et al., 2014).
%
%  Standard three-leader hierarchy: alpha, beta, delta wolves.
%  Linearly decreasing a from 2 to 0 over iterations.
%
%  Reference: Mirjalili S, Mirjalili SM, Lewis A. Grey Wolf Optimizer.
%  Advances in Engineering Software, 2014, 69:46-61.

    d = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end

    % Initialization
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X = lb + rand(N, d) .* (ub - lb);
    end
    F = zeros(N, 1);
    for i = 1:N, F(i) = obj_func(X(i,:)); end

    % Sort to find alpha, beta, delta
    [F_sorted, idx] = sort(F);
    alpha_pos = X(idx(1),:);  alpha_score = F_sorted(1);
    beta_pos  = X(idx(2),:);  beta_score  = F_sorted(2);
    delta_pos = X(idx(3),:);  delta_score = F_sorted(3);

    x_best = alpha_pos;
    f_best = alpha_score;

    convergence = zeros(T_max, 1);

    for t = 1:T_max
        a = 2 - 2*t/T_max;   % linearly decreasing from 2 to 0

        for i = 1:N
            % Update position using alpha, beta, delta
            r1 = rand(1,d); r2 = rand(1,d);
            A1 = 2*a*r1 - a;  C1 = 2*r2;
            D_alpha = abs(C1.*alpha_pos - X(i,:));
            X1 = alpha_pos - A1.*D_alpha;

            r1 = rand(1,d); r2 = rand(1,d);
            A2 = 2*a*r1 - a;  C2 = 2*r2;
            D_beta = abs(C2.*beta_pos - X(i,:));
            X2 = beta_pos - A2.*D_beta;

            r1 = rand(1,d); r2 = rand(1,d);
            A3 = 2*a*r1 - a;  C3 = 2*r2;
            D_delta = abs(C3.*delta_pos - X(i,:));
            X3 = delta_pos - A3.*D_delta;

            X(i,:) = (X1 + X2 + X3) / 3;

            % Boundary
            X(i,:) = max(lb, min(ub, X(i,:)));

            fi = obj_func(X(i,:));

            % Update leaders
            if fi < alpha_score
                delta_pos = beta_pos;   delta_score = beta_score;
                beta_pos  = alpha_pos;  beta_score  = alpha_score;
                alpha_pos = X(i,:);     alpha_score = fi;
            elseif fi < beta_score
                delta_pos = beta_pos;   delta_score = beta_score;
                beta_pos  = X(i,:);     beta_score  = fi;
            elseif fi < delta_score
                delta_pos = X(i,:);     delta_score = fi;
            end
        end

        x_best = alpha_pos;
        f_best = alpha_score;
        convergence(t) = f_best;
    end
end
