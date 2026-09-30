function [x_best, f_best, convergence] = RIME(obj_func, lb, ub, N, T_max, options)
% RIME  Rime Optimization Algorithm (Su et al., 2023)
    d = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X = lb + rand(N,d).*(ub-lb);
    end
    F = zeros(N,1);
    for i=1:N, F(i)=obj_func(X(i,:)); end
    [f_best,idx]=min(F); x_best=X(idx,:);
    convergence=zeros(T_max,1);
    for t=1:T_max
        E = (1-t/T_max)^3;  % rime-ice factor
        for i=1:N
            if rand < 0.5
                % Soft-rime search
                X_new = x_best .* (1 + randn(1,d)*E);
            else
                % Hard-rime puncture
                r_idx = randi(N);
                X_new = X(r_idx,:) + E*(x_best - X(i,:)).*randn(1,d);
            end
            % Positive refraction lens
            if rand < 0.2
                j = randi(d);
                X_new(j) = x_best(j) + (ub(j)-lb(j))*randn*E*0.1;
            end
            X_new = max(lb, min(ub, X_new));
            f_new = obj_func(X_new);
            if f_new < F(i), X(i,:)=X_new; F(i)=f_new; end
            if f_new < f_best, f_best=f_new; x_best=X_new; end
        end
        convergence(t)=f_best;
    end
end
