function [x_best, f_best, convergence] = CCO(obj_func, lb, ub, N, T_max, options)
% CCO  Cuckoo Catfish Optimizer (Wang et al., 2025)
%  Inspired by brood-parasitic behavior of cuckoo catfish.
%  Two phases: infiltration (exploration) and parasitism (exploitation).
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
    pa = 0.25;  % parasitism rate
    for t=1:T_max
        alpha = 1 - t/T_max;  % decreasing step size
        for i=1:N
            if rand < 0.5
                % Infiltration phase (exploration via Levy flight)
                sigma = 0.5;
                u = randn(1,d)*sigma;
                v = randn(1,d);
                step = u./abs(v).^(1/3);
                X_new = X(i,:) + alpha*step.*(X(i,:)-x_best);
            else
                % Host nest approach (exploitation)
                r1=randi(N); r2=randi(N);
                while r1==i, r1=randi(N); end
                while r2==i||r2==r1, r2=randi(N); end
                phi = rand(1,d);
                X_new = X(i,:) + phi.*(X(r1,:)-X(r2,:));
            end
            X_new = max(lb, min(ub, X_new));
            f_new = obj_func(X_new);
            if f_new<F(i), X(i,:)=X_new; F(i)=f_new; end
            if f_new<f_best, f_best=f_new; x_best=X_new; end
        end
        % Parasitism: replace worst fraction with new random solutions
        [~,worst_idx]=sort(F,'descend');
        n_replace = max(1, round(pa*N));
        for ri=1:n_replace
            idx_r = worst_idx(ri);
            X(idx_r,:) = lb + rand(1,d).*(ub-lb);
            F(idx_r) = obj_func(X(idx_r,:));
            if F(idx_r)<f_best, f_best=F(idx_r); x_best=X(idx_r,:); end
        end
        convergence(t)=f_best;
    end
end
