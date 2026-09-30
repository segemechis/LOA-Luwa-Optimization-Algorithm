function [x_best, f_best, convergence] = MVO(obj_func, lb, ub, N, T_max, options)
% MVO  Multi-Verse Optimizer (Mirjalili et al., 2016)
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
        WEP = 0.2 + (1-0.2)*(t/T_max);  % wormhole existence probability
        TDR = 1 - (t/T_max)^(1/6);       % travelling distance rate
        [~,sorted_idx]=sort(F);
        NI = cumsum(ones(N,1))./N;  % normalized inflation rates
        for i=1:N
            for j=1:d
                r1=rand;
                if r1 < NI(i)
                    white_idx = roulette_select(F);
                    X(i,j) = X(sorted_idx(white_idx),j);
                end
                if rand < WEP
                    if rand < 0.5
                        X(i,j) = x_best(j) + TDR*((ub(j)-lb(j))*rand+lb(j));
                    else
                        X(i,j) = x_best(j) - TDR*((ub(j)-lb(j))*rand+lb(j));
                    end
                end
            end
            X(i,:) = max(lb, min(ub, X(i,:)));
            F(i) = obj_func(X(i,:));
            if F(i)<f_best, f_best=F(i); x_best=X(i,:); end
        end
        convergence(t)=f_best;
    end
end
function idx = roulette_select(F)
    F_inv = 1./(F-min(F)+1e-30);
    probs = F_inv/sum(F_inv);
    cum = cumsum(probs);
    idx = find(cum >= rand, 1);
    if isempty(idx), idx=numel(F); end
end
