function [x_best, f_best, convergence] = SO(obj_func, lb, ub, N, T_max, options)
% SO  Snake Optimizer (Hashim and Hussien, 2022)
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
    % Split into males and females
    N_half = floor(N/2);
    convergence=zeros(T_max,1);
    for t=1:T_max
        temp = exp(-t/T_max);  % temperature
        food = x_best;         % food position = best
        Q = rand;              % food quality
        for i=1:N
            if temp > 0.6  % cold - exploration
                if i <= N_half  % male
                    X_new = X(randi(N),:) + randn(1,d).*exp((X(randi(N),:)-X(i,:))/...
                            (d^2+eps));
                else  % female
                    X_new = X(randi(N),:) + randn(1,d).*exp((X(randi(N),:)-X(i,:))/...
                            (d^2+eps));
                end
            else  % warm - exploitation
                if Q > 0.5  % food quality high
                    if i <= N_half
                        X_new = food + 0.1*randn(1,d);
                    else
                        X_new = food + 0.1*randn(1,d);
                    end
                else  % food quality low - fight/mate
                    if i <= N_half
                        X_new = X(i,:) + rand*exp(food-X(i,:)).*cos(2*pi*rand(1,d));
                    else
                        X_new = X(i,:) + rand*exp(food-X(i,:)).*cos(2*pi*rand(1,d));
                    end
                end
            end
            X_new = max(lb, min(ub, X_new));
            f_new = obj_func(X_new);
            if f_new<F(i), X(i,:)=X_new; F(i)=f_new; end
            if f_new<f_best, f_best=f_new; x_best=X_new; end
        end
        convergence(t)=f_best;
    end
end
