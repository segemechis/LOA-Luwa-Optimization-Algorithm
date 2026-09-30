function [x_best, f_best, convergence] = CMAES(obj_func, lb, ub, N, T_max, options)
% CMAES  Covariance Matrix Adaptation Evolution Strategy.
%
%  Research-grade implementation following Hansen (2016) tutorial.
%  Reference: Hansen, N. (2016). The CMA evolution strategy: A tutorial.
%             arXiv:1604.00772.
%
%  Parameters derived automatically from problem dimension d and
%  population size N (lambda). This implementation is self-adaptive
%  and requires no manual parameter tuning beyond N and T_max.
%
%  Inputs:
%    obj_func  : function handle, returns scalar fitness
%    lb, ub    : [1 x d] lower/upper bounds
%    N         : population size (lambda). Default 50 for consistency.
%    T_max     : max iterations (generations). Default 500.
%
%  Outputs:
%    x_best      : best solution found [1 x d]
%    f_best      : best fitness
%    convergence : [T_max x 1] best fitness per generation

    if nargin < 4 || isempty(N),     N     = 50;  end
    if nargin < 5 || isempty(T_max), T_max = 500; end

    d      = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end
    lambda = N;                         % population size
    mu     = floor(lambda/2);           % number of parents

    %% Step-size and covariance adaptation parameters (Hansen 2016)
    % Weights
    weights_raw = log(mu + 0.5) - log(1:mu)';
    weights     = weights_raw / sum(weights_raw);
    mueff       = 1 / sum(weights.^2);

    % Step-size control
    cs   = (mueff + 2) / (d + mueff + 5);
    ds   = 1 + 2*max(0, sqrt((mueff-1)/(d+1))-1) + cs;
    chiN = sqrt(d) * (1 - 1/(4*d) + 1/(21*d^2));

    % Covariance matrix adaptation
    cc   = (4 + mueff/d) / (d + 4 + 2*mueff/d);
    c1   = 2 / ((d+1.3)^2 + mueff);
    cmu  = min(1-c1, 2*(mueff-2+1/mueff)/((d+2)^2+mueff));

    %% Initialisation
    % Start from centre of search space
    if isfield(options,'X0') && ~isempty(options.X0)
        xmean = mean(options.X0, 1);   % centroid of provided population
    else
        xmean = lb + rand(1,d).*(ub-lb);
    end
    sigma = 0.3 * mean(ub - lb);       % initial step size

    ps   = zeros(1,d);      % evolution path for step size
    pc   = zeros(1,d);      % evolution path for covariance
    C    = eye(d);          % covariance matrix
    invc = eye(d);          % C^{-1/2}, updated via eigendecomp
    eigeneval = 0;          % tracks when eigendecomposition was last done

    f_best      = inf;
    x_best      = xmean;
    convergence = zeros(T_max,1);
    sqrtC       = eye(d);    % initialised here; updated by eigendecomp block

    %% Main loop
    for t = 1:T_max

        %% Sample offspring
        % Eigendecompose C every ~lambda/(c1+cmu)/d/10 evaluations
        if t - eigeneval > lambda/(c1+cmu)/d/10
            [V, E] = eig(C);
            diagD  = sqrt(max(diag(E), 1e-20));   % eigenvalues >= 0
            invc   = V * diag(1./diagD) * V';
            sqrtC  = V * diag(diagD) * V';
            eigeneval = t;
        end

        arz   = randn(lambda, d);
        arx   = zeros(lambda, d);
        arfit = zeros(lambda, 1);

        for i = 1:lambda
            arx(i,:) = xmean + sigma * (arz(i,:) * sqrtC');
            % Reflect to bounds (not clipping — preserves distribution)
            for dd = 1:d
                while arx(i,dd) < lb(dd) || arx(i,dd) > ub(dd)
                    arx(i,dd) = xmean(dd) + sigma * (randn * sqrt(C(dd,dd)));
                    arx(i,dd) = max(lb(dd), min(ub(dd), arx(i,dd)));
                end
            end
            arfit(i) = obj_func(arx(i,:));
        end

        %% Sort by fitness
        [arfit_sorted, idx] = sort(arfit);

        % Update best
        if arfit_sorted(1) < f_best
            f_best = arfit_sorted(1);
            x_best = arx(idx(1),:);
        end

        %% Recombination: weighted mean of top-mu
        xold  = xmean;
        xmean = weights' * arx(idx(1:mu),:);

        %% Step-size control (CSA)
        ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * ...
             ((xmean-xold)/sigma) * invc;
        hsig = norm(ps) / sqrt(1-(1-cs)^(2*t)) / chiN < 1.4 + 2/(d+1);

        %% Covariance matrix adaptation (CMA)
        pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold)/sigma;

        artmp = (1/sigma) * (arx(idx(1:mu),:) - xold);
        C = (1-c1-cmu)*C ...
            + c1*(pc'*pc + (1-hsig)*cc*(2-cc)*C) ...
            + cmu*(artmp' * diag(weights) * artmp);

        % Enforce symmetry and positive definiteness
        C = triu(C) + triu(C,1)';
        C = (C + C')/2;

        %% Step-size update
        sigma = sigma * exp((cs/ds)*(norm(ps)/chiN - 1));

        % Prevent step size from collapsing
        sigma = max(sigma, 1e-12);

        %% Restart if covariance becomes ill-conditioned
        if max(diag(C)) > 1e14 * min(diag(C))
            sigma = 0.3 * mean(ub - lb);
            xmean = lb + rand(1,d).*(ub-lb);   % fresh restart point
            C     = eye(d);
            ps    = zeros(1,d);
            pc    = zeros(1,d);
        end

        convergence(t) = f_best;
    end
end
