function [x_best, f_best, convergence] = LSHADE(obj_func, lb, ub, N_init, T_max, options)
% LSHADE  L-SHADE: Success-History based Adaptive DE with Linear pop reduction.
%
%  Reference:
%    Tanabe & Fukunaga (2014). "Improving the search performance of SHADE
%    using linear population size reduction." IEEE CEC 2014, pp. 1658-1665.
%
%  This is the standard L-SHADE implementation used as a strong baseline
%  in metaheuristic benchmarking (CEC2017 winner-class algorithm).
%
%  Inputs:
%    obj_func  - function handle f(x), x is a row vector
%    lb, ub    - lower/upper bounds (row vectors, length D)
%    N_init    - initial population size (default 18*D, standard L-SHADE)
%    T_max     - max function evaluations budget
%
%  Outputs:
%    x_best      - best solution found (row vector)
%    f_best      - best objective value
%    convergence - f_best recorded at each of T_max evaluations
%                  (downsampled to T_max points for compatibility)

    D = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end
    if nargin < 4 || isempty(N_init)
        N_init = max(18*D, 50);
    end
    if nargin < 5 || isempty(T_max)
        T_max = 500 * D;
    end

    %% L-SHADE hyper-parameters (standard settings from Tanabe & Fukunaga 2014)
    N_min  = 4;          % minimum population size
    H      = 6;          % history size for mu_CR and mu_F
    p_best = 0.11;       % greediness of current-to-pbest/1 mutation
    arc_rate = 2.6;      % archive size = arc_rate * N_init

    %% Initialisation
    N    = N_init;
    if isfield(options,'X0') && ~isempty(options.X0)
        pop = options.X0;
    else
        pop  = lb + rand(N, D) .* (ub - lb);
    end
    fit  = zeros(N, 1);
    for i = 1:N
        fit(i) = obj_func(pop(i,:));
    end

    [f_best, idx] = min(fit);
    x_best = pop(idx,:);

    %% History
    mu_CR = 0.5 * ones(H, 1);
    mu_F  = 0.5 * ones(H, 1);
    hist_idx = 1;

    archive   = zeros(0, D);   % external archive of replaced individuals
    arc_max   = round(arc_rate * N_init);

    NFE       = N;             % function evaluations used
    convergence_full = zeros(T_max, 1);
    convergence_full(1:NFE) = f_best;

    %% Main loop
    while NFE < T_max
        S_CR = [];  S_F = [];  delta_f = [];

        N_new = round(N_init + (N_min - N_init) * NFE / T_max);
        N_new = max(N_new, N_min);

        trial_pop = zeros(N, D);
        trial_fit = inf(N, 1);
        accept    = false(N, 1);

        for i = 1:N
            %% Adaptive CR and F
            r   = randi(H);
            CR_i = max(0, min(1, mu_CR(r) + 0.1*randn));
            F_i  = -1;
            while F_i <= 0
                F_i = mu_F(r) + 0.1*tan(pi*(rand-0.5));
            end
            F_i = min(F_i, 1);

            %% current-to-pbest/1 mutation
            p_num = max(2, round(p_best * N));
            [~, sorted_idx] = sort(fit(1:N));
            pbest_pool = sorted_idx(1:p_num);
            p_idx = pbest_pool(randi(p_num));

            r1 = i;
            while r1 == i, r1 = randi(N); end
            combined = [pop(1:N,:); archive];
            n_combined = size(combined, 1);
            r2 = randi(n_combined);
            retries = 0;
            while (r2 == i || r2 == r1) && retries < 100
                r2 = randi(n_combined);
                retries = retries + 1;
            end

            mutant = pop(i,:) + F_i*(pop(p_idx,:) - pop(i,:)) + F_i*(pop(r1,:) - combined(r2,:));
            mutant = max(lb, min(ub, mutant));

            %% Binomial crossover
            j_rand = randi(D);
            mask   = rand(1,D) < CR_i;
            mask(j_rand) = true;
            trial  = pop(i,:);
            trial(mask) = mutant(mask);

            %% Evaluate trial
            if NFE >= T_max, break; end
            f_trial = obj_func(trial);
            NFE = NFE + 1;
            convergence_full(NFE) = f_best;

            trial_pop(i,:) = trial;
            trial_fit(i)   = f_trial;

            if f_trial <= fit(i)
                accept(i) = true;
                delta_f   = [delta_f; abs(fit(i) - f_trial)]; %#ok<AGROW>
                S_CR      = [S_CR; CR_i];                     %#ok<AGROW>
                S_F       = [S_F;  F_i];                      %#ok<AGROW>
                if f_trial < f_best
                    f_best = f_trial;
                    x_best = trial;
                end
            end
        end

        %% Update population and archive
        for i = 1:N
            if accept(i)
                archive(end+1,:) = pop(i,:); %#ok<AGROW>
                pop(i,:) = trial_pop(i,:);
                fit(i)   = trial_fit(i);
            end
        end

        %% Trim archive
        if size(archive,1) > arc_max
            del = randperm(size(archive,1), size(archive,1)-arc_max);
            archive(del,:) = [];
        end

        %% Update history with Lehmer mean
        if ~isempty(S_CR)
            if sum(delta_f) > 0
                w = delta_f / sum(delta_f);
            else
                w = ones(size(delta_f)) / numel(delta_f);
            end
            % Weighted Lehmer mean
            if sum(w .* S_CR) > 0
                mu_CR(hist_idx) = sum(w .* S_CR.^2) / sum(w .* S_CR);
            end
            mu_F(hist_idx) = sum(w .* S_F.^2) / sum(w .* S_F);
            hist_idx = mod(hist_idx, H) + 1;
        end

        %% Linear population size reduction
        if N_new < N
            [~, sorted_idx] = sort(fit(1:N));
            keep = sorted_idx(1:N_new);
            pop  = pop(keep,:);
            fit  = fit(keep);
            N    = N_new;
            arc_max = round(arc_rate * N);
            if size(archive,1) > arc_max
                del = randperm(size(archive,1), size(archive,1)-arc_max);
                archive(del,:) = [];
            end
        end
    end

    %% Fill remainder of convergence vector
    convergence_full(NFE+1:end) = f_best;

    %% Downsample to T_max iterations (for compatibility with run_01 format)
    % run_01 records one value per iteration; here we record per NFE.
    % Return the full NFE-length vector and let the caller handle it,
    % OR downsample to exactly T_max points.
    idx_ds = round(linspace(1, T_max, T_max));
    convergence = convergence_full(idx_ds);
end
