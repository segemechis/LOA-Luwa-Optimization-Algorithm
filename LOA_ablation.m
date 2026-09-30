function [x_best, f_best, convergence, convergence_cv, FE_count] = ...
         LOA_ablation(obj_func, lb, ub, N_g, FE_max, options)
% =========================================================================
%  LOA_ABLATION — Luwa Optimization Algorithm (Ablation Variant)
%
%  Identical to LOA.m in every respect except two additional options:
%
%    options.disable_yakka  (logical, default false)
%      When true, the Yakka-lite Qqarubba diversity-veto block is
%      completely skipped.  All other mechanisms are unchanged.
%
%    options.disable_garo   (logical, default false)
%      When true, the Garo grade-restart block is completely skipped.
%      All other mechanisms are unchanged.
%
%  These flags allow systematic ablation of the two mechanisms that the
%  paper claims are responsible for escape from spatially separated local
%  optima.  Running four variants:
%    LOA-Full     disable_yakka=false, disable_garo=false
%    LOA-NoYakka  disable_yakka=true,  disable_garo=false
%    LOA-NoGaro   disable_yakka=false, disable_garo=true
%    LOA-None     disable_yakka=true,  disable_garo=true
%  isolates the causal contribution of each mechanism.
%
%  All signatures, parameters, outputs and internal logic match LOA.m
%  exactly.  Do not use this file for the main benchmarking experiments;
%  use LOA.m there.
%
%  Author : Sena Gemechis File, Jimma Institute of Technology
% =========================================================================

    % ------------------------------------------------------------------ %
    %  0. DEFAULTS & PARAMETER SETUP                                       %
    % ------------------------------------------------------------------ %
    if nargin < 4 || isempty(N_g),    N_g    = 10;     end
    if nargin < 5 || isempty(FE_max), FE_max = 300000; end
    if nargin < 6,                    options = struct(); end

    T_L_init   = get_opt(options, 'T_L_init',    7);
    T_L_min    = get_opt(options, 'T_L_min',     4);
    T_L_max    = get_opt(options, 'T_L_max',    12);
    T_enc      = get_opt(options, 'T_enc',       3);
    NY         = get_opt(options, 'NY',         10);
    T_seera    = get_opt(options, 'T_seera',     5);
    T_exile    = get_opt(options, 'T_exile',     3);
    T_garo     = get_opt(options, 'T_garo',     20);
    eps_impr   = get_opt(options, 'eps_improve', 1e-6);
    D_frac     = get_opt(options, 'D_thresh',    0.01);
    n_clusters = get_opt(options, 'n_clusters',  3);
    beta       = get_opt(options, 'beta',        0.1);
    rng_seed   = get_opt(options, 'seed',        []);

    % ---- ABLATION FLAGS ------------------------------------------------ %
    disable_yakka = get_opt(options, 'disable_yakka', false);
    disable_garo  = get_opt(options, 'disable_garo',  false);

    if ~isempty(rng_seed), rng(rng_seed); end

    lb = lb(:)';  ub = ub(:)';
    d  = numel(lb);

    D_search = norm(ub - lb);
    D_thresh = D_frac * D_search;

    N_GRADES    = 5;

    FE_count = 0;
    safe_eval = @(x) counted_eval(x, obj_func);

    % ------------------------------------------------------------------ %
    %  1. INITIALIZATION                                                   %
    % ------------------------------------------------------------------ %
    max_records    = ceil(FE_max / N_g) + 2;
    convergence    = zeros(max_records, 1);
    convergence_cv = zeros(max_records, 1);
    rec_ptr        = 0;
    next_record_FE = N_g;
    f_best  = Inf;
    cv_best = Inf;

    pop = struct();
    for g = 1:N_GRADES
        if isfield(options,'X0') && ~isempty(options.X0) && ...
                size(options.X0,1) >= g*N_g
            pop(g).X = options.X0((g-1)*N_g+1 : g*N_g, :);
        else
            pop(g).X           = lhs_init(N_g, d, lb, ub);
        end
        pop(g).F           = inf(N_g, 1);
        pop(g).CV          = inf(N_g, 1);
        pop(g).exile_count = zeros(N_g, 1);
        pop(g).viol_streak = zeros(N_g, 1);
        for j = 1:N_g
            [pop(g).F(j), pop(g).CV(j)] = safe_eval(pop(g).X(j,:));
            if deb_compare(pop(g).F(j), pop(g).CV(j), f_best, cv_best)
                f_best  = pop(g).F(j);
                cv_best = pop(g).CV(j);
                x_best  = pop(g).X(j,:);
            end
        end
    end

    Yuba.X  = zeros(0, d);
    Yuba.F  = zeros(0, 1);
    Yuba.CV = zeros(0, 1);

    x_best  = pop(1).X(1,:);
    [x_best, f_best, cv_best] = find_global_best(pop, N_GRADES);

    ruling_grade = 5;
    [gadaana_x, gadaana_f, gadaana_cv] = grade_best_valid(pop(ruling_grade));

    T_L         = T_L_init;
    t_since_rot = 0;
    f_at_start  = f_best;

    stag_count = 0;
    f_prev     = f_best;

    gadaana_incoming_x  = [];
    gadaana_incoming_f  = Inf;
    gadaana_incoming_cv = Inf;

    rec_ptr = rec_ptr + 1;
    convergence(rec_ptr)    = f_best;
    convergence_cv(rec_ptr) = cv_best;

    % ------------------------------------------------------------------ %
    %  2. MAIN LOOP                                                        %
    % ------------------------------------------------------------------ %
    while FE_count < FE_max

        alpha_t = max(2 * (1 - FE_count / FE_max), 0.02);

        % -------------------------------------------------------------- %
        %  2a. SONGO                                                       %
        % -------------------------------------------------------------- %
        X_pool = [];
        for g_p = 1:N_GRADES
            valid = pop(g_p).exile_count <= 0;
            X_pool = [X_pool; pop(g_p).X(valid, :)]; %#ok<AGROW>
        end
        if ~isempty(Yuba.X)
            X_pool = [X_pool; Yuba.X];
        end
        n_pool = size(X_pool, 1);

        for g = 1:N_GRADES
            x_centroid = mean(pop(g).X, 1);
            prev_g = g - 1;  if prev_g < 1, prev_g = N_GRADES; end
            next_g = mod(g, N_GRADES) + 1;

            for j = 1:N_g
                if pop(g).exile_count(j) > 0
                    pop(g).exile_count(j) = pop(g).exile_count(j) - 1;
                    continue;
                end

                x_old  = pop(g).X(j,:);
                f_old  = pop(g).F(j);
                cv_old = pop(g).CV(j);

                switch g
                    case 1
                        levy_step = levy_flight(d);
                        sigma_d   = alpha_t * D_search / sqrt(d);
                        x_new = x_centroid + sigma_d * levy_step;
                        x_new = reflect_bounds(x_new, lb, ub);

                    case 2
                        if rand < 0.5
                            rival_g = next_g;
                        else
                            rival_g = prev_g;
                        end
                        [xr_best,~,~] = grade_best_valid(pop(rival_g), x_best, f_best, cv_best);
                        if n_pool >= 3
                            idx = randperm(n_pool, 3);
                            xr1 = X_pool(idx(1),:);
                            xr2 = X_pool(idx(2),:);
                        else
                            xr1 = lb + rand(1,d).*(ub-lb);
                            xr2 = lb + rand(1,d).*(ub-lb);
                        end
                        F_h   = 0.5 + 0.3 * rand;
                        x_mut = xr1 + F_h * (xr_best - xr2);
                        x_new = binomial_crossover(x_old, x_mut, 0.7, d);
                        x_new = reflect_bounds(x_new, lb, ub);

                    case 3
                        F_m = 0.5 + 0.3 * rand;
                        if ~isempty(Yuba.X)
                            y_idx  = randi(size(Yuba.X, 1));
                            x_yuba = Yuba.X(y_idx,:);
                        else
                            x_yuba = x_best;
                        end
                        if n_pool >= 2
                            idx = randperm(n_pool, 2);
                            xr1 = X_pool(idx(1),:);
                            xr2 = X_pool(idx(2),:);
                        else
                            xr1 = lb + rand(1,d).*(ub-lb);
                            xr2 = lb + rand(1,d).*(ub-lb);
                        end
                        x_mut = x_old + F_m * (x_best - x_old) + ...
                                beta  * (x_yuba - x_old) + ...
                                F_m   * (xr1 - xr2);
                        x_new = binomial_crossover(x_old, x_mut, 0.8, d);
                        x_new = reflect_bounds(x_new, lb, ub);

                    case 4
                        r1 = rand; r2 = rand;
                        if ~isempty(Yuba.X)
                            y_idx  = randi(size(Yuba.X, 1));
                            x_yuba = Yuba.X(y_idx,:);
                        else
                            x_yuba = gadaana_x;
                        end
                        x_new = x_old ...
                              + r1 * alpha_t * (gadaana_x - x_old) ...
                              + r2 * beta    * (x_yuba    - x_old);
                        x_new = reflect_bounds(x_new, lb, ub);

                    case 5
                        r1 = rand;
                        x_new = x_old ...
                              + r1 * alpha_t * (gadaana_x - x_old) ...
                              + 0.01 * D_search * randn(1, d);
                        x_new = reflect_bounds(x_new, lb, ub);
                end

                [f_new, cv_new] = safe_eval(x_new);
                if FE_count >= FE_max, break; end

                if cv_new > 1e-10
                    pop(g).viol_streak(j) = pop(g).viol_streak(j) + 1;
                else
                    pop(g).viol_streak(j) = 0;
                end

                if pop(g).viol_streak(j) >= T_seera
                    pop(g).exile_count(j) = T_exile;
                    pop(g).viol_streak(j) = 0;
                    continue;
                end

                if deb_compare(f_new, cv_new, f_old, cv_old)
                    pop(g).X(j,:) = x_new;
                    pop(g).F(j)   = f_new;
                    pop(g).CV(j)  = cv_new;
                end
            end
            if FE_count >= FE_max, break; end

            if g == ruling_grade && (FE_count < 0.7 * FE_max)
                for j = 1:N_g
                    if FE_count >= FE_max, break; end
                    if pop(g).exile_count(j) > 0, continue; end
                    x_opp = max(lb, min(ub, lb + ub - pop(g).X(j,:)));
                    [f_opp, cv_opp] = safe_eval(x_opp);
                    if deb_compare(f_opp, cv_opp, pop(g).F(j), pop(g).CV(j))
                        pop(g).X(j,:) = x_opp;
                        pop(g).F(j)   = f_opp;
                        pop(g).CV(j)  = cv_opp;
                    end
                end
            end
        end
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2b. YAKKA-LITE  (skipped if disable_yakka = true)              %
        % -------------------------------------------------------------- %
        if ~disable_yakka
            for g = 1:N_GRADES
                if FE_count >= FE_max, break; end
                D_g = mean(vecnorm(pop(g).X - mean(pop(g).X, 1), 2, 2));
                if D_g < D_thresh
                    n_reinit  = ceil(N_g / 2);
                    comp_f    = pop(g).F + 1e12 * (pop(g).CV > 1e-10);
                    [~, sidx] = sort(comp_f, 'descend');
                    for wi = 1:n_reinit
                        if FE_count >= FE_max, break; end
                        j = sidx(wi);
                        pop(g).X(j,:) = lb + rand(1,d) .* (ub - lb);
                        [pop(g).F(j), pop(g).CV(j)] = safe_eval(pop(g).X(j,:));
                        pop(g).exile_count(j) = 0;
                        pop(g).viol_streak(j) = 0;
                    end
                end
            end
        end
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2c. UPDATE Gaadaana                                             %
        % -------------------------------------------------------------- %
        [x_cand, f_cand, cv_cand] = find_global_best(pop, N_GRADES);
        if deb_compare(f_cand, cv_cand, f_best, cv_best)
            x_best  = x_cand;
            f_best  = f_cand;
            cv_best = cv_cand;
        end

        [g_x, g_f, g_cv] = grade_best_valid(pop(ruling_grade));
        if deb_compare(g_f, g_cv, gadaana_f, gadaana_cv)
            gadaana_x  = g_x;
            gadaana_f  = g_f;
            gadaana_cv = g_cv;
        end

        % -------------------------------------------------------------- %
        %  2d. GADAANA PRE-SELECTION                                       %
        % -------------------------------------------------------------- %
        if t_since_rot == T_L - 3
            incoming_pre = mod(ruling_grade, N_GRADES) + 1;
            [gadaana_incoming_x, gadaana_incoming_f, gadaana_incoming_cv] = ...
                grade_best_valid(pop(incoming_pre));
        end

        % -------------------------------------------------------------- %
        %  2e. GARO RESTART  (skipped if disable_garo = true)             %
        % -------------------------------------------------------------- %
        if abs(f_best - f_prev) < eps_impr * (abs(f_prev) + eps)
            stag_count = stag_count + 1;
        else
            stag_count = 0;
        end
        f_prev = f_best;

        if ~disable_garo && (stag_count >= T_garo)
            avg_f = zeros(N_GRADES, 1);
            for g = 1:N_GRADES
                if g == ruling_grade
                    avg_f(g) = -Inf;
                    continue;
                end
                feas_f = pop(g).F(pop(g).CV <= 1e-10);
                if isempty(feas_f)
                    avg_f(g) = mean(pop(g).F) + 1e6 * mean(pop(g).CV);
                else
                    avg_f(g) = mean(feas_f);
                end
            end
            [~, worst_g] = max(avg_f);
            pop(worst_g).X           = lhs_init(N_g, d, lb, ub);
            pop(worst_g).F           = inf(N_g, 1);
            pop(worst_g).CV          = inf(N_g, 1);
            pop(worst_g).exile_count = zeros(N_g, 1);
            pop(worst_g).viol_streak = zeros(N_g, 1);
            n_inject = 0;
            if ~isempty(Yuba.X)
                n_inject = min(floor(N_g / 3), size(Yuba.X, 1));
                pop(worst_g).X(1:n_inject, :) = Yuba.X(1:n_inject, :);
                pop(worst_g).F(1:n_inject)    = Yuba.F(1:n_inject);
                pop(worst_g).CV(1:n_inject)   = Yuba.CV(1:n_inject);
            end
            for j = (n_inject + 1):N_g
                if FE_count >= FE_max, break; end
                [pop(worst_g).F(j), pop(worst_g).CV(j)] = ...
                    safe_eval(pop(worst_g).X(j,:));
            end
            stag_count = 0;
        end
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2f. LUWA ROTATION                                               %
        % -------------------------------------------------------------- %
        t_since_rot = t_since_rot + 1;

        if t_since_rot >= T_L
            improvement = f_at_start - f_best;
            if improvement > eps_impr * (abs(f_at_start) + eps)
                T_L = min(T_L + 1, T_L_max);
            else
                T_L = max(T_L - 1, T_L_min);
            end

            [best_x, best_f, best_cv] = grade_best_valid(pop(ruling_grade));
            Yuba = yuba_insert(Yuba, best_x, best_f, best_cv, NY);

            incoming = mod(ruling_grade, N_GRADES) + 1;

            n_seed = ceil(0.3 * N_g);
            comp_out = pop(ruling_grade).F + ...
                       1e12 * (pop(ruling_grade).CV > 1e-10);
            [~, out_sort] = sort(comp_out);
            top_out = out_sort(1:n_seed);

            comp_in = pop(incoming).F + ...
                      1e12 * (pop(incoming).CV > 1e-10);
            [~, in_sort] = sort(comp_in, 'descend');
            bot_in = in_sort(1:n_seed);

            sigma_seed = 0.05 * D_search;
            for si = 1:n_seed
                if FE_count >= FE_max, break; end
                x_seed = pop(ruling_grade).X(top_out(si),:) + ...
                         sigma_seed * randn(1, d);
                x_seed = reflect_bounds(x_seed, lb, ub);
                [f_seed, cv_seed] = safe_eval(x_seed);
                ji = bot_in(si);
                pop(incoming).X(ji,:) = x_seed;
                pop(incoming).F(ji)   = f_seed;
                pop(incoming).CV(ji)  = cv_seed;
                pop(incoming).exile_count(ji) = 0;
                pop(incoming).viol_streak(ji)  = 0;
            end

            if FE_count < FE_max
                [pop(incoming), x_best, f_best, cv_best] = ...
                    encampment(pop(incoming), N_g, d, lb, ub, ...
                               n_clusters, T_enc, x_best, f_best, cv_best, ...
                               alpha_t, safe_eval, FE_max, FE_count);
            end

            if ~isempty(gadaana_incoming_x)
                gadaana_x  = gadaana_incoming_x;
                gadaana_f  = gadaana_incoming_f;
                gadaana_cv = gadaana_incoming_cv;
            else
                [gadaana_x, gadaana_f, gadaana_cv] = ...
                    grade_best_valid(pop(incoming));
            end
            gadaana_incoming_x = [];

            ruling_grade = incoming;
            t_since_rot  = 0;
            f_at_start   = f_best;
        end

    end % main while loop

    convergence    = convergence(1:rec_ptr);
    convergence_cv = convergence_cv(1:rec_ptr);

    % ------------------------------------------------------------------ %
    %  NESTED FUNCTION                                                     %
    % ------------------------------------------------------------------ %
    function [f, cv] = counted_eval(x, func)
        f  = Inf;  cv = Inf;
        try
            [f, cv] = func(x);
            cv = max(0, cv);
        catch
            try
                f  = func(x);
                cv = 0;
            catch
                f  = Inf;  cv = Inf;
            end
        end
        if ~isfinite(f),  f  = Inf; end
        if ~isfinite(cv), cv = Inf; end
        FE_count = FE_count + 1;

        while FE_count >= next_record_FE
            rec_ptr = rec_ptr + 1;
            if rec_ptr > max_records
                convergence    = [convergence;    zeros(max_records, 1)];
                convergence_cv = [convergence_cv; zeros(max_records, 1)];
                max_records    = max_records * 2;
            end
            convergence(rec_ptr)    = f_best;
            convergence_cv(rec_ptr) = cv_best;
            next_record_FE          = next_record_FE + N_g;
        end
    end

end  % LOA_ablation


% =========================================================================
%  LOCAL FUNCTIONS  (identical to LOA.m)
% =========================================================================

function better = deb_compare(f1, cv1, f2, cv2)
    tol   = 1e-10;
    feas1 = cv1 <= tol;
    feas2 = cv2 <= tol;
    if feas1 && feas2
        better = f1 < f2;
    elseif feas1
        better = true;
    elseif feas2
        better = false;
    else
        better = cv1 < cv2;
    end
end

function x = reflect_bounds(x, lb, ub)
    x(x < lb) = 2*lb(x < lb) - x(x < lb);
    x(x > ub) = 2*ub(x > ub) - x(x > ub);
    x = max(lb, min(ub, x));
end

function x_trial = binomial_crossover(x_old, x_mut, CR, d)
    j_rand = randi(d);
    mask   = rand(1, d) < CR;
    mask(j_rand) = true;
    x_trial = x_old;
    x_trial(mask) = x_mut(mask);
end

function step = levy_flight(d)
    beta_lv = 1.5;
    sigma_u = (0.75 * sqrt(pi) / (0.906402 * 1.5 * 2^0.25))^(1/beta_lv);
    u    = randn(1, d) * sigma_u;
    v    = randn(1, d);
    step = u ./ (abs(v).^(1/beta_lv));
    step = sign(step) .* min(abs(step), 10);
end

function X = lhs_init(n, d, lb, ub)
    X = zeros(n, d);
    for k = 1:d
        perm   = randperm(n);
        X(:,k) = lb(k) + (perm' - rand(n,1)) / n * (ub(k) - lb(k));
    end
end

function [x_gb, f_gb, cv_gb] = grade_best_valid(grade_pop, x_fallback, f_fallback, cv_fallback)
    n    = size(grade_pop.X, 1);
    f_gb = Inf;  cv_gb = Inf;  x_gb = grade_pop.X(1,:);
    for j = 1:n
        if grade_pop.exile_count(j) > 0, continue; end
        if deb_compare(grade_pop.F(j), grade_pop.CV(j), f_gb, cv_gb)
            f_gb  = grade_pop.F(j);
            cv_gb = grade_pop.CV(j);
            x_gb  = grade_pop.X(j,:);
        end
    end
    if isinf(f_gb)
        if nargin >= 4 && ~isempty(x_fallback)
            x_gb  = x_fallback;
            f_gb  = f_fallback;
            cv_gb = cv_fallback;
        else
            for j = 1:n
                if deb_compare(grade_pop.F(j), grade_pop.CV(j), f_gb, cv_gb)
                    f_gb  = grade_pop.F(j);
                    cv_gb = grade_pop.CV(j);
                    x_gb  = grade_pop.X(j,:);
                end
            end
        end
    end
end

function [x_gb, f_gb, cv_gb] = find_global_best(pop, N_GRADES)
    f_gb = Inf;  cv_gb = Inf;  x_gb = pop(1).X(1,:);
    for g = 1:N_GRADES
        for j = 1:size(pop(g).X, 1)
            if pop(g).exile_count(j) > 0, continue; end
            if deb_compare(pop(g).F(j), pop(g).CV(j), f_gb, cv_gb)
                f_gb  = pop(g).F(j);
                cv_gb = pop(g).CV(j);
                x_gb  = pop(g).X(j,:);
            end
        end
    end
    if isinf(f_gb)
        for g = 1:N_GRADES
            for j = 1:size(pop(g).X, 1)
                if deb_compare(pop(g).F(j), pop(g).CV(j), f_gb, cv_gb)
                    f_gb  = pop(g).F(j);
                    cv_gb = pop(g).CV(j);
                    x_gb  = pop(g).X(j,:);
                end
            end
        end
    end
end

function Yuba = yuba_insert(Yuba, x_new, f_new, cv_new, NY)
    if isempty(Yuba.X)
        Yuba.X  = x_new;
        Yuba.F  = f_new;
        Yuba.CV = cv_new;
    else
        n = size(Yuba.X, 1);
        if n < NY
            Yuba.X  = [Yuba.X;  x_new];
            Yuba.F  = [Yuba.F;  f_new];
            Yuba.CV = [Yuba.CV; cv_new];
        else
            worst_idx = 1;
            for i = 2:n
                if deb_compare(Yuba.F(worst_idx), Yuba.CV(worst_idx), ...
                               Yuba.F(i), Yuba.CV(i))
                    worst_idx = i;
                end
            end
            if deb_compare(f_new, cv_new, Yuba.F(worst_idx), Yuba.CV(worst_idx))
                Yuba.X(worst_idx,:) = x_new;
                Yuba.F(worst_idx)   = f_new;
                Yuba.CV(worst_idx)  = cv_new;
            end
        end
    end
    sort_key = Yuba.F;
    sort_key(Yuba.CV > 1e-10) = sort_key(Yuba.CV > 1e-10) + 1e12;
    [~, ord] = sort(sort_key);
    Yuba.X   = Yuba.X(ord,:);
    Yuba.F   = Yuba.F(ord);
    Yuba.CV  = Yuba.CV(ord);
end

function [grade_pop, x_best, f_best, cv_best] = encampment( ...
             grade_pop, N_g, d, lb, ub, n_clusters, T_enc, ...
             x_best, f_best, cv_best, alpha_t, safe_eval, FE_max, FE_count)
    FE_local = FE_count;
    n_clust = min(n_clusters, N_g);
    c_idx   = randperm(N_g, n_clust);
    centres = grade_pop.X(c_idx, :);
    labels  = zeros(N_g, 1);
    for km_iter = 1:10
        for j = 1:N_g
            dists = vecnorm(centres - grade_pop.X(j,:), 2, 2);
            [~, labels(j)] = min(dists);
        end
        for c = 1:n_clust
            mem = find(labels == c);
            if ~isempty(mem)
                centres(c,:) = mean(grade_pop.X(mem,:), 1);
            end
        end
    end
    done = false;
    for enc_iter = 1:T_enc
        if done, break; end
        for c = 1:n_clust
            if done, break; end
            members = find(labels == c);
            if isempty(members), continue; end
            c_cen = mean(grade_pop.X(members,:), 1);
            for j = members'
                if FE_local >= FE_max
                    done = true;
                    break;
                end
                r1 = rand;
                x_new = grade_pop.X(j,:) ...
                      + r1     * alpha_t * (c_cen   - grade_pop.X(j,:)) ...
                      + (1-r1) * 0.5 * alpha_t * (x_best - grade_pop.X(j,:));
                x_new = reflect_bounds(x_new, lb, ub);
                [f_new, cv_new] = safe_eval(x_new);
                FE_local = FE_local + 1;
                if deb_compare(f_new, cv_new, grade_pop.F(j), grade_pop.CV(j))
                    grade_pop.X(j,:) = x_new;
                    grade_pop.F(j)   = f_new;
                    grade_pop.CV(j)  = cv_new;
                    if deb_compare(f_new, cv_new, f_best, cv_best)
                        x_best  = x_new;
                        f_best  = f_new;
                        cv_best = cv_new;
                    end
                end
            end
        end
    end
end

function v = get_opt(s, field, default)
    if isfield(s, field) && ~isempty(s.(field))
        v = s.(field);
    else
        v = default;
    end
end
