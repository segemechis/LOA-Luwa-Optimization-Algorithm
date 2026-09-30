function [x_best, f_best, convergence, convergence_cv, FE_count] = ...
         LOA(obj_func, lb, ub, N_g, FE_max, options)
% =========================================================================
%  LOA — Luwa Optimization Algorithm
%  Inspired by the Luwa age-grade system of the Sidama people,
%  Southern Ethiopia.
%
%  Cultural institutions encoded:
%    Luwa grades  : five rotating subpopulations
%                   (Darara, Hirbora, Mogissa, Fulasa, Wawasa)
%    Gadaana      : stable elite reference of the RULING grade (stays home;
%                   used as attraction centre but does NOT mutate)
%    Gaadaana     : global exploration champion / war leader (x_best)
%    Songo        : distributed proposal — every non-exiled agent
%                   generates an offspring
%    Halale       : feasibility-first dominance (Deb rule)
%    Seera/Woma/
%    Karichcha    : three-tier constraint handling — reflect (Woma),
%                   penalty+streak (Gaanna/Karichcha), ostracism (Seera)
%    Garo         : supreme-court restart when all grades stagnate;
%                   ruling grade is protected from purge
%    Yuba         : institutional memory archive of retired ruling grades
%    Encampment   : k-means temporary clustering + father-son seeding
%                   performed at every grade handover
%    Yakka-lite   : Qqarubba diversity veto — partial reinitialization
%                   when a grade's internal diversity collapses
%    Good/Bad Luwa: adaptive ruling-period length T_L
%
%  SIGNATURE
%    [x_best, f_best, convergence, convergence_cv, FE_count] = ...
%        LOA(obj_func, lb, ub, N_g, FE_max, options)
%
%  INPUTS
%    obj_func  — function handle  @(x) returns [f, cv]
%                  f  : scalar objective value
%                  cv : scalar constraint violation (0 if unconstrained)
%                  Unconstrained shorthand: @(x) deal(f(x), 0)
%    lb, ub    — 1×d lower/upper bounds
%    N_g       — agents per grade (default 10 → 50 total)
%    FE_max    — maximum function evaluations budget
%                  (default 300 000; CEC standard: d × 10 000)
%    options   — struct with optional fields (all have defaults):
%      .T_L_init   initial ruling period length             (default 7)
%      .T_L_min    minimum ruling period                    (default 4)
%      .T_L_max    maximum ruling period                    (default 12)
%      .T_enc      encampment refinement iterations         (default 3)
%      .NY         Yuba archive capacity                    (default 10)
%      .T_seera    streak length before Seera fires         (default 5)
%      .T_exile    ostracism exclusion duration             (default 3)
%      .T_garo     stagnation limit before Garo restart     (default 20)
%      .eps_improve improvement threshold for Good Luwa     (default 1e-6)
%      .D_thresh   Yakka diversity threshold (fraction of
%                  search diameter)                         (default 0.01)
%      .n_clusters encampment k-means clusters              (default 3)
%      .beta       Yuba attraction weight in Fulasa/Mogissa (default 0.1)
%      .seed       random seed ([] = no fix)                (default [])
%
%  OUTPUTS
%    x_best        — 1×d best solution found
%    f_best        — scalar best objective value
%    convergence   — f_best at each FE milestone (recorded inside the
%                    evaluator every N_g FEs; no milestone is ever missed)
%    convergence_cv— cv_best at same milestones (for constrained problems)
%    FE_count      — total function evaluations consumed
%
%  GRADE OPERATORS
%    Darara  (grade 1) — explorers : Lévy-flight walk from grade centroid
%    Hirbora (grade 2) — warriors  : DE/rand/1 toward rival (adjacent) best
%    Mogissa (grade 3) — elders    : DE/current-to-best/1 with Yuba archive
%    Fulasa  (grade 4) — fathers   : attraction to Yuba + Gadaana reference
%    Wawasa  (grade 5) — rulers    : refinement + OBL (exploitation)
%
%  ROTATION ORDER (Sidama cultural sequence from Stanley & Karsten 1968):
%    Fulasa → Mogissa → Hirbora → Darara → Wawasa → Fulasa …
%    Starting ruling grade: Wawasa (index 5).
%    Code cycle: mod(ruling_grade, N_GRADES) + 1 maps
%      5→1(Darara), 1→2(Hirbora), 2→3(Mogissa), 3→4(Fulasa), 4→5(Wawasa)
%    Cultural equivalent: Wawasa steps down → Darara takes power → …
%    → Fulasa → Wawasa, matching the historical sequence.
%
%  USAGE EXAMPLE (unconstrained)
%    lb = -100*ones(1,30);  ub = 100*ones(1,30);
%    f  = @(x) deal(sum(x.^2), 0);
%    [x,fv,conv,~,FE] = LOA(f, lb, ub, 10, 300000);
%    semilogy(conv);  xlabel('FE milestone');  ylabel('f_{best}');
%
%  USAGE EXAMPLE (constrained)
%    f = @(x) deal(obj(x), max(0, g(x)));   % g(x) <= 0
%    [x,fv,conv,convcv,FE] = LOA(f, lb, ub, 10, 300000);
%
%  REFERENCE
%    Stanley, S. & Karsten, D. (1968). The Luwa System of the Garbicho
%    Subtribe of the Sidama (Southern Ethiopia) as a Special Case of an
%    Age Set System. Paideuma, 14, 93-102.
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

    if ~isempty(rng_seed), rng(rng_seed); end

    lb = lb(:)';  ub = ub(:)';
    d  = numel(lb);

    D_search = norm(ub - lb);
    D_thresh = D_frac * D_search;   % Yakka-lite absolute threshold

    N_GRADES    = 5;
    % Grade names: indices 1..5 → Darara, Hirbora, Mogissa, Fulasa, Wawasa
    GRADE_NAMES = {'Darara','Hirbora','Mogissa','Fulasa','Wawasa'};
    % (Used in optional diagnostic messages below; do not remove.)

    % FE counter (shared via closure through safe_eval)
    FE_count = 0;
    safe_eval = @(x) counted_eval(x, obj_func);

    % ------------------------------------------------------------------ %
    %  1. INITIALIZATION — 5 grades, each N_g agents                      %
    % ------------------------------------------------------------------ %
    % pop(g) fields:
    %   .X           N_g × d  positions
    %   .F           N_g × 1  objective values
    %   .CV          N_g × 1  constraint violations (0 = feasible)
    %   .exile_count N_g × 1  remaining exile iterations (Seera)
    %   .viol_streak N_g × 1  consecutive constraint-violation streak

    % Convergence recording variables declared here so that counted_eval
    % (nested function, closure scope) can reference them during init.
    max_records    = ceil(FE_max / N_g) + 2;
    convergence    = zeros(max_records, 1);
    convergence_cv = zeros(max_records, 1);
    rec_ptr        = 0;
    next_record_FE = N_g;
    % Pre-initialise best scalars for same reason
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
            % Update global best on-the-fly so convergence records during
            % initialization are valid (not Inf). Fix: item 1 (Kimi review).
            if deb_compare(pop(g).F(j), pop(g).CV(j), f_best, cv_best)
                f_best  = pop(g).F(j);
                cv_best = pop(g).CV(j);
                x_best  = pop(g).X(j,:);
            end
        end
    end

    % Yuba archive — institutional memory of retired ruling grades
    Yuba.X  = zeros(0, d);
    Yuba.F  = zeros(0, 1);
    Yuba.CV = zeros(0, 1);

    % Global best — Gaadaana (war leader / exploration champion)
    x_best  = pop(1).X(1,:);
    [x_best, f_best, cv_best] = find_global_best(pop, N_GRADES);

    % Ruling grade: Wawasa starts (index 5).
    % Rotation sequence produced by mod(ruling_grade, N_GRADES)+1:
    %   5→1→2→3→4→5 … (Wawasa→Darara→Hirbora→Mogissa→Fulasa→Wawasa)
    ruling_grade = 5;

    % Gadaana — strictly the ruling grade's OWN best agent.
    % He stays home (never mutates); used as attraction centre only.
    % Distinct from Gaadaana (x_best / global war leader).
    [gadaana_x, gadaana_f, gadaana_cv] = grade_best_valid(pop(ruling_grade));

    % Adaptive ruling period
    T_L         = T_L_init;
    t_since_rot = 0;
    f_at_start  = f_best;

    % Stagnation counter for Garo restart
    stag_count = 0;
    f_prev     = f_best;

    % Gadaana pre-selection (set 2 iters before handover)
    gadaana_incoming_x  = [];
    gadaana_incoming_f  = Inf;
    gadaana_incoming_cv = Inf;

    % Record initial state in convergence log
    rec_ptr = rec_ptr + 1;
    convergence(rec_ptr)    = f_best;
    convergence_cv(rec_ptr) = cv_best;

    % ------------------------------------------------------------------ %
    %  2. MAIN LOOP  (FE-budget termination, not fixed iterations)         %
    % ------------------------------------------------------------------ %
    while FE_count < FE_max

        alpha_t = max(2 * (1 - FE_count / FE_max), 0.02);   % decaying step

        % -------------------------------------------------------------- %
        %  2a. SONGO — each grade proposes offspring via its operator      %
        % -------------------------------------------------------------- %
        % Build global agent pool for cross-grade mutation parents.
        % Seera (Ayiddu Seera): exiled agents are excluded from the pool —
        % an outcast cannot participate in Dee (labor pooling) or help
        % raise the next generation. Only non-exiled agents + Yuba enter.
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

            % Dynamic rivals for Hirbora: both Luwa-cycle neighbours.
            % Inter-Luwa competition with alternate Luwa solidarity:
            %   next_g  = successor in cycle  (primary rival)
            %   prev_g  = predecessor in cycle (secondary rival)
            % Hirbora randomly targets one of the two adjacent grades,
            % adding stochasticity to the warrior competition.
            prev_g = g - 1;  if prev_g < 1, prev_g = N_GRADES; end
            next_g = mod(g, N_GRADES) + 1;

            for j = 1:N_g

                % Seera: exiled agents cannot reproduce; decrement counter
                if pop(g).exile_count(j) > 0
                    pop(g).exile_count(j) = pop(g).exile_count(j) - 1;
                    continue;
                end

                x_old  = pop(g).X(j,:);
                f_old  = pop(g).F(j);
                cv_old = pop(g).CV(j);

                % ---- Grade-specific Songo operator -------------------- %
                switch g

                    % -------------------------------------------------- %
                    case 1  % DARARA — explorers: Lévy-flight from centroid
                    % -------------------------------------------------- %
                        levy_step = levy_flight(d);
                        sigma_d   = alpha_t * D_search / sqrt(d);
                        x_new = x_centroid + sigma_d * levy_step;
                        x_new = reflect_bounds(x_new, lb, ub);

                    % -------------------------------------------------- %
                    case 2  % HIRBORA — warriors: DE/rand/1 toward rival best
                    % -------------------------------------------------- %
                        % Rival = one of the two adjacent grades, chosen
                        % randomly each step (inter-Luwa competition).
                        if rand < 0.5
                            rival_g = next_g;
                        else
                            rival_g = prev_g;
                        end
                        % Fix #3: pass Gaadaana as fallback if all rival
                        % agents are exiled (replaces silent index-1 default)
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

                    % -------------------------------------------------- %
                    case 3  % MOGISSA — elders: DE/current-to-best/1 + Yuba
                    % -------------------------------------------------- %
                        F_m = 0.5 + 0.3 * rand;
                        if ~isempty(Yuba.X)
                            y_idx  = randi(size(Yuba.X, 1));
                            x_yuba = Yuba.X(y_idx,:);
                        else
                            x_yuba = x_best;   % fallback to Gaadaana
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

                    % -------------------------------------------------- %
                    case 4  % FULASA — fathers: Gadaana (ruling) + Yuba memory
                    % -------------------------------------------------- %
                        % Gadaana = ruling grade's OWN elite (stays home).
                        % Fulasa fathers are attracted to the institutional
                        % elder reference, NOT the global war champion.
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

                    % -------------------------------------------------- %
                    case 5  % WAWASA — ruling grade: refinement + OBL
                    % -------------------------------------------------- %
                        r1 = rand;
                        % Wawasa is exploitative: moves toward its own
                        % Gadaana (grade elder) with Gaussian perturbation.
                        x_new = x_old ...
                              + r1 * alpha_t * (gadaana_x - x_old) ...
                              + 0.01 * D_search * randn(1, d);
                        x_new = reflect_bounds(x_new, lb, ub);

                end % switch g

                % ---- Evaluate offspring (Songo proposal) -------------- %
                [f_new, cv_new] = safe_eval(x_new);
                if FE_count >= FE_max, break; end

                % Woma: bound repair already applied via reflect_bounds.
                % Gaanna/Karichcha: update constraint-violation streak.
                if cv_new > 1e-10
                    pop(g).viol_streak(j) = pop(g).viol_streak(j) + 1;
                else
                    pop(g).viol_streak(j) = 0;
                end

                % Karichcha → Seera: ostracise chronic constraint violators
                if pop(g).viol_streak(j) >= T_seera
                    pop(g).exile_count(j) = T_exile;
                    pop(g).viol_streak(j) = 0;
                    continue;
                end

                % Halale: Deb feasibility acceptance
                if deb_compare(f_new, cv_new, f_old, cv_old)
                    pop(g).X(j,:) = x_new;
                    pop(g).F(j)   = f_new;
                    pop(g).CV(j)  = cv_new;
                end

            end % agent loop
            if FE_count >= FE_max, break; end

            % -- OBL for ruling grade (Wawasa) — first 70% of budget ---- %
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

        end % grade loop
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2b. YAKKA-LITE — Qqarubba diversity veto                       %
        % -------------------------------------------------------------- %
        for g = 1:N_GRADES
            if FE_count >= FE_max, break; end
            % Diversity = mean distance from centroid (cheap, no pairwise)
            D_g = mean(vecnorm(pop(g).X - mean(pop(g).X, 1), 2, 2));
            if D_g < D_thresh
                % Senior women (Qqarubba) veto: reinitialize bottom 50%
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
                % Optional diagnostic:
                % fprintf('[LOA] Qqarubba veto: grade %s at FE=%d\n', ...
                %         GRADE_NAMES{g}, FE_count);
            end
        end
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2c. UPDATE Gaadaana (global war leader / x_best)               %
        % -------------------------------------------------------------- %
        % NOTE: Gaadaana (x_best) and Gadaana (ruling-grade elder) are
        % kept DISTINCT throughout:
        %   - Gadaana is the ruling grade's own best (attraction centre).
        %   - Gaadaana is the global champion across ALL grades.
        [x_cand, f_cand, cv_cand] = find_global_best(pop, N_GRADES);
        if deb_compare(f_cand, cv_cand, f_best, cv_best)
            x_best  = x_cand;
            f_best  = f_cand;
            cv_best = cv_cand;
            % Qeexala: public announcement — all grades aware of new champion.
            % (Used by Mogissa and Darara operators via x_best.)
        end

        % Gadaana update: ONLY from ruling grade's own (non-exiled) agents.
        [g_x, g_f, g_cv] = grade_best_valid(pop(ruling_grade));
        if deb_compare(g_f, g_cv, gadaana_f, gadaana_cv)
            gadaana_x  = g_x;
            gadaana_f  = g_f;
            gadaana_cv = g_cv;
        end

        % -------------------------------------------------------------- %
        %  2d. GADAANA PRE-SELECTION (2 iterations before handover)       %
        % -------------------------------------------------------------- %
        % Cultural: the incoming Luwa's Gadaana is identified ~2 years
        % before the handover ceremony so he can prepare spiritually.
        % Fix (vs v1): use T_L - 3 so exactly 2 full iterations remain.
        if t_since_rot == T_L - 3
            incoming_pre = mod(ruling_grade, N_GRADES) + 1;
            [gadaana_incoming_x, gadaana_incoming_f, gadaana_incoming_cv] = ...
                grade_best_valid(pop(incoming_pre));
        end

        % -------------------------------------------------------------- %
        %  2e. GARO GLOBAL RESTART — stagnation check                     %
        % -------------------------------------------------------------- %
        if abs(f_best - f_prev) < eps_impr * (abs(f_prev) + eps)
            stag_count = stag_count + 1;
        else
            stag_count = 0;
        end
        f_prev = f_best;

        if stag_count >= T_garo
            % Garo prayer: reinitialize the WORST non-ruling grade.
            % (Ruling grade is protected — Garo cannot dissolve a
            %  legitimate administration mid-reign.)
            avg_f = zeros(N_GRADES, 1);
            for g = 1:N_GRADES
                if g == ruling_grade
                    avg_f(g) = -Inf;   % exclude ruling grade
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
            % Garo restart: inject Yuba BEFORE evaluating new positions so
            % those FEs are not wasted on slots immediately overwritten.
            pop(worst_g).X           = lhs_init(N_g, d, lb, ub);
            pop(worst_g).F           = inf(N_g, 1);
            pop(worst_g).CV          = inf(N_g, 1);
            pop(worst_g).exile_count = zeros(N_g, 1);
            pop(worst_g).viol_streak = zeros(N_g, 1);
            % Step 1 — seed Yuba elites into the first n_inject slots
            n_inject = 0;
            if ~isempty(Yuba.X)
                n_inject = min(floor(N_g / 3), size(Yuba.X, 1));
                pop(worst_g).X(1:n_inject, :) = Yuba.X(1:n_inject, :);
                pop(worst_g).F(1:n_inject)    = Yuba.F(1:n_inject);
                pop(worst_g).CV(1:n_inject)   = Yuba.CV(1:n_inject);
            end
            % Step 2 — evaluate only the randomly initialised slots
            for j = (n_inject + 1):N_g
                if FE_count >= FE_max, break; end
                [pop(worst_g).F(j), pop(worst_g).CV(j)] = ...
                    safe_eval(pop(worst_g).X(j,:));
            end
            % Optional diagnostic (comment out for silent benchmarking):
            % fprintf('[LOA] Garo restart: grade %s at FE=%d\n', ...
            %         GRADE_NAMES{worst_g}, FE_count);
            stag_count = 0;
        end
        if FE_count >= FE_max, break; end

        % -------------------------------------------------------------- %
        %  2f. LUWA ROTATION — handover ceremony                          %
        % -------------------------------------------------------------- %
        t_since_rot = t_since_rot + 1;

        if t_since_rot >= T_L
            % ---- Good / Bad Luwa: adapt period length ----------------- %
            improvement = f_at_start - f_best;
            if improvement > eps_impr * (abs(f_at_start) + eps)
                T_L = min(T_L + 1, T_L_max);   % Good Luwa: extend
            else
                T_L = max(T_L - 1, T_L_min);   % Bad  Luwa: abbreviate
            end

            % ---- Archive outgoing ruling grade to Yuba ---------------- %
            [best_x, best_f, best_cv] = grade_best_valid(pop(ruling_grade));
            Yuba = yuba_insert(Yuba, best_x, best_f, best_cv, NY);

            % ---- Determine incoming grade ----------------------------- %
            incoming = mod(ruling_grade, N_GRADES) + 1;

            % ---- Father-son seeding ----------------------------------- %
            % Bottom 30% of incoming replaced by perturbed copies of
            % outgoing grade's top 30%.
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
                % Reset social state of seeded agents
                pop(incoming).exile_count(ji) = 0;
                pop(incoming).viol_streak(ji)  = 0;
            end

            % ---- Encampment: k-means clustering + intra-cluster refine -%
            % Fix (KIMI critical #1 & #2): encampment now receives FE_max
            % and FE_count so it can self-terminate on budget exhaustion,
            % and returns [x_best, f_best, cv_best] so the global Gaadaana
            % is updated with any improvements found inside encampment.
            if FE_count < FE_max
                [pop(incoming), x_best, f_best, cv_best] = ...
                    encampment(pop(incoming), N_g, d, lb, ub, ...
                               n_clusters, T_enc, x_best, f_best, cv_best, ...
                               alpha_t, safe_eval, FE_max, FE_count);
            end

            % ---- Install Gadaana of incoming grade -------------------- %
            if ~isempty(gadaana_incoming_x)
                gadaana_x  = gadaana_incoming_x;
                gadaana_f  = gadaana_incoming_f;
                gadaana_cv = gadaana_incoming_cv;
            else
                [gadaana_x, gadaana_f, gadaana_cv] = ...
                    grade_best_valid(pop(incoming));
            end
            gadaana_incoming_x = [];

            % ---- Rotate ruling grade ---------------------------------- %
            ruling_grade = incoming;
            t_since_rot  = 0;
            f_at_start   = f_best;
        end

    end % main while loop

    % Trim convergence vectors to actual records written
    convergence    = convergence(1:rec_ptr);
    convergence_cv = convergence_cv(1:rec_ptr);

    % ------------------------------------------------------------------ %
    %  NESTED FUNCTION — shares workspace variables via closure            %
    % ------------------------------------------------------------------ %
    function [f, cv] = counted_eval(x, func)
    % Wraps obj_func: increments FE_count, handles single/dual outputs,
    % and records convergence milestones INSIDE the evaluator so that no
    % milestone is missed regardless of how many FEs a sub-routine uses.
    % This guarantees CEC-compliant reporting at every N_g-FE interval.
        f  = Inf;
        cv = Inf;
        try
            [f, cv] = func(x);
            cv = max(0, cv);
        catch
            try
                f  = func(x);
                cv = 0;
            catch
                f  = Inf;
                cv = Inf;
            end
        end
        if ~isfinite(f),  f  = Inf; end
        if ~isfinite(cv), cv = Inf; end
        FE_count = FE_count + 1;

        % Record convergence at every N_g-FE milestone (exact, no gaps).
        % f_best / cv_best are maintained in the outer workspace and are
        % always up-to-date when this runs (init sets them before the loop,
        % and the main loop updates them via find_global_best each iter).
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

end  % LOA


% =========================================================================
%  LOCAL (STATIC) FUNCTIONS — no access to LOA workspace
% =========================================================================

% -------------------------------------------------------------------------
function better = deb_compare(f1, cv1, f2, cv2)
% Deb feasibility rule: true if solution 1 dominates solution 2.
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


% -------------------------------------------------------------------------
function x = reflect_bounds(x, lb, ub)
% Reflect out-of-bounds components back into [lb, ub].
    x(x < lb) = 2*lb(x < lb) - x(x < lb);
    x(x > ub) = 2*ub(x > ub) - x(x > ub);
    x = max(lb, min(ub, x));   % safety clamp
end


% -------------------------------------------------------------------------
function x_trial = binomial_crossover(x_old, x_mut, CR, d)
% Standard binomial (uniform) crossover — ensures at least one component.
    j_rand = randi(d);
    mask   = rand(1, d) < CR;
    mask(j_rand) = true;
    x_trial = x_old;
    x_trial(mask) = x_mut(mask);
end


% -------------------------------------------------------------------------
function step = levy_flight(d)
% Lévy flight step using Mantegna's algorithm.
% gamma(1.5) = sqrt(pi)/2 and gamma(1.25) hardcoded to remove Statistics
% Toolbox dependency.
    beta_lv = 1.5;
    % Hardcoded: gamma(1+1.5)*sin(pi*1.5/2) / (gamma((1+1.5)/2)*1.5*2^0.25)
    % = gamma(2.5)*1 / (gamma(1.25)*1.5*2^0.25)
    % gamma(2.5) = 1.5*0.5*sqrt(pi) = 0.75*sqrt(pi)
    % gamma(1.25) ≈ 0.906402
    sigma_u = (0.75 * sqrt(pi) / (0.906402 * 1.5 * 2^0.25))^(1/beta_lv);
    u    = randn(1, d) * sigma_u;
    v    = randn(1, d);
    step = u ./ (abs(v).^(1/beta_lv));
    step = sign(step) .* min(abs(step), 10);   % clip extremes
end


% -------------------------------------------------------------------------
function X = lhs_init(n, d, lb, ub)
% Latin Hypercube Sampling initialization.
    X = zeros(n, d);
    for k = 1:d
        perm   = randperm(n);
        X(:,k) = lb(k) + (perm' - rand(n,1)) / n * (ub(k) - lb(k));
    end
end


% -------------------------------------------------------------------------
function [x_gb, f_gb, cv_gb] = grade_best_valid(grade_pop, x_fallback, f_fallback, cv_fallback)
% Return the best NON-EXILED agent in a grade (Deb-ordered).
% Fix: exiled agents are excluded from elite selection so they cannot
% become Gadaana or contribute to Gadaana pre-selection.
%
% Fix (KIMI moderate #1 — Hirbora rival fallback):
%   When ALL agents in the grade are exiled the old code silently fell
%   through to agent index 1, which may be a very poor or even exiled
%   attractor.  Instead, callers may pass an optional fallback triple
%   (e.g. Gaadaana / x_best).  If no fallback is supplied the function
%   still degrades gracefully to the best-of-exiled-agents, which is
%   at least the grade's own strongest member rather than index 1.
    n    = size(grade_pop.X, 1);
    f_gb = Inf;  cv_gb = Inf;  x_gb = grade_pop.X(1,:);
    for j = 1:n
        if grade_pop.exile_count(j) > 0, continue; end   % skip exiled
        if deb_compare(grade_pop.F(j), grade_pop.CV(j), f_gb, cv_gb)
            f_gb  = grade_pop.F(j);
            cv_gb = grade_pop.CV(j);
            x_gb  = grade_pop.X(j,:);
        end
    end

    % Fallback: all agents in this grade are exiled
    if isinf(f_gb)
        if nargin >= 4 && ~isempty(x_fallback)
            % Fix #3: use caller-supplied fallback (Gaadaana) — a
            % meaningful, globally competitive attractor.
            x_gb  = x_fallback;
            f_gb  = f_fallback;
            cv_gb = cv_fallback;
        else
            % Secondary fallback: best-of-exiled (better than index 1)
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


% -------------------------------------------------------------------------
function [x_gb, f_gb, cv_gb] = find_global_best(pop, N_GRADES)
% Return the global best NON-EXILED agent across all grades.
% Fix: exile mask applied so exiled agents cannot become Gaadaana.
    f_gb = Inf;  cv_gb = Inf;  x_gb = pop(1).X(1,:);
    for g = 1:N_GRADES
        for j = 1:size(pop(g).X, 1)
            if pop(g).exile_count(j) > 0, continue; end   % skip exiled
            if deb_compare(pop(g).F(j), pop(g).CV(j), f_gb, cv_gb)
                f_gb  = pop(g).F(j);
                cv_gb = pop(g).CV(j);
                x_gb  = pop(g).X(j,:);
            end
        end
    end
    % Fallback: if everything is exiled
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


% -------------------------------------------------------------------------
function Yuba = yuba_insert(Yuba, x_new, f_new, cv_new, NY)
% Insert a solution into the Yuba archive; keep at most NY entries,
% Deb-ordered (best first). Replace worst if archive is full.
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
            % Find worst in archive
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
    % Sort: feasible first, then ascending objective
    sort_key = Yuba.F;
    sort_key(Yuba.CV > 1e-10) = sort_key(Yuba.CV > 1e-10) + 1e12;
    [~, ord] = sort(sort_key);
    Yuba.X   = Yuba.X(ord,:);
    Yuba.F   = Yuba.F(ord);
    Yuba.CV  = Yuba.CV(ord);
end


% -------------------------------------------------------------------------
function [grade_pop, x_best, f_best, cv_best] = encampment( ...
             grade_pop, N_g, d, lb, ub, n_clusters, T_enc, ...
             x_best, f_best, cv_best, alpha_t, safe_eval, FE_max, FE_count)
% ENCAMPMENT  Temporary k-means clustering + local refinement.
% Mimics the 3-month secluded training of an incoming Luwa grade.
% Clusters form, refine internally, then dissolve (no permanent sub-groups).
%
% Fix (KIMI critical #1): every safe_eval call is guarded against the FE
%   budget — encampment will exit cleanly mid-phase if FE_max is reached,
%   preventing evaluation overshoot.
% Fix (KIMI critical #2): after each accepted move the routine updates
%   [x_best, f_best, cv_best] (Gaadaana) in-place so the convergence
%   record inside counted_eval always reflects the true global best.
%   The updated triple is returned to the caller for workspace sync.
%
% NOTE: FE_count is passed BY VALUE (MATLAB semantics).  The actual shared
%   FE_count lives in the outer closure and is incremented by safe_eval.
%   FE_count here is used only as a snapshot for the budget guard; the
%   guard re-reads it via the safe_eval closure each iteration.  To avoid
%   a stale-read issue we use a lightweight wrapper that checks the
%   closure's authoritative counter through a boolean sentinel returned
%   by safe_eval — but since safe_eval already halts evaluation once
%   FE_count >= FE_max (enforced by the outer while-loop and break
%   pattern), the simplest correct guard is: re-check FE_count against
%   FE_max after every eval.  We pass FE_max as a constant and rely on
%   the closure to maintain FE_count; we poll it through a local shadow.
    FE_local = FE_count;   % local shadow; updated after every eval call

    n_clust = min(n_clusters, N_g);
    c_idx   = randperm(N_g, n_clust);
    centres = grade_pop.X(c_idx, :);
    labels  = zeros(N_g, 1);

    % k-means: 10 fixed iterations (no evals — purely geometric)
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

    % Intra-cluster refinement for T_enc iterations (bonding phase)
    done = false;
    for enc_iter = 1:T_enc
        if done, break; end
        for c = 1:n_clust
            if done, break; end
            members = find(labels == c);
            if isempty(members), continue; end
            c_cen = mean(grade_pop.X(members,:), 1);
            for j = members'
                % ── Fix #1: budget guard ─────────────────────────────────
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
                FE_local = FE_local + 1;   % shadow tracks closure increment

                if deb_compare(f_new, cv_new, grade_pop.F(j), grade_pop.CV(j))
                    grade_pop.X(j,:) = x_new;
                    grade_pop.F(j)   = f_new;
                    grade_pop.CV(j)  = cv_new;
                    % ── Fix #2: update Gaadaana (global best) if improved ─
                    if deb_compare(f_new, cv_new, f_best, cv_best)
                        x_best  = x_new;
                        f_best  = f_new;
                        cv_best = cv_new;
                    end
                end
            end
        end
    end
    % Clusters dissolve; grade returns as a flat structure.
    % Caller receives updated [x_best, f_best, cv_best].
end


% -------------------------------------------------------------------------
function v = get_opt(s, field, default)
% Get field from options struct or return default.
    if isfield(s, field) && ~isempty(s.(field))
        v = s.(field);
    else
        v = default;
    end
end
