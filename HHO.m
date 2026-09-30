function [x_best, f_best, convergence] = HHO(obj_func, lb, ub, N, T_max, options)
% HHO  Harris Hawks Optimization (Heidari et al., 2019).
%
%  Implements the four phases of Harris hawks hunting:
%    1. Exploration (perch based on random tall tree / random location)
%    2. Transition from exploration to exploitation (energy drop)
%    3. Exploitation: soft besiege / hard besiege / rapid dives
%    4. Levy flight-based rapid dives
%
%  Reference: Heidari AA, Mirjalili S, Faris H, Aljarah I, Mafarja M,
%  Chen H. Harris hawks optimization: Algorithm and applications.
%  Future Generation Computer Systems, 2019, 97:849-872.
%
%  Inputs:
%    obj_func  - function handle: f = obj_func(x), x is [1 x d]
%    lb, ub    - bounds [1 x d]
%    N         - population size
%    T_max     - maximum iterations
%
%  Outputs:
%    x_best      - best position found [1 x d]
%    f_best      - best objective value
%    convergence - convergence curve [T_max x 1]

    d = numel(lb);
    if nargin < 6 || ~isstruct(options), options = struct(); end

    % Initialization
    if isfield(options,'X0') && ~isempty(options.X0)
        X = options.X0;
    else
        X = lb + rand(N, d) .* (ub - lb);
    end
    F = zeros(N, 1);
    for i = 1:N
        F(i) = obj_func(X(i,:));
    end

    [f_best, idx] = min(F);
    x_best = X(idx,:);   % rabbit (prey) position

    convergence = zeros(T_max, 1);

    for t = 1:T_max
        % Escaping energy of rabbit
        E0 = 2 * rand - 1;                    % initial energy in [-1, 1]
        E  = 2 * E0 * (1 - t / T_max);        % linearly decreasing energy

        for i = 1:N
            q = rand;
            r = rand;

            if abs(E) >= 1
                %% Exploration phase
                if q >= 0.5
                    % Perch near a random hawk
                    rand_idx = randi(N);
                    X_rand = X(rand_idx,:);
                    X(i,:) = X_rand - rand * abs(X_rand - 2 * rand * X(i,:));
                else
                    % Perch near rabbit with random component
                    X(i,:) = (x_best - mean(X, 1)) - rand * ...
                             (lb + rand(1,d) .* (ub - lb));
                end
            else
                %% Exploitation phase
                J = 2 * (1 - rand);    % random jump strength

                if r >= 0.5 && abs(E) >= 0.5
                    % Soft besiege
                    X(i,:) = x_best - E * abs(J * x_best - X(i,:));

                elseif r >= 0.5 && abs(E) < 0.5
                    % Hard besiege
                    X(i,:) = x_best - E * abs(x_best - X(i,:));

                elseif r < 0.5 && abs(E) >= 0.5
                    % Soft besiege with progressive rapid dives
                    Y = x_best - E * abs(J * x_best - X(i,:));
                    Y = max(lb, min(ub, Y));
                    f_Y = obj_func(Y);

                    S = Y + rand(1,d) .* levy_flight(d);
                    S = max(lb, min(ub, S));
                    f_S = obj_func(S);

                    if f_Y < F(i)
                        X(i,:) = Y;
                        F(i) = f_Y;
                    end
                    if f_S < F(i)
                        X(i,:) = S;
                        F(i) = f_S;
                    end

                    % Update best
                    if F(i) < f_best
                        f_best = F(i);
                        x_best = X(i,:);
                    end
                    continue;   % skip normal evaluation below

                else
                    % Hard besiege with progressive rapid dives (r<0.5, |E|<0.5)
                    Y = x_best - E * abs(J * x_best - mean(X, 1));
                    Y = max(lb, min(ub, Y));
                    f_Y = obj_func(Y);

                    S = Y + rand(1,d) .* levy_flight(d);
                    S = max(lb, min(ub, S));
                    f_S = obj_func(S);

                    if f_Y < F(i)
                        X(i,:) = Y;
                        F(i) = f_Y;
                    end
                    if f_S < F(i)
                        X(i,:) = S;
                        F(i) = f_S;
                    end

                    if F(i) < f_best
                        f_best = F(i);
                        x_best = X(i,:);
                    end
                    continue;
                end
            end

            % Boundary handling
            X(i,:) = max(lb, min(ub, X(i,:)));

            % Evaluate
            F(i) = obj_func(X(i,:));

            % Update rabbit (best)
            if F(i) < f_best
                f_best = F(i);
                x_best = X(i,:);
            end
        end

        convergence(t) = f_best;
    end
end


function L = levy_flight(d)
% Levy flight step using Mantegna's algorithm (beta = 1.5)
    beta = 1.5;
    sigma_u = (gamma(1+beta)*sin(pi*beta/2) / ...
               (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1, d) * sigma_u;
    v = randn(1, d);
    L = u ./ (abs(v).^(1/beta));
end
