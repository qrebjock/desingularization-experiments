function problem = completionlr(r, data, precondition)
% Low-rank completion:
%
%     f(X) = 1/2 \|mask.*(X - A)\|_F^2
%
% Create the Manopt problem with the LR factorization geometry.
%

    m = data.m;
    n = data.n;
    A = data.A;
    I = data.I;
    J = data.J;
    mask = data.mask;

    if ~exist('precondition', 'var') || isempty(precondition)
        precondition = false;
    end
    assert(islogical(precondition), 'precondition must be true or false.');

    problem.M = productmanifold(struct( ...
        'L', euclideanfactory(m, r), ...
        'R', euclideanfactory(n, r) ...
        ));

    if precondition
        problem.name = sprintf('Completion LR m=%d n=%d r=%d (preconditioned)', m, n, r);
    else
        problem.name = sprintf('Completion LR m=%d n=%d r=%d', m, n, r);
    end

    function store = preparespLRmA(X, store)
        if ~isfield(store, 'spLRmA')
            store.spLRmA = spmaskmult(X.L, X.R', I, J) - A;
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = preparespLRmA(X, store);

        f = .5 * norm(store.spLRmA, 'fro')^2;

        store = incrementcounter(store, 'costcalls');
    end

    problem.grad = @grad;
    function [G, store] = grad(X, store)
        store = preparespLRmA(X, store);

        G.L = multsparsefull(store.spLRmA, X.R, mask);
        G.R = multfullsparse(X.L', store.spLRmA, mask)';

        store = incrementcounter(store, 'gradcalls');
    end

    problem.hess = @hess;
    function [H, store] = hess(X, Xd, store)
        store = preparespLRmA(X, store);

        spLdRt = spmaskmult(Xd.L, X.R', I, J);
        spLRdt = spmaskmult(X.L, Xd.R', I, J);

        H.L = multsparsefull(spLdRt, X.R, mask) ...
            + multsparsefull(spLRdt, X.R, mask) ...
            + multsparsefull(store.spLRmA, Xd.R, mask);

        H.R = multfullsparse(X.L', spLdRt, mask)' ...
            + multfullsparse(X.L', spLRdt, mask)' ...
            + multfullsparse(Xd.L', store.spLRmA, mask)';

        store = incrementcounter(store, 'hesscalls');
    end

    if precondition
        problem.precon = @precon;
    end
    function [PXd, store] = precon(X, Xd, store)
        % See for example the paper about scaled GD for matrix completion:
        % https://jmlr.org/papers/volume22/20-1067/20-1067.pdf
        PXd.L = Xd.L / (X.R'*X.R);
        PXd.R = Xd.R / (X.L'*X.L);
    end

end
