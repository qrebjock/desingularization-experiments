function problem = completiondesingularization(r, alpha, data)
% Low-rank completion:
%
%     f(X) = 1/2 \|mask.*(X - A)\|_F^2
%
% Create the Manopt problem with the desingularization geometry.
%
% alpha is the desingularization metric parameter.
%

    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 0.5;
    end

    m = data.m;
    n = data.n;
    A = data.A;
    I = data.I;
    J = data.J;
    mask = data.mask;

    M = desingularizationfactory(m, n, r, alpha);
    problem.M = M;

    problem.name = sprintf('Completion desingularization m=%d n=%d r=%d a=%d', m, n, r, alpha);

    function store = prepareinvSf(X, store)
        if ~isfield(store, 'invSf')
            store.invSf = M.sfactorinv(X);
        end
    end

    function store = prepareUS(X, store)
        if ~isfield(store, 'US')
            store.US = X.U * X.S;
        end
    end

    function store = prepareegrad(X, store)
        if ~isfield(store, 'egrad')
            store = prepareUS(X, store);
            store.egrad = spmaskmult(store.US, X.V', I, J) - A;
        end
    end

    function store = prepareUSinvSf(X, store)
        if ~isfield(store, 'USinvSf')
            store = prepareUS(X, store);
            store = prepareinvSf(X, store);

            store.USinvSf = store.US * store.invSf;
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = prepareegrad(X, store);

        f = .5*norm(store.egrad, 'fro')^2;

        store = incrementcounter(store, 'costcalls');
    end

    problem.grad = @grad;
    function [G, store] = grad(X, store)
        store = prepareegrad(X, store);
        store = prepareinvSf(X, store);
        store = prepareUS(X, store);
        store = prepareUSinvSf(X, store);

        ZUSinvSf = multfullsparse(store.USinvSf', store.egrad, mask)';

        G.K = multsparsefull(store.egrad, X.V, mask);
        G.Vp = ZUSinvSf - X.V * (X.V' * ZUSinvSf);

        store = incrementcounter(store, 'gradcalls');
    end

    problem.hess = @hess;
    function [H, store] = hess(X, Xd, store)
        store = prepareegrad(X, store);
        store = prepareinvSf(X, store);
        store = prepareUS(X, store);
        store = prepareUSinvSf(X, store);

        spXd = spmaskmult(Xd.K, X.V', I, J) ...
            + spmaskmult(store.US, Xd.Vp', I, J);

        XdV = multsparsefull(spXd, X.V, mask);
        ZVp = multsparsefull(store.egrad, Xd.Vp, mask);

        H.K = XdV + ZVp - store.USinvSf * (store.US' * ZVp);

        MK = Xd.K - store.USinvSf * (store.US' * Xd.K);
        ZtMK = multfullsparse(MK', store.egrad, mask)';
        XdtUS = multfullsparse(store.US', spXd, mask)';
        W = (XdtUS + ZtMK) * store.invSf;

        H.Vp = W - X.V * (X.V' * W);

        store = incrementcounter(store, 'hesscalls');
    end

end
