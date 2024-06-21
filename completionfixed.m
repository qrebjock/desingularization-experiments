function problem = completionfixed(r, data)
% Low-rank completion:
%
%     f(X) = 1/2 \|mask.*(X - A)\|_F^2
%
% Create the Manopt problem with the fixed-rank geometry.
%

    m = data.m;
    n = data.n;
    A = data.A;
    I = data.I;
    J = data.J;
    mask = data.mask;

    problem.M = fixedrankembeddedfactory(m, n, r);

    problem.name = sprintf('Completion fixed m=%d n=%d r=%d', m, n, r);

    function store = prepareegrad(X, store)
        if ~isfield(store, 'egrad')
            spX = spmaskmult(X.U * X.S, X.V', I, J);
            store.egrad = spX - A;
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = prepareegrad(X, store);

        f = .5 * norm(store.egrad, 'fro')^2;

        store = incrementcounter(store, 'costcalls');
    end

    problem.grad = @grad;
    function [G, store] = grad(X, store)
        store = prepareegrad(X, store);

        ZV = multsparsefull(store.egrad, X.V, mask);
        ZtU = multfullsparse(X.U', store.egrad, mask)';

        G.M = X.U' * ZV;
        G.Up = ZV - X.U * G.M;
        G.Vp = ZtU - X.V * G.M';

        store = incrementcounter(store, 'gradcalls');
    end

    problem.hess = @hess;
    function [H, store] = hess(X, Xd, store)
        store = prepareegrad(X, store);

        ZVpSinv = multsparsefull(store.egrad, Xd.Vp, mask) / X.S;
        ZtUpSinv = multfullsparse(Xd.Up', store.egrad, mask)' / X.S;

        spXd = spmaskmult(X.U * Xd.M, X.V', I, J) ...
            + spmaskmult(Xd.Up, X.V', I, J) ...
            + spmaskmult(X.U, Xd.Vp', I, J);
        XdV = multsparsefull(spXd, X.V, mask);
        XdtU = multfullsparse(X.U', spXd, mask)';

        toprojectU = XdV + ZVpSinv;
        toprojectV = XdtU + ZtUpSinv;

        H.M = X.U' * XdV;
        H.Up = toprojectU - X.U * (X.U' * toprojectU);
        H.Vp = toprojectV - X.V * (X.V' * toprojectV);

        store = incrementcounter(store, 'hesscalls');
    end

end
