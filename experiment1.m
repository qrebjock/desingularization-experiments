clear all; close all; clc;

options.verbosity = 2;
options.maxiter = 500;
options.maxtime = 10 * 60;
options.tolcost = 1e-12;
options.tolgradnorm = 0;
options.theta = sqrt(2) - 1; % theta parameter for tCG

m = 5000;
n = 5000;
r = 20;
rstar = 10;
rho = 5 * r * (m + n - r);

data.m = m;
data.n = n;
[data.I, data.J, data.mask] = randmask(m, n, rho);
AU = stiefelfactory(m, rstar).rand();
AS = diag(.5 + .5*rand(rstar, 1));
AV = stiefelfactory(n, rstar).rand();
data.A = spmaskmult(AU * AS, AV', data.I, data.J);

problems = {
    completionlr(r, data, false), ...
    completionfixed(r, data), ...
    completiondesingularization(r, 1/20, data), ...
    completiondesingularization(r, 1/2, data), ...
    completiondesingularization(r, 5, data)
    };

X0 = fixedrankembeddedfactory(m, n, r).rand();
X0.S = X0.S/1000;
LR0.L = X0.U * sqrtm(X0.S);
LR0.R = X0.V * sqrtm(X0.S);
X0s = {LR0, X0, X0, X0, X0};

num_problems = size(problems, 2);
infos = cell(1, num_problems);
for p = 1:num_problems
    problem = problems{p};
    fprintf('Solving %s\n', problem.name);
    [~, ~, info, ~] = trustregions(problem, X0s{p}, options);
    fprintf('\n');
    infos{p} = info;
end

plotinfo(infos, 'cost')
