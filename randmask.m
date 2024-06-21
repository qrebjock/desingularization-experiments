function [I, J, mask, kk] = randmask(m, n, k, fencemax)
%
% This code is used only to generate random problem instances, not to solve
% problems.
%
% Random mask: we randomly select k entries in an mxn matrix and return
% their indices (row indices in I, column indices in J) in uint32 class.
% The code is based on the assumption that k << mn, that an mn vector does
% not fit in memory and that it is complicated to uniformly generate a
% single integer in the range 1:mn. The code will usually select k entries,
% but it sometimes abandons with slightly less than k. The number of
% selected entries is returned in kk.
%
% fencemax is the maximum number of tries (a higher number increases the
% chance that we will get the correct number of samples in the mask, but
% increases computation time too).
%
% Nicolas Boumal, UCLouvain, Sept. 6, 2011.
% http://perso.uclouvain.be/nicolas.boumal/RTRMC/

    if nargin < 4 || isempty(fencemax)
        fencemax = 5;
    end

    % If m*n is unmanageable, do the random picking dance
    if m*n > 1e7 || ~exist('randsample') % randsample requires a toolbox

        pos = zeros(0, 2, 'uint32');
        kk = 0;
        fence = 1;
        while kk < k
            if fence > fencemax
                warning('randmask: Could not select exactly k = %d entries; selected %d instead.\n', k, kk);
                break;
            end
            rows = randi(m, k-kk, 1, 'uint32');
            cols = randi(n, k-kk, 1, 'uint32');
            pos((kk+1):k, 1:2) = [rows, cols];
            pos = unique(pos, 'rows');
            kk = size(pos, 1);
            fence = fence + 1;
        end

        I = pos(:, 1);
        J = pos(:, 2);

    % Otherwise, generate it explicitly
    else
        idx = uint32(randsample(m*n, k));
        [I, J] = ind2sub([m n], idx);
        kk = length(I);

    end

    % Order the indices I and J properly (which weridly requires switching
    % back to doubles ... Matlab, seriously, typing much?
    mask = sparse(double(I), double(J), ones(kk, 1), m, n, kk);
    [I, J] = find(mask);
    I = uint32(I);
    J = uint32(J);

end


