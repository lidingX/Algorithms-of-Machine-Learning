function [nz_n, on, y] = svmPred(oX, oY, X, alpha, b, kernel, kpar1, kpar2)
% Compute SVM model target  according to  y = -b + ((oT .* alpha) * K);
% INPUT ARGUMENTS:
% oX:           D x on Training points matrix for both class
% oY:           1 x on Class vector corresponding to training points
% X:            D x n Class values matrix corresponding to validation points
% alpha:        1 x on alpha
% b:
% kernel:       Type of Kernel mapping to be used
%               'knLin' : Linear (Default)
%               'knPoly' : Polynomial
%               'knGauss' : Gauss
%
% kpar1:        1st parameter for kernel function (optional, default=1)
% kpar2:        1st parameter for kernel function (optional, default=1)
% INPUT ARGUMENTS:
% y:            1 x n Class values Class vector corresponding to validation points

if(nargin < 8)
  kpar2 = 1;
end

if(nargin < 7)
  kpar1 = 1;
end

if(nargin < 6)
  kernel = 'knLin';
end

on = size(alpha,2);
for i = 1:on
    if oY(i) == 2
        oY(i) = 1;
    else
        oY(i) = -1;
    end
end
% sparsity: find nonzero alpha.
nz_ind = find(alpha ~= 0);
nz_alpha = alpha(nz_ind);
nz_oX = oX(:,nz_ind);
nz_oY = oY(nz_ind);
nz_n = size(nz_alpha,2);

K = Calkernel(nz_oX, X, kernel, kpar1, kpar2);
y = ((nz_oY .* nz_alpha) * K);
y = y - b;
[~, n] = size(y);
for i = 1:n
    if y(i) > 0
        y(i) = 2;
    else
        y(i) = 1;
    end
end
