function Kc = Calkernel(X1, X2, kernel, kpar1, kpar2)
% Compute kernel matrix
% INPUT ARGUMENTS:
% X1:            D x n1 points matrix
% X2:            D x n2 points matrix
% kernel:       Type of Kernel mapping to be used
%               'knLin' : Linear (Default)
%               'knPoly' : Polynomial
%               'knGauss' : Gauss
%
% kpar1:        1st parameter for kernel function (optional, default=0)
% kpar2:        1st parameter for kernel function (optional, default=0)
%
% OUTPUT ARGUMENTS:
% Kc：           n x n kernel matrix

if(nargin < 5)
  kpar2 = 1;
end

if(nargin < 4)
  kpar1 = 1;
end

switch kernel
        case 'knPoly'
            K = knPoly(X1,X1,kpar1,kpar2);
            mK = mean(K);
            mmK = mean(mK);
            Kc = knPoly(X1,X2,kpar1,kpar2)+mmK-bsxfun(@plus,mean(knPoly(X1,X1,kpar1,kpar2))',mean(knPoly(X1,X2,kpar1,kpar2)));
        case 'knGauss'
            K = knGauss(X1,X1,kpar1);
            mK = mean(K);
            mmK = mean(mK);
            Kc = knGauss(X1,X2,kpar1)+mmK-bsxfun(@plus,mean(knGauss(X1,X1,kpar1))',mean(knGauss(X1,X2,kpar1)));
        otherwise
              K = knLin(X1, X2);
end
