function model = kfisher(X, t, lamda, kernel,kpar1, kpar2)
% Output:
% X:        d x n data
% t:        1 x n target(1/2)
% lamda:    regularization coefficient
% kernel:   Type of Kernel mapping to be used
%           'knLin' : Linear (Default)
%           'knPoly' : Polynomial
%           'knGauss' : Gauss
% kpar1:        1st parameter for kernel function (optional, default=1)
% kpar2:        2nd parameter for kernel function (optional, default=2)
%
% OUTPUT ARGUMENTS:
% alpha?           1 x n
if(nargin < 6)
  kpar2 = 1;
end

if(nargin < 5)
  kpar1 = 1;
end

K = Calkernel(X,X,kernel,kpar1,kpar2);
index1 = find(t==1);
n1 = size(index1,2);
index2 = find(t==2);
n2 = size(index2,2);

K1 = K(:,index1);
sum1 = sum(K1,2);
K2 = K(:,index2);
sum2 = sum(K2,2);


N1 = K1*K1' - sum1*sum1'/n1;
N2 = K2*K2' - sum2*sum2'/n2;
Gamma = sum1/n1 - sum2/n2;
A = N1 + N2 + lamda*K;
A = A/norm(A);
Gamma = Gamma/norm(A);
alpha = A\Gamma;
model.alpha = alpha';
model.b = -sum(alpha'*K,2)/(n1+n2);
model.kernel = kernel;
model.kpar1 = kpar1;
model.kpar2 = kpar2;
model.oX = X;
