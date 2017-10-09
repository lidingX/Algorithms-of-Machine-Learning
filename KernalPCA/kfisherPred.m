function y = kfisherPred(model,X)
% INPUT ARGUMENTS:
% oX:           D x on Training points matrix for both class
% X:            D x n Class values matrix corresponding to validation points
% alpha:        1 x on alpha
% kernel:       Type of Kernel mapping to be used
%               'knLin' : Linear (Default)
%               'knPoly' : Polynomial
%               'knGauss' : Gauss
%
% kpar1:        1st parameter for kernel function (optional, default=1)
% kpar2:        2nd parameter for kernel function (optional, default=1)
% OutPUT ARGUMENTS:
% t:            1 x n Class values Class vector corresponding to validation points
alpha = model.alpha;
b = model.b;
kernel = model.kernel;
kpar1 = model.kpar1;
kpar2 = model.kpar2 ;
oX = model.oX;

K = Calkernel(oX, X, kernel, kpar1, kpar2);
y = alpha * K + b;
index1 = find(y >= 0);
index2 = find(y < 0);
y(index1) = 1;
y(index2) = 2;
