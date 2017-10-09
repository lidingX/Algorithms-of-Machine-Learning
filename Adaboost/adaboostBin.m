function model = adaboostBin(X, t)
% Adaboost for binary classification (weak learner: kmeans)
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1/2)
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
w = ones(1,n)/n;
M = d;
Alpha = zeros(1,M);
t(t==1) = -1;
t(t==2) = 1;
for m = 1:M
    % weak learner
    dim = mod(m,d) + 1;
    %dim = ceil(rand()*d);
    weakmodel = stump(X,t,w,dim);
    y = stumpPred(weakmodel, X);
    model.weakmodels(m) = weakmodel;
    % adaboost
    I = y~=t;
    e = dot(w,I);
    if(e < 1e-4)
      break;
    end
    alpha = 0.5*log((1-e)/e);
    w = w.*exp(-alpha*(y.*t));
    w = w/sum(w);
    Alpha(m) = alpha;
end
model.alpha = Alpha;
