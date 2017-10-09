function t = adaboostBinPred(model, X)
% Prediction of binary Adaboost
% input:
%   model: trained model structure
%   X: d x n data matrix
% output:
%   t: 1 x n prediction
% Written by Mo Chen (sth4nth@gmail.com).
Alpha = model.alpha;
M = size(Alpha,2);
t = zeros(1,size(X,2));
for m = 1:M
    y = stumpPred(model.weakmodels(m), X);
    t = t+Alpha(m)*y;
end
t = sign(t);
t(t== 1) = 2;
t(t== -1) = 1;
