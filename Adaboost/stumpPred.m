function y = stumpPred(model, X)
% a stump classifier prediction
% Input:
%   model: trained model structure
%   X: d x n data matrix
% Output:
%   y: 1 x n prediction (-1/1)
 y = (2*(X(model.dim,:)> model.thre) - 1) * model.rever;
