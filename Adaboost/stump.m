function model = stump(X,t,w,dim)
% a stump learner
% Input:
%   X: d x n data matrix
%   t: 1 x n label (-1/1)
%   w: 1 x n training weights
%   dim: training dimension of features
% Output:
%   model: trained model structure
X = X(dim,:);
[x,I] = sort(X);
w = w(I);
t = t(I);
n = length(x);
ws = sum(w(t==1));
thre = x(1) - 1;
[wmax, ind] = max([ws, 1-ws]);
if(ind == 2)
  ind = -1;
end
change = 0;
for i=1:n
   if(t(i) == 1)
     change = change - w(i);
   else
     change = change + w(i);
   end
   if(i<n && x(i) == x(i+1))
       continue;
   end
   tw = ws + change;
   [twmax, tind]    =   max([tw, 1-tw]);
   if(twmax > wmax)
    wmax = twmax;
    if(tind == 2)
      tind = -1;
    end
    ind = tind;
    if(i == n)
     thre = x(i)+1;
    else
     thre = (x(i)+x(i+1))/2;
    end
   end
end
% display(x);
% display(t);
% display(w);
% display(wmax);
% display(ind);
% display(thre);
model.thre = thre;
model.dim = dim;
model.rever = ind;
