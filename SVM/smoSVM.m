function [alpha, b, w, evals, stp, flag] = smoSVM(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps, method)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [alpha, b, w, evals, stp, glob] = SMO2(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps, method)
% SMO2 algorithm of Platt with Keerthi modifications:
% ALL KERNEL EVALUATIONS ARE DONE IN THE BEGINING AND KEPT IN MEMORY
% Implements:
% 1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
%    John C. Platt
% 2. Improvements to Platt's SMO Algorithm for SVM Classifier Design
%    S.S. Keerthi, S.K. Shevade, C. Bhattacharyya and K.R.K. Murthy
%    Technical Report CD-99-14
% The classifier that this second algorithm outputs is f(x)=wx-b
% Furthermore, note that in SMO Platt, due to a randomisation,
% the results might not be identical even for the same data
%
% INPUT ARGUMENTS:
% X:            D x n Training points matrix for both class
% Y:            1 x n Class values vector corresponding to training points
% kernel:       Type of Kernel mapping to be used
%               'knLin' : Linear (Default)
%               'knPoly' : Polynomial
%               'knGauss' : Gauss
%
% kpar1:        1st parameter for kernel function (optional, default=1)
% kpar2:        1st parameter for kernel function (optional, default=1)
% C:            parameter which trades off wide margin with a small number of margin failures
% tol:          tollerance (Keerthi used 0.001
% steps:        maximum allowed steps to be taken (1st stopping condition)
% eps:          accuracy
% method:       1->Keerthi modification 1, 2->Keerthi modification 2
%
% OUTPUT ARGUMENTS:
% alpha:        the column-vector of m+n Lagrange multipliers for each point. The
%               first m are for the m points of A set, and the next n for the n points of
%               B set.
% b:            the threshold value
% w:            the normal to the optimal separating hyperplane (meaningfull ONLY for
%               Linear kernel. (For test purposes only.)
% evals:        num of norm evaluations
% stp:          steps taken till the end
% flag:         a logical value which indicates whether or not the selected SVM
%               training algorihm has been terminated abnormally (1) or normally
%               (0). Abnormal termination means that no solution exists with the
%               specific kernel function choice and another kernel function should
%               be selected. The results obtained in this case are unreliable.
%
% Functions in this file have been provided by Michael Mavroforakis (c) 2003.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dbstop if error
dbg = 2;

if (nargin < 10)
    method = 1;
end

if (nargin < 9)
    eps = 0.0001;
end

if (nargin < 8)
    steps = 10000;
end

if (nargin < 7)
    tol = 0.001;
end

if (nargin < 6)
    C = inf;
end

if (nargin < 5)
    kpar2 = 1;
end

if (nargin < 4)
    kpar1 = 1;
end

if (nargin < 3) %!!!!!!!!!!!!!!!!!!!!!!!!!!!!kernel ones matrix
    kernel = 'knLin';
end

if (nargin < 2)
    error('Error: At least two arguments (training points and class values) must be supplied');
else
    [D, n] = size(X);
    [D1, n1] = size(Y);
end

if (1 ~= D1)
    error('Error: Class values cannot be vectors but real numbers');
end

if (n ~= n1)
    error('Error: Number of rows of X and Y must be the same (one class value for each sample)');
end


%here we should check if representatives of both classes are presented as training points


for iter = 1:n1
    if Y(iter) == 2
        Y(iter) = 1;
    else
        Y(iter) = -1;
    end
end
if (method == 1)
    [alpha, b, w, evals, stp, glob] = SMO_Keerthi_modif1(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps);
else
    [alpha, b, w, evals, stp, glob] = SMO_Keerthi_modif2(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps);
end
flag=(glob.b_up<glob.b_low- 2*tol)|(stp>=steps);


% if(flag==1)
%     fprintf('The algorithm has not converged. This may be due to:\n (a) the maximum number of iterations has been reached and convergence has not, yet, been achieved or \n (b) the chosen values for the hyperparameters (C as well as the parameters that define the kernel function) \n can not lead to a solution. \n')
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [alpha, b, w, evals, stp, glob] = SMO_Keerthi_modif1(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps)
        [D, n] = size(X);
        %initialize alpha array to all zero
        %--------------------------------------------------------------------------
        alpha = zeros(1,n);
        w = zeros(D,1);
        b = 0;
        evals = 0;
        %--------------------------------------------------------------------------

        %!!!!!!!!!!!!!!!!!!!!!!!!!!calkernel
        K=  Calkernel(X, X, kernel, kpar1, kpar2);


        %initialize struct for temporary variables that must be global
        %--------------------------------------------------------------------------

        glob = struct('fcache',[],'b_up',0,'b_low',0,'i_up',0,'i_low',0,'v_1',[],'v_2',[],...
            'I_0',[],'I_1',[],'I_2',[],'I_3',[],'I_4',[]);
        %%initialize fcache array to all zero and its size to n
        glob.fcache = zeros(1,n);
        %initialize b_up = -1, i_up to any one index of class 1
        glob.b_up = -1;
        glob.v_1 = find( Y==1 );
        glob.i_up = glob.v_1(1);
        %initialize b_low = 1, i_low to any one index of class 2
        glob.b_low = 1;
        glob.v_2 = find( Y==-1 );
        glob.i_low = glob.v_2(1);
        %set fcache[i_low] = 1 and fcache[i_up] = -1
        glob.fcache(glob.i_low) = 1;
        glob.fcache(glob.i_up) = -1;

        %Initialize the I_* sets
        for i=1:size(alpha,2)
            if(alpha(i) < eps)
                alpha(i) = 0;
            elseif(alpha(i) > C - eps)
                alpha(i) = C;
            end
        end
        glob.I_0 = find(alpha > 0 & alpha < C);
        glob.I_1 = find(alpha(glob.v_1) == 0 );
        glob.I_1 = glob.v_1(glob.I_1);
        glob.I_2 = find(alpha(glob.v_2) == C);
        glob.I_2 = glob.v_2(glob.I_2);
        glob.I_3 = find(alpha(glob.v_1) == C);
        glob.I_3 = glob.v_1(glob.I_3);
        glob.I_4 = find(alpha(glob.v_2) == 0);
        glob.I_4 = glob.v_2(glob.I_4);
        %--------------------------------------------------------------------------


        stp = 0;
        numChanged = 0;
        examineAll = 1;
        while ( ((numChanged > 0) || (examineAll==1))&&(stp <= steps))
            numChanged = 0;
            if (examineAll==1)
                for i = 1 : n
                    stp = stp + 1;
                    if (stp>steps)  break;      end
                    [retval, alpha, w, b, stp, evals,glob] =...
                        examineExampleK(i, glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, steps, stp, evals, eps,K);
                    numChanged = numChanged + retval;
                end
            else
                k = length(glob.I_0);
                for i = 1 : k
                    stp = stp + 1;
                    if (stp > steps) break; end
                    if (i > length(glob.I_0)) break; end %glob.I_0 changes inside loop (in examineExampleK)
                    [retval, alpha, w, b, stp, evals, glob] =...
                        examineExampleK(glob.I_0(i), glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, steps, stp, evals, eps,K);
                    numChanged = numChanged + retval;
                    %it is easy to check if optimality on I_0 is attained...
                    if ( (glob.b_up) > ( glob.b_low - (2*tol) ) )
                        %exit the loop after setting numChanged = 0
                        numChanged = 0;
                        break;      %EDW TA PRAGMATA ALLAZOUN AN XRHSIMOPOIHSOUME THN RETURN. SYSKEKRIMENA MERIKES FORES ENW EXEI SYMBEI
                        %TO b_up>b_low POY EINAI TO ZHTOUMENO, ME THN BREAK BGAINOUME MONO APO TO ESWTERIKO FOR (TO PSAKSIMO STO I_0)
                        %ENW ISWS (DEN EIMAI SIGOUROS) THA EPREPE NA TERMATIZEI O ALGORITHMOS. ME THN BREAK TO PROGRAMMA SYNEXIZEI
                        %KAI ALLAZOUN TA b_up KAI b_low KAI SXHMATIKA FAINETAI KALYTEROS O TAKSIMOMHTHS.
                        %ME THN RETURN OMWS EINAI PIO SYNTOMOS KAI PALI FAINETAI SWSTOS. NA TO PSAKSOYME.
                    end
                end
            end
            if (examineAll == 1)
                examineAll = 0;
            elseif (numChanged == 0)
                examineAll = 1;
            end
            b=(glob.b_up+glob.b_low)/2;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [alpha, b, w, evals, stp, glob] = SMO_Keerthi_modif2(X, Y, kernel, kpar1, kpar2, C, tol, steps, eps)
        [D, n] = size(X);
        %initialize alpha array to all zero
        %--------------------------------------------------------------------------
        alpha = zeros(1,n);
        w = zeros(D,1);
        b = 0;
        evals = 0;
        %--------------------------------------------------------------------------

        %!!!!!!!!!!!!!!!!!!!!!!!!!!calkernel
        K=  Calkernel(X, X, kernel, kpar1, kpar2);


        %initialize struct for temporary variables that must be global
        %--------------------------------------------------------------------------

        glob = struct('fcache',[],'b_up',0,'b_low',0,'i_up',0,'i_low',0,'v_1',[],'v_2',[],...
            'I_0',[],'I_1',[],'I_2',[],'I_3',[],'I_4',[]);
        %%initialize fcache array to all zero and its size to n
        glob.fcache = zeros(1,n);
        %initialize b_up = -1, i_up to any one index of class 1
        glob.b_up = -1;
        glob.v_1 = find( Y==1 );
        glob.i_up = glob.v_1(1);
        %initialize b_low = 1, i_low to any one index of class 2
        glob.b_low = 1;
        glob.v_2 = find( Y==-1 );
        glob.i_low = glob.v_2(1);
        %set fcache[i_low] = 1 and fcache[i_up] = -1
        glob.fcache(glob.i_low) = 1;
        glob.fcache(glob.i_up) = -1;

        %Initialize the I_* sets
        for i=1:size(alpha,2)
            if(alpha(i) < eps)
                alpha(i) = 0;
            elseif(alpha(i) > C - eps)
                alpha(i) = C;
            end
        end
        glob.I_0 = find(alpha > 0 & alpha < C);
        glob.I_1 = find(alpha(glob.v_1) == 0 );
        glob.I_1 = glob.v_1(glob.I_1);
        glob.I_2 = find(alpha(glob.v_2) == C);
        glob.I_2 = glob.v_2(glob.I_2);
        glob.I_3 = find(alpha(glob.v_1) == C);
        glob.I_3 = glob.v_1(glob.I_3);
        glob.I_4 = find(alpha(glob.v_2) == 0);
        glob.I_4 = glob.v_2(glob.I_4);
        %--------------------------------------------------------------------------


        stp = 0;
        numChanged = 0;
        examineAll = 1;
        while ( ((numChanged > 0) || (examineAll==1))&&(stp <= steps))
            numChanged = 0;
            found = 0;
            if (examineAll==1)
                %fprintf('examineall:\n');
                for i = 1 : n
                    stp = stp + 1;
                    if (stp>steps)
                        break;
                    end
                    [cul,retval, alpha, w, b, stp, evals,glob] =...
                        examineExampleK(i, glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, steps, stp, evals, eps,K);
                    numChanged = numChanged + retval;
                    found = found + cul;
                end
            else
                k = length(glob.I_0);
                for i = 1 : k
                    stp = stp + 1;
                    if (stp > steps)
                        break;
                    end
                    if (i > length(glob.I_0))
                        break;
                    end %glob.I_0 changes inside loop (in examineExampleK)
                    [cul,retval, alpha, w, b, stp, evals, glob] =...
                        examineExampleK(glob.I_0(i), glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, steps, stp, evals, eps,K);
                    numChanged = numChanged + retval;
                    %it is easy to check if optimality on I_0 is attained...
                    if ( (glob.b_up) > ( glob.b_low - (2*tol) ) )
                        %exit the loop after setting numChanged = 0
                        numChanged = 0;
                        break;      %EDW TA PRAGMATA ALLAZOUN AN XRHSIMOPOIHSOUME THN RETURN. SYSKEKRIMENA MERIKES FORES ENW EXEI SYMBEI
                        %TO b_up>b_low POY EINAI TO ZHTOUMENO, ME THN BREAK BGAINOUME MONO APO TO ESWTERIKO FOR (TO PSAKSIMO STO I_0)
                        %ENW ISWS (DEN EIMAI SIGOUROS) THA EPREPE NA TERMATIZEI O ALGORITHMOS. ME THN BREAK TO PROGRAMMA SYNEXIZEI
                        %KAI ALLAZOUN TA b_up KAI b_low KAI SXHMATIKA FAINETAI KALYTEROS O TAKSIMOMHTHS.
                        %ME THN RETURN OMWS EINAI PIO SYNTOMOS KAI PALI FAINETAI SWSTOS. NA TO PSAKSOYME.
                    end
                end
            end
            if (examineAll == 1)
                %fprintf('numChanged:%d found:%d\n',numChanged,found);
                examineAll = 0;
            elseif (numChanged == 0)
                examineAll = 1;
            end
            b=(glob.b_up+glob.b_low)/2;
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [retval, alpha, w, b, evals, glob] =...
    takeStepK(i1, i2, glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, evals, eps,K)
%Much of this procedure is same as that in Platt's SMO pseudo-code.

%drawnow; %%used to give Matlab the opportunity to examine any pending ctrl+C (while in deep loops)
if (get(0,'PointerLocation')==[1 1])
    %disp('Press <ctrl+c> to stop or any key to interrupt execution temporarily ...');
    %pause;
    %disp('Type "return" to carry on');
    keyboard;
end
if (i1 == i2)
    %fprintf('i1==i2\n');
    retval = 0;
    return;
end
alph1 = alpha(i1);
y1 = Y(i1);
F1 = glob.fcache(i1);
alph2 = alpha(i2);
y2 = Y(i2);
F2 = glob.fcache(i2);
s = y1 * y2;


%Compute L, H - If L=H return 0
%--------------------------------------------------------------------------
if (y1 ~= y2)
    L = max([0, alph2 - alph1]);
    H = min([C, C + alph2 - alph1]);
else % y1 == y2
    L = max([0, alph1 + alph2 - C]);
    H = min([C, alph1 + alph2]);
end
if (L == H) %%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    %fprintf('L == H\n');
    retval = 0;
    return;
end
%--------------------------------------------------------------------------


%%computation of the derivative eta
%--------------------------------------------------------------------------
k11 = K(i1,i1);
k12 = K(i1,i2);
k22 = K(i2,i2);
evals = evals + 3;
eta = (2*k12) - k11 - k22;
%%computation of new alpha(i2)
if (eta<0)
    a2 =alph2-(y2*(F1-F2)/eta); %%HERE is different from Platt
    %fprintf('eta<0: i1:%d i2:%d dalph2:%f unclutter a2:%f L:%f H:%f\n',i1,i2,alph2,a2,L,H);
    if (a2 < L)
        a2 = L;
    elseif (a2 > H)
        a2 = H;
    end
else %%the derivative is 0 => we have to make optimization by other means
    %%Lobj = objective function at a2=L (according to Platt)
    %%Hobj = objective function at a2=H (according to Platt)
    L1 = alph1 + (s*(alph2-L));
    H1 = alph1 + (s*(alph2-H));
    f1 = (-y1 * F1) + (alph1 * k11) + (s * alph2 * k12);
    f2 = (-y2 * F2) + (alph2 * k22) + (s * alph1 * k12);
    Lobj = (L1 * f1) + (L * f2) - (0.5 * k11 * L1^2) - (0.5 * k22 * L^2) - (s * k12 * L * L1);
    Hobj = (H1 * f1) + (H * f2) - (0.5 * k11 * H1^2) - (0.5 * k22 * H^2) - (s * k12 * H * H1);
    %fprintf('eta>=0 i1:%d i2:%d dalph2:%f unclutter L:%f H:%f\n',i1,i2,alph2,L,H);
    if (Lobj > Hobj)
        a2 = L;
    elseif (Lobj < Hobj)
        a2 = H;
    else
        a2 = alph2;
    end

end
%--------------------------------------------------------------------------


%Calculate the change in a - if very small  return 0
%%!!!!!NOTE!! if eps not small enought it may stop the algorithm early!!!!!!
%--------------------------------------------------------------------------
if (abs(a2-alph2)<eps*(a2+alph2))
    %fprintf('a2-alpha2:%e ==0 \n',abs(a2-alph2));
    retval=0;
    %fprintf('k:%f %f %f delta:%f eps:%f\n',k,abs(a2-alph2),eps*(a2+alph2+eps),abs(delta),eps);
    return
end
%--------------------------------------------------------------------------


% computation on new a1pha1(a1)
%--------------------------------------------------------------------------
a1 = alph1 + (s*(alph2-a2));
%--------------------------------------------------------------------------
%fprintf('step start:low:%f i_low:%f cache_low:%f up:%f i_up:%f cache_up:%f\n',glob.b_low,glob.i_low,glob.fcache(glob.i_low),glob.b_up,glob.i_up,glob.fcache(glob.i_up));
%fprintf('delat:%f\n',alph2-a2);
%Update weight vector to reflect change in a1 & a2, if linear SVM
%--------------------------------------------------------------------------
if (strcmpi(kernel, 'knLin') == 1)
    w = w + (y1 * (a1 - alph1) * X(:,i1)) + (y2 * (a2 - alph2) * X(:,i2));
end
%--------------------------------------------------------------------------


%Store a1 and a2 in the alpha array
%--------------------------------------------------------------------------
alpha(i1) = a1;
alpha(i2) = a2;
%--------------------------------------------------------------------------

%Update fcache[i] for i in I_0 using new Lagrange multipliers
%--------------------------------------------------------------------------
k = length(glob.I_0);
for i = 1 : k
    ki1 = K(glob.I_0(i),i1);
    ki2 = K(glob.I_0(i),i2);
    evals = evals + 2;
    glob.fcache(glob.I_0(i)) = glob.fcache(glob.I_0(i)) + (y1 * (a1 - alph1) * ki1) + (y2 * (a2 - alph2) * ki2);
end
%--------------------------------------------------------------------------


%The update below is simply achieved by keeping and updating information
%about alpha_i being at 0, C or in between them. Using this together with
%target[i] gives information as to which index set i belongs.
% Update I_0, I_1, I_2, I_3, I_4
%--------------------------------------------------------------------------
for i=1:size(alpha,2)
    if(alpha(i) < eps)
        alpha(i) = 0;
    elseif(alpha(i) > C - eps)
        alpha(i) = C;
    end
end
glob.I_0 = find(alpha > 0 & alpha < C);
glob.I_1 = find(alpha(glob.v_1) == 0);
glob.I_1 = glob.v_1(glob.I_1);
glob.I_2 = find(alpha(glob.v_2) == C);
glob.I_2 = glob.v_2(glob.I_2);
glob.I_3 = find(alpha(glob.v_1) == C);
glob.I_3 = glob.v_1(glob.I_3);
glob.I_4 = find(alpha(glob.v_2) == 0);
glob.I_4 = glob.v_2(glob.I_4);
%--------------------------------------------------------------------------




%Compute updated F values for i1 and i2
%--------------------------------------------------------------------------
glob.fcache(i1) = F1 + (y1 * (a1 - alph1) * k11) + (y2 * (a2 - alph2) * k12);
glob.fcache(i2) = F2 + (y1 * (a1 - alph1) * k12) + (y2 * (a2 - alph2) * k22);
%--------------------------------------------------------------------------


%Compute (i_low, b_low) and (i_up, b_up),
%using only i1, i2 and indices in I_0

%--------------------------------------------------------------------------
%--GIA TO i1 -------------------------------------------------------
v = find(glob.I_1==i1);
i1_in_I_1 = length(v);
v = find(glob.I_2==i1);
i1_in_I_2 = length(v);
v = find(glob.I_3==i1);
i1_in_I_3 = length(v);
v = find(glob.I_4==i1);
i1_in_I_4 = length(v);

%%--------------------------------------------------------------------------
%--GIA TO i2 -------------------------------------------------------
v = find(glob.I_1==i2);
i2_in_I_1 = length(v);
v = find(glob.I_2==i2);
i2_in_I_2 = length(v);
v = find(glob.I_3==i2);
i2_in_I_3 = length(v);
v = find(glob.I_4==i2);
i2_in_I_4 = length(v);


% 1)First Compute i_low, i_up for I_0
%--------------------------------------------------------------------------
glob.b_up = inf;
glob.b_low = -inf;
if size(glob.I_0)~=0   % Trying to run the smo mod1 for the alult datasets I diskovered that for small values of C there was this problem.
    [glob.b_up, glob.i_up] = min(glob.fcache(glob.I_0));
    glob.i_up=glob.I_0(glob.i_up);
    if size(glob.i_up)~=1
        glob.i_up = glob.i_up(1);
    end
    [glob.b_low, glob.i_low] = max(glob.fcache(glob.I_0));
    glob.i_low=glob.I_0(glob.i_low);
    if size(glob.i_low)~=1
        glob.i__low = glob.i_low(1);
    end
end



%2)Then check if i1 or i2 should replace i_up or i_low

%2a) For i1

if ( (glob.b_up>glob.fcache(i1)) && (i1_in_I_1+i1_in_I_2) )
    glob.b_up = glob.fcache(i1);
    glob.i_up = i1;
end
if ((glob.b_low <glob.fcache(i1))&&(i1_in_I_3+i1_in_I_4))
    glob.b_low = glob.fcache(i1);
    glob.i_low = i1;
end

%2b) For i2

if ((glob.b_up > glob.fcache(i2))&&(i2_in_I_1+i2_in_I_2))
    glob.b_up = glob.fcache(i2);
    glob.i_up = i2;
end
if ((glob.b_low <glob.fcache(i2))&&(i2_in_I_3+i2_in_I_4))
    glob.b_low = glob.fcache(i2);
    glob.i_low = i2;
end

retval = 1;

return
%--------------------------------------------------------------------------
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cul,retval, alpha, w, b, stp, evals, glob] =...
    examineExampleK(i2, glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, steps, stp, evals, eps,K)

%drawnow; %%used to give Matlab the opportunity to examine any pending ctrl+C (while in deep loops)
if (get(0,'PointerLocation')==[1 1])
    %disp('Press <ctrl+c> to stop or any key to interrupt execution temporarily ...');
    %pause;
    %disp('Type "return" to carry on');
    keyboard;
end
cul = 0;
retval = 0;
[D n] = size(X);
y2 = Y(i2);
alph2 = alpha(i2);
v = find(glob.I_0==i2);
i2_in_I_0 = length(v);
%fprintf('example start:low:%f i_low:%f cache_low:%f up:%f i_up:%f cache_up:%f\n',glob.b_low,glob.i_low,glob.fcache(glob.i_low),glob.b_up,glob.i_up,glob.fcache(glob.i_up));
if (i2_in_I_0 > 0)
    F2 = glob.fcache(i2);
else
    ki2 = K(:,i2);
    evals = evals + n;
    F2 = -y2 + ((Y .* alpha) * ki2);
    glob.fcache(i2) = F2;
end
%fprintf('example end:low:%f i_low:%f cache_low:%f up:%f i_up:%f cache_up:%f\n',glob.b_low,glob.i_low,glob.fcache(glob.i_low),glob.b_up,glob.i_up,glob.fcache(glob.i_up));
%--GIA TO i2 -------------------------------------------------------
v = find(glob.I_1==i2);
i2_in_I_1 = length(v);
v = find(glob.I_2==i2);
i2_in_I_2 = length(v);
v = find(glob.I_3==i2);
i2_in_I_3 = length(v);
v = find(glob.I_4==i2);
i2_in_I_4 = length(v);

if ((i2_in_I_1 + i2_in_I_2 > 0) && (F2 < glob.b_up))
    glob.b_up = F2;
    glob.i_up = i2;
elseif ((i2_in_I_3 + i2_in_I_4 > 0) && (F2 > glob.b_low))
    glob.b_low = F2;
    glob.i_low = i2;
end
%------------------------------------------------------------------


%Chech optimality using current b_low and b_up and, if
%violated, find an index i1 to do joint optimization with i2
%------------------------------------------------------------------
optimality = 1;
if ((i2_in_I_0 + i2_in_I_1 + i2_in_I_2) > 0)
    if ( (glob.b_low - F2) > (2*tol) )
        optimality = 0;
        i1 = glob.i_low;

    end
end
if ((i2_in_I_0 + i2_in_I_3 + i2_in_I_4) > 0)
    if ((F2 - glob.b_up) > (2*tol))
        optimality = 0;
        i1 = glob.i_up;

    end
end
if (optimality == 1)
    retval = 0;
    return;
end
%------------------------------------------------------------------


%For i2 in I_0 choose the better i1
%------------------------------------------------------------------

if (i2_in_I_0 > 0)
    if ((glob.b_low - F2) > (F2 - glob.b_up))
        i1 = glob.i_low;
    else
        i1 = glob.i_up;
    end
end
%------------------------------------------------------------------
cul = 1;
stp = stp + 1;
[retval, alpha, w, b, evals, glob] =...
    takeStepK(i1, i2, glob, alpha, w, b, X, Y, kernel, kpar1, kpar2, C, tol, evals, eps,K);
%fprintf('retval:%d\n',retval);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
