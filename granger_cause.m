function [val, P] = granger_cause(y,x,alpha,max_lag)
% I added the correction for multiple comparison
% I also corrected the loop to calculate y_lag for optimum BIC. 

% [F,c_v] = granger_cause(x,y,alpha,max_lag)
% Granger Causality test
% Does Y Granger Cause X?
%
% User-Specified Inputs:
%   x -- A column vector of data
%   y -- A column vector of data
%   alpha -- the significance level specified by the user
%   max_lag -- the maximum number of lags to be considered. It is also
%   called the order
% User-requested Output:
%   F -- The value of the F-statistic
%   c_v -- The critical value from the F-distribution
%
% The lag length selection is chosen using the Bayesian information
% Criterion 
% Note that if F > c_v we reject the null hypothesis that y does not
% Granger Cause x

% Chandler Lutz, UCR 2009
% Questions/Comments: chandler.lutz@email.ucr.edu
% $Revision: 1.0.0 $  $Date: 09/30/2009 $
% $Revision: 1.0.1 $  $Date: 10/20/2009 $
% $Revision: 1.0.2 $  $Date: 03/18/2009 $

% References:
% [1] Granger, C.W.J., 1969. "Investigating causal relations by econometric
%     models and cross-spectral methods". Econometrica 37 (3), 424438.

% Acknowledgements:
%   I would like to thank Mads Dyrholm for his helpful comments and
%   suggestions


% y = rand(1000,1);
% x = rand(1000,1);
% alpha = 0.05;
% max_lag = 20;


% we will be checking if chan1 causes chan2 or not

%Make sure x & y are the same length
if (length(x) ~= length(y))
    error('x and y must be the same length');
end

%Make sure x is a column vector
[a,b] = size(x);
if (b>a)
    %x is a row vector -- fix this
    x = x';
end

%Make sure y is a column vector
[a,b] = size(y);
if (b>a)
    %y is a row vector -- fix this
    y = y';
end

%         x = x + 10*rand(size(x,1),1);
%         y = y + 10*rand(size(y,1),1);


%Make sure max_lag is >= 1
if max_lag < 1
    error('max_lag must be greater than or equal to one');
end

%making the data stationary
%         x = diff(x,3);
%         y = diff(y,3);

%First find the proper model specification using the Bayesian Information
%Criterion for the number of lags of x

T = length(x);

BIC = zeros(max_lag,1);

%Specify a matrix for the restricted RSS
RSS_R = zeros(max_lag,1); 
i = 1;
while i <= max_lag
    ystar = x(i+1:T,:);
    xstar = [ones(T-i,1) zeros(T-i,i)];
    %Populate the xstar matrix with the corresponding vectors of lags
    j = 1;
    while j <= i
        xstar(:,j+1) = x(i+1-j:T-j);
        j = j+1;
    end
    %Apply the regress function. b = betahat, bint corresponds to the 95%
    %confidence intervals for the regression coefficients and r = residuals
    [b,bint,r] = regress(ystar,xstar);

%             eig_value = eig(xstar);
%             max(eig_value)/min(eig_value)

    %Find the bayesian information criterion, under the assumption that the model errors or disturbances are independent and identically distributed according to a normal distribution and that the boundary condition that the derivative of the log likelihood with respect to the true variance is zero,
    BIC(i,:) = T*log(r'*r/T) + (i+1)*log(T);

    %Put the restricted residual sum of squares in the RSS_R vector
    RSS_R(i,:) = r'*r;

    i = i+1;

end

[dummy,x_lag] = min(BIC);

%First find the proper model specification using the Bayesian Information
%Criterion for the number of lags of y

BIC = zeros(max_lag,1);

%Specify a matrix for the unrestricted RSS
RSS_U = zeros(max_lag,1);

i = 1;
while i <= max_lag

    ystar = x(i+x_lag:T,:);   % In my opinion, there will not be 1 in the first argument
    xstar = [ones(T-(i+x_lag)+1,1)  zeros(T-(i+x_lag)+1,x_lag+i)];
    %Populate the xstar matrix with the corresponding vectors of lags of x
    j = 1;
    while j <= x_lag
        xstar(:,j+1) = x(i+x_lag-j:T-j,:);
        j = j+1;
    end
    %Populate the xstar matrix with the corresponding vectors of lags of y
    j = 1;
    while j <= i
        xstar(:,x_lag+j+1) = y(i+x_lag-j:T-j,:);
        j = j+1;
    end
    %Apply the regress function. b = betahat, bint corresponds to the 95%
    %confidence intervals for the regression coefficients and r = residuals
    [b,bint,r] = regress(ystar,xstar);

%             eig_value = eig(xstar);
%             max(eig_value)/min(eig_value)

    %Find the bayesian information criterion
    BIC(i,:) = T*log(r'*r/T) + (i+1)*log(T);

    RSS_U(i,:) = r'*r;

    i = i+1;

end

[dummy,y_lag] = min(BIC);

%The numerator of the F-statistic
v1 = y_lag; % dof of numerator
F_num = ((RSS_R(x_lag,:) - RSS_U(y_lag,:))/v1);

%The denominator of the F-statistic
v2 = T-(x_lag+y_lag+1); % dog of denomenator
F_den = RSS_U(y_lag,:)/v2;

%The F-Statistic
F = F_num/F_den;

alpha = 1 - (1-alpha)^(1/y_lag);  % for multiple channel correction

c_v = finv(1-alpha,y_lag,(T-(x_lag+y_lag+1)));

p = 1 - fcdf(F,v1,v2);

comparisons = 1 + ((1-p)^2)*log(max_lag);
P = 1.0 - ( 1.0 - p)^comparisons  ; %multiple comparison correction

val = F-c_v;  % measure for chan1 causing chan2


