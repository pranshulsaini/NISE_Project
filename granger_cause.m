function [val, p, p_corr, x_lag, y_lag, BIC_R, BIC_U, F_den] = granger_cause(y,x,alpha,x_lag0, y_lag0, x_max_lag, y_max_lag, user_spec, down_samp)
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

%Downsampling
if down_samp 
    x = downsample(x,4,3);  % phase offset 3
    y = downsample(y,4,3); % phase offset 3
end


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
if ((x_max_lag < 1)||(y_max_lag < 1))
    error('max_lag must be greater than or equal to one');
end

%making the data stationary
%         x = diff(x,3);
%         y = diff(y,3);

%First find the proper model specification using the Bayesian Information
%Criterion for the number of lags of x

T = length(x);

BIC_R = zeros(x_max_lag,1);

%Specify a matrix for the restricted RSS
RSS_R = zeros(x_max_lag,1); 

ystar_R_len = zeros(x_max_lag,1);    

i = 1;

if user_spec
    i = x_lag0;
    x_max_lag = x_lag0;
end

while i <= x_max_lag
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

    n_obs = length(ystar);
    ystar_R_len(i) = n_obs;

    %Find the bayesian information criterion, under the assumption that the model errors or disturbances are independent and identically distributed according to a normal distribution and that the boundary condition that the derivative of the log likelihood with respect to the true variance is zero,
    BIC_R(i,:) = n_obs*log(r'*r/n_obs) + (i+1)*log(n_obs);

    %Put the restricted residual sum of squares in the RSS_R vector
    RSS_R(i,:) = r'*r;

    i = i+1;

end


[dummy,x_lag] = min(BIC_R);

if user_spec
    x_lag = x_lag0;
end


%First find the proper model specification using the Bayesian Information
%Criterion for the number of lags of y


BIC_U = zeros(y_max_lag,1);
%Specify a matrix for the unrestricted RSS
RSS_U = zeros(y_max_lag,1);

ystar_U_len = zeros(y_max_lag,1);  % will be used in DOF of numerator


i = 1;

if user_spec
    i = y_lag0;
    y_max_lag = y_lag0;
end

while i <=y_max_lag

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

    n_obs = length(ystar);
    ystar_U_len(i) = n_obs;


    %Find the bayesian information criterion
    BIC_U(i,:) = n_obs*log(r'*r/n_obs) + (i+1)*log(n_obs);

    RSS_U(i,:) = r'*r;


    i = i+1;

end

[dummy,y_lag] = min(BIC_U);


if user_spec
    y_lag = y_lag0;
end


%The numerator of the F-statistic
v1 = y_lag; % dof of numerator
if (ystar_U_len(y_lag) == ystar_R_len(x_lag) )
    % no missing data
    % warning(' N constant = %d  ',  Nvalid_U(y_lag)   );
    F_num = ((RSS_R(x_lag,:) - RSS_U(y_lag,:))/v1);

else
    % we have missing data
    Tave = ( ystar_U_len(y_lag) + ystar_R_len(x_lag) )  / 2.0 ;
    tmp_MSS_R =  RSS_R(x_lag) / ystar_R_len(x_lag);
    tmp_MSS_U =  RSS_U(y_lag) / ystar_U_len(y_lag);
    
    F_num  =   ( Tave * ( tmp_MSS_R -  tmp_MSS_U) ) / v1 ;
    
end


%The denominator of the F-statistic
v2 = ystar_U_len(y_lag)-(x_lag+y_lag+1); % dog of denomenator, Modified by looking at granger_cause_1
F_den = RSS_U(y_lag,:)/v2;

% sanity checks
if (F_num < 0 )
    warning('F num is negative %8.3e', F_num);
end

if (F_den < 0 )
    warning('F den  is negative %8.3E', F_den);
end

% 
% x_lag
% y_lag
% ystar_U_len(y_lag)
% v2

%The F-Statistic
if (F_den ~= 0 )  % sanity check
    F = F_num/F_den;

    alpha = 1 - (1-alpha)^(1/y_lag);  % for multiple comparison correction

    c_v = finv(1-alpha,v1,v2);

    p = 1 - fcdf(F,v1,v2);

    comparisons = 1 + ((1-p)^2)*log(y_max_lag);
    p_corr = 1.0 - ( 1.0 - p)^comparisons  ; %multiple comparison correction

    val = F;%-c_v;  % measure for chan1 causing chan2

else 
    F = 0;
    alpha = 0;
    c_v = 0;
    p = 0;
    p_corr = 0;
    val = 0;
end





