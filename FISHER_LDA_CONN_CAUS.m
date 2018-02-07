al%% This code uses fisherrank function to rank the features and then LDA classifier to classify the classes
% I am also using connectivity feature to reach a better classification

%% Loading EEGLAB
clc; clear all;
addpath('D:\TUM\3rd Sem\Neuroinspired System Engineering\eeglab14_1_1b');
addpath('C:\Users\user\Dropbox\Semester3\Neuro Inspired Engineering\Project_Pranshul\functions');
eeglab;

%% Loading of data and creating epochs
clc;
EEG = pop_loadset('filename','calibration.set','filepath','../data/');

% S5 are errors and S4 are no_errors. This information is
% in the third row of every trial (EEG.event)

eegplot(EEG.data,'eloc_file',EEG.chanlocs);   % plotting eeg data

EEG_epo_noError = pop_epoch(EEG,{'S  4'},[0.0 1.0]); % retrieving the desired epoch with reference as S4 when there was no error

EEG_epo_Error = pop_epoch(EEG,{'S  5'},[0.0 1.0]); % retrieving the desired epoch with reference as S5 when there was error

n_chan = size(EEG_epo_noError.data, 1);
n_no_error =  size(EEG_epo_noError.data, 3);
n_error = size(EEG_epo_Error.data, 3);
total_trials = n_no_error + n_error; 

data_noError = EEG_epo_noError.data;
data_Error = EEG_epo_Error.data;
data = cat(3,data_noError,data_Error);


labels = [-1*ones(n_no_error,1); 1*ones(n_error,1)];

%% Time course Visualisation
clc;
avg_no_error = mean(EEG_epo_noError.data,3);  % mean across all trials
%avg_no_error = EEG_epo_noError.data;
cz_avg_no_error = avg_no_error(21,:); %Cz channel is at the center of the head. index is 20

avg_error = mean(EEG_epo_Error.data,3); % mean across all trials
%avg_error = EEG_epo_Error.data;
cz_avg_error = avg_error(21,:);   %Cz channel is at the center of the head. index is 20

figure;
plot(EEG_epo_noError.times,cz_avg_no_error);
hold on;
plot(EEG_epo_Error.times,cz_avg_error);
legend('No error','error')
title('ERP time course of channel C4')
xlabel('time locked to key press (ms)') % x-axis label
ylabel('Signal (microvolts') % y-axis label


%% Sanity check for granger causality
% Even for random values, F makes dip after a certain time lag. This is just noise. SO, I will only use P now onwards for granger causality
noise1 =  rand(1,1000);
noise2 =  rand(1,1000);
[cz_caus,cz_P,cz_P_corr, cz_x_lag, cz_y_lag_init, cz_BIC_R, cz_BIC_U, F_den] = granger_cause(noise1,noise2, 0.05, 5, 5, 5, 20,0,1); % transpose make channels as columns
            
figure;
subplot(2,1,1);
plot(cz_BIC_R);
hold on;

subplot(2,1,2);
plot(cz_BIC_U);



%% Topographic Visualisation
clc;
time_pt_1 = zeros(size(avg_error,1),1);
time_pt_2 = zeros(size(avg_error,1),1);

avg_tp1_no_error = zeros(size(avg_error,1),1);
avg_tp1_error = zeros(size(avg_error,1),1);
avg_tp2_no_error = zeros(size(avg_error,1),1);
avg_tp2_error = zeros(size(avg_error,1),1);

for i = 1:size(avg_error,1)  % finding max diff points for every channel
    [~,time_pt_1(i)] = max(avg_no_error(i,:) - avg_error(i,:));
    [~,time_pt_2(i)] = min(avg_no_error(i,:) - avg_error(i,:));
    avg_tp1_no_error(i) = avg_no_error(i,time_pt_1(i));
    avg_tp2_no_error(i) = avg_no_error(i,time_pt_2(i));
    avg_tp1_error(i) = avg_error(i,time_pt_1(i));
    avg_tp2_error(i) = avg_error(i,time_pt_2(i));
end


figure;   % green color would represent 0. Towards red = positive; Towards blue = negative
ax1 = subplot(2,2,1);
title(ax1,'no Error time point 1');
topoplot(avg_tp1_no_error,EEG.chanlocs,'maplimits',[-4 4], 'conv', 'on');

ax2 = subplot(2,2,2);
title(ax2,'no Error time point 2');
topoplot(avg_tp2_no_error,EEG.chanlocs,'maplimits',[-4 4],  'conv', 'on');

ax3 = subplot(2,2,3);
title(ax3,'Error time point 1');
topoplot(avg_tp1_error,EEG.chanlocs, 'maplimits',[-4 4], 'conv', 'on');

ax4 = subplot(2,2,4);
title(ax4,'Error time point 2');
topoplot(avg_tp2_error,EEG.chanlocs,'maplimits',[-4 4], 'conv', 'on');



%% Creation of temporal feature vector

clc;
no_error_tp1 = zeros(size(EEG_epo_noError.data, 3), size(EEG_epo_noError.data, 1)); % number of trials x number of channels
no_error_tp2 = zeros(size(EEG_epo_noError.data, 3), size(EEG_epo_noError.data, 1)); % number of trials x number of channels
error_tp1 = zeros(size(EEG_epo_Error.data, 3), size(EEG_epo_Error.data, 1)); % number of trials x number of channels
error_tp2 = zeros(size(EEG_epo_Error.data, 3), size(EEG_epo_Error.data, 1)); % number of trials x number of channels

for i = 1:size(avg_error,1) 
    no_error_tp1(:,i) = EEG_epo_noError.data(i,time_pt_1(i),:);
    no_error_tp2(:,i) = EEG_epo_noError.data(i,time_pt_2(i),:);
    
    error_tp1(:,i) = EEG_epo_Error.data(i,time_pt_1(i),:);
    error_tp2(:,i) = EEG_epo_Error.data(i,time_pt_2(i),:);
   
end

no_error = [no_error_tp1 no_error_tp2]; % dimension = no_error_trials x (2*channels)
error = [error_tp1 error_tp2]; %  dimension = error_trials x (2*channels)


feature_mat = [no_error; error];

%standardization
mean_feature_mat = mean(feature_mat);  % dimension = 4*channels
mean_feature_mat = repmat(mean_feature_mat,total_trials,1);  % 300 are the total number of trials
std_feature_mat = std(feature_mat);
std_feature_mat = repmat(std_feature_mat, total_trials,1);
feature_mat = (feature_mat - mean_feature_mat)./std_feature_mat;

figure;  % just to see if the data is normally distributed or not. Otherwise we could do log normalisation
histogram(feature_mat(:,47));  % It appears that the data is approximately normally distribued. So log normal transformation is not needed. Moreover, it can't be used here because we can't feed negative values to log function
title('Approximatly gaussian data (Log normal transform not required)')



%% plotting rank wise

[v, rank] = fisherrank(feature_mat, labels);
figure;
plot(1:54, v);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label

%scatter plot of best and second best features
figure;
scatter(no_error(:,rank(1)), no_error(:,rank(2)), 'b');
hold on;
scatter(error(:,rank(1)), error(:,rank(2)), 'r');
legend('No error,-1','error,1');
xlabel('1st best feature'); % x-axis label
ylabel('2nd best feature'); % y-axis label

%% Creating a classifier

n_feat = size(feature_mat,2);
class_error = zeros(100,round(total_trials/10),10);  % max number of features allowed: n_feat/2

% choosing best lambda and optimum number of features using Kfold crossvalidation
for i =1:100   % I am multiplying in the for loop for 0.01. So it is 0.01 change per loop   
    for j = 1:round(total_trials/10) % across all the features
        indices = crossvalind('Kfold', labels, 10); % for 10 fold cv
        for k = 1:10
            test = (indices == k); train = ~test;
            test_indx = find(test == 1);
            train_indx = find(train == 1);
            model = trainShrinkLDA(feature_mat(train_indx,rank(1:j)),labels(train_indx,:),i*0.01);
            [y] = predictShrinkLDA(model,feature_mat(test_indx,rank(1:j)));
            class_error(i,j,k) = sum(y' ~= labels(test_indx,:));   % summing up the instants when both are not equal
        end
    end
end

avg_class_error = median(class_error,3);  % doing mean across the k fold 10 values
lambda_set = 0.01:0.01:1.0;
[M,I] = min(avg_class_error); %it will give the index of min value in the array
[~,n] = min(M);  % n is the column containing the min value of the matrix. This is the number of features
m = I(n);  % m is the row containing the min value of the matrix
lambda = lambda_set(m);  % lamda value corresponding to min error

figure;
[X,Y] = meshgrid(1:round(total_trials/10),lambda_set);
surf(X,Y,avg_class_error);
xlabel('Number of features'); % x-axis label
ylabel('Lambda value'); % y-axis label
zlabel('Average error');


% Creating final feature matrix 
feature_mat_red = zeros(300,n);
for i = 1:n
    feature_mat_red(:,i) = feature_mat(:,rank(i));        % feature matrix with optimum number of features
end

%% Connectivity features (correlation measure [-1,1])

conn_mat = zeros(total_trials, n_chan, n_chan);
P = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 

%for input matrix of corrcoed, columnsrepresent random variables and the rows represent observations
for i = 1:n_no_error % there will be a connectivity matrix for every no error trial
    [conn_mat(i,:,:),P(i,:,:)] = corrcoef(squeeze(data_noError(:,:,i))'); % transpose make channels as columns
end

for i = 1:n_error % there will be a connectivity matrix for every error trial
    [conn_mat(i+n_no_error,:,:),P(i+n_no_error,:,:)] = corrcoef(squeeze(data_Error(:,:,i))'); % 
end

conn_fish_val = zeros(n_chan,n_chan);


for i = 1:n_chan
    for j = 1:n_chan
        x = squeeze(conn_mat(:,i,j));
        mu = mean(x);
        thresh = std(x) * sqrt(2*log(2));
        indx = (abs(x-mu)<thresh);  % we want to reject outliers
        x = x(indx);
        y = labels(indx);
        [conn_fish_val(i,j),~] = fisherrank(x,y);
    end
end
conn_fish_val(isnan(conn_fish_val))= 0;  % this was done because there were some NaN values. 



% choosing  optimum number of features using Kfold crossvalidation
class_error = zeros(round(total_trials/10),10);  % max number of features allowed: n_feat/2
for j = 1:round(total_trials/10) % across all the features
    indices = crossvalind('Kfold', labels, 10); % for 10 fold cv
    for k = 1:10
        test = (indices == k); train = ~test;
        test_indx = find(test == 1);
        train_indx = find(train == 1);
        
        [~, locs_conns] = maxNvalues(conn_fish_val,j);
        conn_mat_final_train = zeros(270,j);
        conn_mat_final_test = zeros(30,j);
        for i = 1: j
            conn_mat_final_train(:,i) = conn_mat(train_indx,locs_conns(i,1),locs_conns(i,2));
            conn_mat_final_test(:,i) = conn_mat(test_indx,locs_conns(i,1),locs_conns(i,2));
        end

        model = trainShrinkLDA(conn_mat_final_train,labels(train_indx,:),lambda);
        [y] = predictShrinkLDA(model,conn_mat_final_test);
        class_error(j,k) = sum(y' ~= labels(test_indx,:));   % summing up the instants when both are not equal
    end
end


avg_class_error = median(class_error,2);  % doing mean across the k fold 10 values
[~,n_conn_feat] = min(avg_class_error); %it will give the index of min value in the array


[max_conn, locs_conn] = maxNvalues(conn_fish_val,n_conn_feat);

conn_mat_final = zeros(total_trials,n_conn_feat);
for i = 1: n_conn_feat
    conn_mat_final(:,i) = conn_mat(:,locs_conn(i,1),locs_conn(i,2));
end

figure;
plot(1:n_conn_feat, max_conn);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label


conn_mat_final_mean_no_error = squeeze(median(conn_mat_final(1:n_no_error,:)))';  % needed for topoplot
conn_mat_final_mean_error = squeeze(median(conn_mat_final(n_no_error+1:total_trials,:)))';  % needed for topoplot

%% Plotting the connection along with their correlation strengths
ds.chanPairs = locs_conn;
ds.connectStrength = conn_mat_final_mean_no_error;
figure;
subplot(1,2,1);
topoplot_connect(ds, EEG.chanlocs);
title('No-error connectivity map');
colorbar;

ds.connectStrength = conn_mat_final_mean_error;
subplot(1,2,2);
topoplot_connect(ds, EEG.chanlocs);
title('Error connectivity map');
colorbar;

%% Plotting the connection along with their Fisher score
best_fisher_conn = zeros(n_conn_feat,1);
for i = 1: n_conn_feat
    best_fisher_conn(i) = conn_fish_val(locs_conn(i,1),locs_conn(i,2));
end
ds.chanPairs = locs_conn;
ds.connectStrength = best_fisher_conn;
figure;
topoplot_connect(ds, EEG.chanlocs);
title('Fisher scores Topoplot');
colorbar;

 


%% Loading data for granger causality
% clc;
% EEG = pop_loadset('filename','calibration.set','filepath','../data/');
% 
% EEG_epo_noError = pop_epoch(EEG,{'S  4'},[0.0 1.0]); % retrieving the desired epoch with reference as S4 when there was no error
% 
% EEG_epo_Error = pop_epoch(EEG,{'S  5'},[0.0 1.0]); % retrieving the desired epoch with reference as S5 when there was error
% 
% data_noError = EEG_epo_noError.data;
% data_Error = EEG_epo_Error.data;
% data = cat(3,data_noError,data_Error);
% 
% n_chan = size(EEG_epo_noError.data, 1);
% n_no_error =  size(EEG_epo_noError.data, 3);
% n_error = size(EEG_epo_Error.data, 3);
% total_trials = n_no_error + n_error; 

%% Forming stationary data. Example shown for the average for Cz channel
%https://machinelearningmastery.com/time-series-data-stationary-python/

clc;
sample_data_no_err = data(20,:,1);
sample_data_err = data(20,:,200);
[~,pValue] = adftest(sample_data_no_err );
fprintf('The p value for the original no-error data is: %f \n', pValue);
[~,pValue] = adftest(sample_data_err);
fprintf('The p value for the original error data is: %f \n', pValue);

sample_data_no_error_diff = diff(sample_data_no_err,1);
sample_data_error_diff = diff(sample_data_err ,1);
[~,pValue] = adftest(sample_data_no_error_diff);
fprintf('The p value for the differenced no-error data is: %f \n', pValue);
[~,pValue] = adftest(sample_data_error_diff);
fprintf('The p value for the differenced error data is: %f \n', pValue);

figure;
subplot(2,1,1)
p1 = plot(EEG_epo_Error.times,sample_data_no_err );
hold on;
p2 = plot(EEG_epo_Error.times,sample_data_err);
xlabel('time locked to key press (ms)') % x-axis label
ylabel('Signal (microvolts') % y-axis label
legend([p1 p2],'no-error','error')

subplot(2,1,2)
p1= plot(EEG_epo_Error.times(2:end),sample_data_no_error_diff );
hold on;
p2 = plot(EEG_epo_Error.times(2:end),sample_data_error_diff);
xlabel('time locked to key press (ms)') % x-axis label
ylabel('Signal (microvolts') % y-axis label
legend([p1 p2],'no-error','error')

%making the whole data stationary.
% difference of order 1 is sufficient for individual trials
order = 1;
pValues = ones(n_chan,total_trials);
data_st = zeros(n_chan, size(data,2)-order, total_trials);
for i = 1:n_chan
    for j = 1: total_trials
        data_st(i,:,j) =  diff(data(i,:,j),order);
        [~,pValues(i,j)] = adftest(data_st(i,:,j));
    end
end

data_noError_st = data_st(:,:,1:190);
data_Error_st = data_st(:,:,191:300);

%% Granger causality feature matrix

clc;
x_max_lag = 15;
y_max_lag = 15;
down_samp = 0;
caus_mat = zeros(total_trials, n_chan, n_chan);  % the more the value, the more the causality
P = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 
P_corr = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value
x_lag = zeros(total_trials, n_chan, n_chan);
y_lag = zeros(total_trials, n_chan, n_chan);

data_noError_med = squeeze(median(data_noError_st,3)); 
data_Error_med = squeeze(median(data_Error_st,3));
caus_mat_init = zeros(total_trials,n_chan, n_chan);  % the more the value, the more the causality
P_init = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 
P_corr_init = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value
x_lag_init = zeros(total_trials, n_chan, n_chan);
y_lag_init = zeros(total_trials,n_chan, n_chan);
BIC_R_init = zeros(total_trials,x_max_lag, n_chan, n_chan);
BIC_U_init = zeros(total_trials,y_max_lag, n_chan, n_chan);

%for input matrix of corrcoed, columnsrepresent random variables and the rows represent observations
for i = 1: size(data_Error_st,1)    % chan1
    for j =  1: size(data_Error_st,1)  % chan2
        for k = 1:total_trials
            [caus_mat_init(k,i,j),P_init(k,i,j),P_corr_init(k,i,j), x_lag_init(k,i,j), y_lag_init(k,i,j), BIC_R_init(k,:,i,j), BIC_U_init(k,:,i,j),F_den] = granger_cause(squeeze(data_st(i,:,k)),squeeze(data_st(j,:,k)), 0.05, 5, 5, x_max_lag, y_max_lag,0,down_samp); % transpose make channels as columns
            
            if (F_den ~= 0)
                past_caus_mat_init = caus_mat_init(k,i,j);
                past_P_init = P_init(k,i,j);
                past_P_corr_init = P_corr_init(k,i,j);
                past_x_lag_init = x_lag_init(k,i,j);
                past_y_lag_init = y_lag_init(k,i,j);
                past_BIC_R_init = BIC_R_init(k,:,i,j);
                past_BIC_U_init = BIC_U_init(k,:,i,j);
                
            else
                caus_mat_init(k,i,j)=  past_caus_mat_init;
                P_init(k,i,j)= past_P_init;
                P_corr_init(k,i,j) =  past_P_corr_init ;
                x_lag_init(k,i,j) = past_x_lag_init;
                y_lag_init(k,i,j) = past_x_lag_init;
                BIC_R_init(k,:,i,j) = past_BIC_R_init ;
                BIC_U_init(k,:,i,j) = past_BIC_U_init;
                
            end
            if isnan(P_init(k,i,j)) == 1
                fprintf('P NaN');
                break
            end
        end
        if isnan(P_init(k,i,j)) == 1
            fprintf('P NaN');
            break
        end
    end
    if isnan(P_init(k,i,j)) == 1
        fprintf('P NaN');
        break
    end
end

x_lag_med = round(squeeze(median(x_lag_init,1)));
y_lag_med = round(squeeze(median(y_lag_init,1)));

save('15_15' );

%% plotting BIC 
factor = 3.90625;

if down_samp
    factor = 4 * 3.90625;
end

t_x = factor * [1:x_max_lag];
t_y = factor * [1:y_max_lag];

BIC_R_mean_no_err = squeeze(median(BIC_R_init(1:n_no_error,:,:,:),1));
BIC_U_mean_no_err = squeeze(median(BIC_U_init(1:n_no_error,:,:,:),1));
BIC_R_mean_err = squeeze(median(BIC_R_init(n_no_error+1:total_trials,:,:,:),1));
BIC_U_mean_err = squeeze(median(BIC_U_init(n_no_error+1:total_trials,:,:,:),1));

figure;
subplot(2,1,1);
for i = 1:size(data_Error,1)    % chan1
    for j = 1:size(data_Error,1)  % chan2
        
        if i~=j
            p1 = plot(t_x, BIC_R_mean_no_err(:,i,j), 'b');
            hold on
            p2 = plot(t_y, BIC_U_mean_no_err(:,i,j), 'r');
            hold on
        end
    end
end
xlabel('Lag (ms)');
ylabel('BIC');
title('No Error Trials')
legend([p1 p2],'Auto-reg','Causal')

subplot(2,1,2);
for i = 1:size(data_Error,1)    % chan1
    for j = 1: size(data_Error,1)  % chan2
        if i~=j
            p1 = plot(t_x, BIC_R_mean_err(:,i,j), 'k');
            hold on
            p2 = plot(t_y, BIC_U_mean_err(:,i,j), 'm');
            hold on
        end
        
    end
end
xlabel('Lag (ms)');
ylabel('BIC');
title(' Error Trials')
legend([p1 p2],'Auto-reg','Causal')

figure;
subplot(2,1,1);
hist(factor*x_lag_med(:));
xlabel('Lag (ms)');
ylabel('Count');
title(' Autoregression')

subplot(2,1,2);
hist(factor*y_lag_med(:));
xlabel('Lag (ms)');
ylabel('Count');
title(' Causal Entity')


%% plotting p distribution across non-diagonal elements 
% This is to check the causality in the data
P_init_off_diag_no_error = zeros(n_no_error, n_chan*n_chan-n_chan);
for i = 1: n_no_error
    idx = eye(n_chan,n_chan);
    Y = squeeze(P_init(i,:,:));
    P_init_off_diag_no_error(i,:) = Y(~idx);
end

figure;
subplot(2,1,1);
hist(P_init_off_diag_no_error(:))
title('P-dist for no-error trials')

P_init_off_diag_error = zeros(n_error, n_chan*n_chan-n_chan);
for i = 1: n_error
    idx = eye(n_chan,n_chan);
    Y = squeeze(P_init(n_no_error+i,:,:));
    P_init_off_diag_error(i,:) = Y(~idx);
end

subplot(2,1,2);
hist(P_init_off_diag_error(:))
title('P-dist for error trials')

figure;
P_init_med_no_error = squeeze(median(P_init(1:n_no_error,:,:),1));
idx = eye(n_chan,n_chan);
P_init_off_diag_no_error = P_init_med_no_error(~idx);
subplot(2,1,1);
hist(P_init_off_diag_no_error(:))
title('P-dist for median across no-error trials')

P_init_med_error = squeeze(median(P_init(n_no_error+1:total_trials,:,:),1));
idx = eye(n_chan,n_chan);
P_init_off_diag_error = P_init_med_error(~idx);
subplot(2,1,2);
hist(P_init_off_diag_error(:))
title('P-dist for median across error trials')


% for the channel showed least median p value. Let's check across trials
P_init_med  = squeeze(median(P_init,1));
[~,indx_minP] = maxNvalues(-P_init_med,1);

figure;
subplot(2,2,1);
for i = 1:n_no_error    % chan1
     
    p1 = plot(t_x, BIC_R_init(i,:,indx_minP(1),indx_minP(2)), 'b');
    hold on

end
xlabel('Lag (ms)');
ylabel('BIC');
title('minP chan comb, No Error Trials')
legend(p1,'Auto-reg')

subplot(2,2,2);
for i = 1:n_no_error    % chan1
     
    p2 = plot(t_y, BIC_U_init(i,:,indx_minP(1),indx_minP(2)), 'r');
    hold on

end
xlabel('Lag (ms)');
ylabel('BIC');
title('minP chan comb, No Error Trials')
legend( p2,'Causal')

subplot(2,2,3);
for i = n_no_error+1: total_trials    % chan1
    p1 = plot(t_x, BIC_R_init(i,:,indx_minP(1),indx_minP(2)),'b');
    hold on
end
xlabel('Lag (ms)');
ylabel('BIC');
title(' minP chan comb, Error Trials')
legend(p1,'Auto-reg')

subplot(2,2,4);
for i = n_no_error+1: total_trials    % chan1
    p2 = plot(t_y, BIC_U_init(i,:,indx_minP(1),indx_minP(2)),'r');
    hold on
end
xlabel('Lag (ms)');
ylabel('BIC');
title(' minP chan comb, Error Trials')
legend(p2,'Causal')


figure;
subplot(2,1,1);
plot(P_init(:,indx_minP(1),indx_minP(2)));
xlabel('trial number');
title('minP chan comb');

subplot(2,1,2);
hist(P_init(:,indx_minP(1),indx_minP(2)));
title('minP chan comb');

%% Calculating useful features with the help of fisher score
caus_fish_val = zeros(n_chan,n_chan);  % will contain fisher values

labels = [-1*ones(n_no_error,1); 1*ones(n_error,1)];

for i = 1:n_chan
    for j = 1:n_chan
        if i~=j  % only intersted in relationship between different channels
            x = squeeze(P_init(:,i,j));
            mu = mean(x);
            thresh = std(x) * sqrt(2*log(2));
            indx = (abs(x-mu)<thresh);  % we want to reject outliers
            x = x(indx);
            y = labels(indx);
            [caus_fish_val(i,j),~] = fisherrank(x,y);
        end
    end
end

caus_fish_val(isnan(caus_fish_val))= 0;  % this was done because there were some NaN values. 


% optimum number of features using Kfold crossvalidation
class_error = zeros(round(total_trials/10),10);  % max number of features allowed: n_feat/2
for j = 1:round(total_trials/10) % across all the features
    indices = crossvalind('Kfold', labels, 10); % for 10 fold cv
    for k = 1:10
        test = (indices == k); train = ~test;
        test_indx = find(test == 1);
        train_indx = find(train == 1);
        
        [~, locs_causs] = maxNvalues(caus_fish_val,j);
        caus_mat_final_train = zeros(270,j);
        caus_mat_final_test = zeros(30,j);
        
        for i = 1: j
            caus_mat_final_train(:,i) = P_init(train_indx,locs_causs(i,1),locs_causs(i,2));
            caus_mat_final_test(:,i) = P_init(test_indx,locs_causs(i,1),locs_causs(i,2));
        end

        model = trainShrinkLDA(caus_mat_final_train,labels(train_indx,:),lambda);
        [y] = predictShrinkLDA(model,caus_mat_final_test);
        class_error(j,k) = sum(y' ~= labels(test_indx,:));   % summing up the instants when both are not equal
    end
end


avg_class_error = median(class_error,2);  % doing mean across the k fold 10 values
[~,n_caus_feat] = min(avg_class_error); %it will give the index of min value in the array

[max_caus, locs_caus] = maxNvalues(caus_fish_val,n_caus_feat);
caus_mat_final = zeros(total_trials,n_caus_feat);
caus_mat_mean_no_err = zeros(1,n_caus_feat);
caus_mat_mean_err = zeros(1,n_caus_feat);

figure;
plot(1:n_caus_feat, max_caus);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label


for i = 1: n_caus_feat
    caus_mat_final(:,i) = P_init(:,locs_caus(i,1),locs_caus(i,2));
    caus_mat_mean_no_err(i) = squeeze(median(P_init(1:n_no_error,locs_caus(i,1),locs_caus(i,2)),1));
    caus_mat_mean_err(i) = squeeze(median(P_init(n_no_error+1:total_trials,locs_caus(i,1),locs_caus(i,2) ),1));
end

figure;
p1 = plot(caus_mat_mean_no_err,'b');
hold on;

p2 = plot(caus_mat_mean_err,'r');
xlabel('feature number');
ylabel('P value');
legend([p1 p2],'No-Error','error')



%% Plotting the connection along with their causal strengths
caus_mat_final_mean_no_error = squeeze(median(caus_mat_final(1:n_no_error,:)))';  % needed for topoplot
caus_mat_final_mean_error = squeeze(median(caus_mat_final(n_no_error+1:total_trials,:)))';  % needed for topoplot

ds.chanPairs = locs_caus;
ds.connectStrength = caus_mat_final_mean_no_error;
figure;
subplot(1,2,1)
topoplot_connect(ds, EEG.chanlocs);
title('No-error Causality Map');
colorbar;

ds.connectStrength = caus_mat_final_mean_error;
subplot(1,2,2)
topoplot_connect(ds, EEG.chanlocs);
title('Error Causality Map');
colorbar;

%% Plotting the connection along with their Fisher score
best_fisher_caus = zeros(n_caus_feat,1);
for i = 1: n_caus_feat
    best_fisher_caus(i) = caus_fish_val(locs_caus(i,1),locs_caus(i,2));
end
ds.chanPairs = locs_caus;
ds.connectStrength = best_fisher_caus;
figure;
topoplot_connect(ds, EEG.chanlocs);
title('Fisher scores Topoplot');
colorbar;



%% Testing the performance of features on training

% training based on only temporal features
model1 = trainShrinkLDA(feature_mat_red,labels,lambda);  % final model
[y] = predictShrinkLDA(model1,feature_mat_red);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal features is: %f \n', performance_error);

figure;
subplot(3,2,1)
plot(1:n, v(1:n));
ylabel('fisher score'); % x-axis label
% xlabel('ranked features'); % y-axis label
title('temporal (Training)')

% training based on only connectivity features
model2 = trainShrinkLDA(conn_mat_final,labels,lambda);  % final model
[y] = predictShrinkLDA(model2,conn_mat_final);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for connectivity features is: %f \n', performance_error);

[v2, rank2] = fisherrank(conn_mat_final, labels);
subplot(3,2,2)
plot(v2);
ylabel('fisher score'); % x-axis label
% xlabel('ranked features'); % y-axis label
title('conn (Training)')

% training based on only connectivity features
model3 = trainShrinkLDA(caus_mat_final,labels,lambda);  % final model
[y] = predictShrinkLDA(model3,caus_mat_final);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for casuality features is: %f \n', performance_error);

[v3, rank3] = fisherrank(caus_mat_final, labels);
subplot(3,2,3)
plot(v3);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('caus (Training)')


% training based on temporal + connectivity features
feature_mat_red_comb = [feature_mat_red caus_mat_final];
[v4, rank4] = fisherrank(feature_mat_red_comb, labels);
feature_mat_red_comb4 = zeros(300,30);
for i = 1:min(30,size(feature_mat_red_comb,2))
    feature_mat_red_comb4(:,i) = feature_mat_red_comb(:,rank4(i));        % feature matrix with optimum number of features
end
model4 = trainShrinkLDA(feature_mat_red_comb4,labels,lambda);  % final model
[y] = predictShrinkLDA(model4,feature_mat_red_comb4);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + casuality features is: %f \n', performance_error);

subplot(3,2,4)
plot(1:min(30,size(feature_mat_red_comb,2)), v4(1:min(30,size(feature_mat_red_comb,2))));
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + caus')


% training based on connectivity + causal features
feature_mat_red_comb = [feature_mat_red conn_mat_final ];
[v5, rank5] = fisherrank(feature_mat_red_comb, labels);
feature_mat_red_comb5 = zeros(300,30);
for i = 1:30
    feature_mat_red_comb5(:,i) = feature_mat_red_comb(:,rank5(i));        % feature matrix with optimum number of features
end
model5 = trainShrinkLDA(feature_mat_red_comb5,labels,lambda);  % final model
[y] = predictShrinkLDA(model5,feature_mat_red_comb5);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + connectivity features is: %f \n', performance_error);

subplot(3,2,5)
plot(1:30, v5(1:30));
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + conn')


% training based on temporal + connectivity + causal features
feature_mat_red_comb = [feature_mat_red conn_mat_final caus_mat_final];
[v6, rank6] = fisherrank(feature_mat_red_comb, labels);
feature_mat_red_comb6 = zeros(300,30);
for i = 1:30
    feature_mat_red_comb6(:,i) = feature_mat_red_comb(:,rank6(i));        % feature matrix with optimum number of features
end
model6 = trainShrinkLDA(feature_mat_red_comb6,labels,lambda);  % final model
[y] = predictShrinkLDA(model6,feature_mat_red_comb6);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + connectivity + causality features is: %f \n', performance_error);

subplot(3,2,6)
plot(1:30, v6(1:30));
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + conn + caus')


%% Recall dataset loading

recall_load = load('recall_ground_truth.mat');
recall_labels = recall_load.labels;

EEG_recall = pop_loadset('filename','recall.set','filepath','../data/');
EEG_epo_recall = pop_epoch(EEG_recall,{'S  6'},[0.0 0.5]); % Two spaces between S and 6. S6 in this file is the time when the response was made by the user

data_recall = EEG_epo_recall.data;
total_trials_recall =  size(EEG_epo_recall.data, 3);

%% %%%%%%%%%%%%%%%%%%%%% forming temporal feature matrix for Recall Set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_pt_1_recall = time_pt_1;
time_pt_2_recall = time_pt_2;
EEG_recall_tp1 = zeros(size(EEG_epo_recall.data, 3), size(EEG_epo_recall.data, 1)); % number of trials x number of channels
EEG_recall_tp2 = zeros(size(EEG_epo_recall.data, 3), size(EEG_epo_recall.data, 1)); % number of trials x number of channels

for i = 1:size(avg_error,1) 
    EEG_recall_tp1(:,i) = EEG_epo_recall.data(i,time_pt_1_recall(i),:);
    EEG_recall_tp2(:,i) = EEG_epo_recall.data(i,time_pt_2_recall(i),:);
       
end

feature_mat_recall = [EEG_recall_tp1 EEG_recall_tp2];

%standardisation
mean_feature_mat_recall = mean(feature_mat_recall);
mean_feature_mat_recall = repmat(mean_feature_mat_recall,300,1);
std_feature_mat_recall = std(feature_mat_recall);
std_feature_mat_recall = repmat(std_feature_mat_recall, 300,1);
feature_mat_recall = (feature_mat_recall - mean_feature_mat_recall)./std_feature_mat_recall;

%Dimensionality reduction
feature_mat_red_recall = zeros(300,n);
for i = 1:n
    feature_mat_red_recall(:,i) = feature_mat_recall(:,rank(i));
end

%%%%%%%%%%%%%%%%%%%forming connectivity feature matrix %%%%%%%%%%%%%%%%%%%%
conn_mat_recall = zeros(total_trials_recall, n_chan, n_chan);
P_conn_recall = zeros(total_trials_recall, n_chan, n_chan); % will indicate the significance of the correlation value 

for i = 1:total_trials_recall % there will be a connectivity matrix for every trial
    [conn_mat_recall(i,:,:),P_conn_recall(i,:,:)] = corrcoef(squeeze(data_recall(:,:,i))'); % transpose make channels as columns
end

conn_mat_final_recall = zeros(total_trials_recall,n_conn_feat);
for i = 1: n_conn_feat
    conn_mat_final_recall(:,i) = conn_mat_recall(:,locs_conn(i,1),locs_conn(i,2));
end


%%%%%%%%%%%%%%%%%%%%%%%% forming causality feature matrix %%%%%%%%%%%%%%%
caus_mat_recall = zeros(total_trials_recall, n_chan, n_chan);  % the more the value, the more the causality
P_recall = zeros(total_trials_recall, n_chan, n_chan); % will indicate the significance of the correlation value 
P_corr_recall = zeros(total_trials_recall, n_chan, n_chan); % will indicate the significance of the correlation value
x_lag_recall = zeros(total_trials_recall, n_chan, n_chan);
y_lag_recall = zeros(total_trials_recall, n_chan, n_chan);

%making the whole data stationary.
% difference of order 1 is sufficient for individual trials
order = 1;
pValues = ones(n_chan,total_trials);
data_recall_st = zeros(n_chan, size(data_recall,2)-order, total_trials);
for i = 1:n_chan
    for j = 1: total_trials
        data_recall_st(i,:,j) =  diff(data_recall(i,:,j),order);
        [~,pValues(i,j)] = adftest(data_recall_st(i,:,j));
    end
end


for i = 1:total_trials_recall % there will be a connectivity matrix for every no error trial
    for j = 1: size(data_Error,1)    % chan1
        for k =  1: size(data_Error,1)  % chan2 
            [caus_mat_recall(i,j,k),P_recall(i,j,k),P_corr_recall(i,j,k), x_lag_recall(i,j,k), y_lag_recall(i,j,k)] = granger_cause(squeeze(data_recall_st(j,:,i)),squeeze(data_recall_st(k,:,i)), 0.05, x_lag_med(j,k), y_lag_med(j,k), x_max_lag, y_max_lag, 1, down_samp); % transpose make channels as columns
        end
    end
end

caus_mat_final_recall = zeros(total_trials,n_caus_feat);
for i = 1: n_caus_feat
    caus_mat_final_recall(:,i) = P_recall(:,locs_caus(i,1),locs_caus(i,2));
end



%% %%%%%%%%%%%%%%%%%%%%%%%% Testing on Recall %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prediction based on only temporal features
[y_temp] = predictShrinkLDA(model1,feature_mat_red_recall);
performance_error = sum(y_temp ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal features is: %f \n', performance_error);

[v11, rank11] = fisherrank(feature_mat_red_recall, recall_labels);
%figure;
subplot(3,2,1)
plot(v11);
% ylabel('fisher score'); % x-axis label
% xlabel('ranked features'); % y-axis label
title('temporal (Testing)')

% Prediction based on only connectivity features
[y_conn] = predictShrinkLDA(model2,conn_mat_final_recall);
performance_error = sum(y_conn ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for connectivity features is: %f \n', performance_error);

[v22, rank22] = fisherrank(conn_mat_final_recall, recall_labels);
subplot(3,2,2)
plot(v22);
% ylabel('fisher score'); % x-axis label
% xlabel('ranked features'); % y-axis label
title('conn (Testing)')

% prediction based on causality features
[y_caus] = predictShrinkLDA(model3,caus_mat_final_recall);
performance_error = sum(y_caus ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for casuality features is: %f \n', performance_error);

[v33, rank33] = fisherrank(caus_mat_final_recall, recall_labels);
subplot(3,2,3)
plot(v33);
% ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('caus (Testing)')


% prediction based on temporal + causality features 
feature_mat_red_comb = [feature_mat_red_recall caus_mat_final_recall];
feature_mat_red_comb4 = zeros(300,30);
for i = 1:min(30, size(feature_mat_red_comb ,2))
    feature_mat_red_comb4(:,i) = feature_mat_red_comb(:,rank4(i));        % feature matrix with optimum number of features
end
[y_all] = predictShrinkLDA(model4,feature_mat_red_comb4);
performance_error = sum(y_all ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + causality features is: %f \n', performance_error);

[v44, rank44] = fisherrank(feature_mat_red_comb4, recall_labels);
subplot(3,2,4)
plot(v44);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + caus')

% prediction based on temporal + connectivity features
feature_mat_red_comb = [feature_mat_red_recall conn_mat_final_recall];
feature_mat_red_comb5 = zeros(300,30);
for i = 1:30
    feature_mat_red_comb5(:,i) = feature_mat_red_comb(:,rank5(i));        % feature matrix with optimum number of features
end
[y_all] = predictShrinkLDA(model5,feature_mat_red_comb5);
performance_error = sum(y_all ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + connectivity features is: %f \n', performance_error);

[v55, rank55] = fisherrank(feature_mat_red_comb5, recall_labels);
subplot(3,2,5)
plot(v55);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + conn' );

% prediction based on temporal + connectivity + causality features
feature_mat_red_comb = [feature_mat_red_recall conn_mat_final_recall caus_mat_final_recall];
feature_mat_red_comb6 = zeros(300,30);
for i = 1:30
    feature_mat_red_comb6(:,i) = feature_mat_red_comb(:,rank6(i));        % feature matrix with optimum number of features
end
[y_all] = predictShrinkLDA(model6,feature_mat_red_comb6);
performance_error = sum(y_all ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + connectivity + causality features is: %f \n', performance_error);

[v66, rank66] = fisherrank(feature_mat_red_comb6, recall_labels);
subplot(3,2,6)
plot(v66);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp + conn + caus')



save('15_15');



%% Calculation cross-correlation between channel to keep a sanity check on granger causality output

cross_corr_max_lag = 20;
r_cross_corr = zeros(cross_corr_max_lag*2 +1, total_trials, n_chan, n_chan);  % the more the value, the more the causality
lag_cross_corr = zeros(cross_corr_max_lag*2 +1, total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 
r_cross_corr_st = zeros(cross_corr_max_lag*2+1, total_trials, n_chan, n_chan);  % the more the value, the more the causality
lag_cross_corr_st = zeros(cross_corr_max_lag*2 +1, total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 


for i = 1:total_trials
    for j = 1: n_chan
        for k = 1: n_chan
            x = squeeze(data(j,:,i));
            y = squeeze(data(k,:,i));
            x_st = squeeze(data_st(j,:,i));
            y_st = squeeze(data_st(k,:,i));
            
            if down_samp 
                x = downsample(x,4,3);  % phase offset 3
                y = downsample(y,4,3); % phase offset 3
                x_st = downsample(x_st,4,3);  % phase offset 3
                y_st = downsample(y_st,4,3); % phase offset 3
            end
           % a = xcorr(x,y,cross_corr_max_lag);
            [r_cross_corr(:,i,j,k),lag_cross_corr(:,i,j,k)] = xcorr(x,y,cross_corr_max_lag);
            [r_cross_corr_st(:,i,j,k),lag_cross_corr_st(:,i,j,k)] = xcorr(x_st,y_st,cross_corr_max_lag);
        end
    end
end

t = 3.90625 * [1:cross_corr_max_lag];
if down_samp 
    t = t*4;
end

t = cat(2, -fliplr(t), 0, t);

r_cross_corr_med_no_error = squeeze(median(r_cross_corr(:,1:n_no_error,:,:),2));
r_cross_corr_med_error = squeeze(median(r_cross_corr(:,n_no_error+1: total_trials,:,:),2));

r_cross_corr_st_med_no_error = squeeze(median(r_cross_corr_st(:,1:n_no_error,:,:),2));
r_cross_corr_st_med_error = squeeze(median(r_cross_corr_st(:,n_no_error+1: total_trials,:,:),2));


%% Plotting cross-correlation

figure;
subplot(2,2,1);
for i = 1: n_chan
    for j = 1: n_chan
        if i~= j
            plot(t,r_cross_corr_med_no_error(:,i,j))
            hold on
        end
    end
end
title('Cross-corr: No error trials')
xlabel('Lag (ms)');
ylabel('Cross-corr');
ylim([-1000,1000]);

subplot(2,2,2);
for i = 1: n_chan
    for j = 1: n_chan
        if i~= j
            plot(t,r_cross_corr_med_error(:,i,j))
            hold on
        end
    end
end
title('Cross-corr: error trials')
xlabel('Lag (ms)');
ylabel('Cross-corr');
ylim([-1000,1000]);

subplot(2,2,3);
for i = 1: n_chan
    for j = 1: n_chan
        if i~= j
            plot(t,r_cross_corr_st_med_no_error(:,i,j))
            hold on
        end
    end
end
title('Cross-corr: No error stationary trials')
xlabel('Lag (ms)');
ylabel('Cross-corr');
ylim([-60,80]);

subplot(2,2,4);
for i = 1: n_chan
    for j = 1: n_chan
        if i~= j
            plot(t,r_cross_corr_st_med_error(:,i,j))
            hold on
        end
    end
end
title('Cross-corr: Error stationary trials')
xlabel('Lag (ms)');
ylabel('Cross-corr');
ylim([-60,80]);


