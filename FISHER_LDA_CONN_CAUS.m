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
cz_avg_no_error = avg_no_error(20,:); %Cz channel is at the center of the head. index is 20

avg_error = mean(EEG_epo_Error.data,3); % mean across all trials
%avg_error = EEG_epo_Error.data;
cz_avg_error = avg_error(20,:);   %Cz channel is at the center of the head. index is 20

figure;
plot(EEG_epo_noError.times,cz_avg_no_error);
hold on;
plot(EEG_epo_Error.times,cz_avg_error);
legend('No error','error')
title('ERP time course of channel Cz')
xlabel('time locked to key press (ms)') % x-axis label
ylabel('Signal (microvolts') % y-axis label

%% Forming stationary data (not required)
figure;
cz_avg_no_error_diff = diff(cz_avg_no_error,2);
cz_avg_error_diff = diff(cz_avg_error ,2);
plot(EEG_epo_noError.times(3:end),cz_avg_no_error_diff);
hold on;
plot(EEG_epo_Error.times(3:end),cz_avg_error_diff);

min1 =  min(cz_avg_no_error);
min2 =  min(cz_avg_error);

figure;
cz_avg_no_error_diff = diff(log(cz_avg_no_error - min1 + 1),2);
cz_avg_error_diff = diff(log(cz_avg_error - min2 + 1) ,2);
plot(EEG_epo_noError.times(3:end),cz_avg_no_error_diff);
hold on;
plot(EEG_epo_Error.times(3:end),cz_avg_error_diff);



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

avg_class_error = mean(class_error,3);  % doing mean across the k fold 10 values
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


avg_class_error = mean(class_error,2);  % doing mean across the k fold 10 values
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


conn_mat_final_mean = squeeze(mean(conn_mat_final))';  % needed for topoplot

%% Plotting the connection along with their correlation strengths
ds.chanPairs = locs_conn;
ds.connectStrength = conn_mat_final_mean;
figure;
topoplot_connect(ds, EEG.chanlocs);
colorbar;


%% Loading data for granger causality
% clc;
% EEG = pop_loadset('filename','calibration.set','filepath','../data/');
% 
% EEG_epo_noError = pop_epoch(EEG,{'S  4'},[0.0 0.5]); % retrieving the desired epoch with reference as S4 when there was no error
% 
% EEG_epo_Error = pop_epoch(EEG,{'S  5'},[0.0 0.5]); % retrieving the desired epoch with reference as S5 when there was error
% 
% data_noError = EEG_epo_noError.data;
% data_Error = EEG_epo_Error.data;
% data = cat(3,data_noError,data_Error);
% 
% n_chan = size(EEG_epo_noError.data, 1);
% n_no_error =  size(EEG_epo_noError.data, 3);
% n_error = size(EEG_epo_Error.data, 3);
% total_trials = n_no_error + n_error; 

%% Granger causality feature matrix

clc;
x_max_lag = 5;
y_max_lag = 5;
downsample = 0;
caus_mat = zeros(total_trials, n_chan, n_chan);  % the more the value, the more the causality
P = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 
P_corr = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value
x_lag = zeros(total_trials, n_chan, n_chan);
y_lag = zeros(total_trials, n_chan, n_chan);

data_noError_med = squeeze(median(data_noError,3)); 
data_Error_med = squeeze(median(data_Error,3));
caus_mat_init = zeros(total_trials,n_chan, n_chan);  % the more the value, the more the causality
P_init = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value 
P_corr_init = zeros(total_trials, n_chan, n_chan); % will indicate the significance of the correlation value
x_lag_init = zeros(total_trials, n_chan, n_chan);
y_lag_init = zeros(total_trials,n_chan, n_chan);
BIC_R_init = zeros(total_trials,x_max_lag, n_chan, n_chan);
BIC_U_init = zeros(total_trials,y_max_lag, n_chan, n_chan);

%for input matrix of corrcoed, columnsrepresent random variables and the rows represent observations
for i = 1: size(data_Error,1)    % chan1
    for j =  1: size(data_Error,1)  % chan2
        for k = 1:total_trials
            [caus_mat_init(k,i,j),P_init(k,i,j),P_corr_init(k,i,j), x_lag_init(k,i,j), y_lag_init(k,i,j), BIC_R_init(k,:,i,j), BIC_U_init(k,:,i,j),F_den] = granger_cause(squeeze(data(i,:,k)),squeeze(data(j,:,k)), 0.05, 5, 5, x_max_lag, y_max_lag,0,downsample); % transpose make channels as columns
            
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
        end
    end
end

x_lag_med = round(squeeze(median(x_lag_init,1)));
y_lag_med = round(squeeze(median(y_lag_init,1)));

%% plotting BIC 
t_x = 3.90625 * [1:x_max_lag];
t_y = 3.90625 * [1:y_max_lag];

if downsample
    t_x = 4* t_x;
    t_y = 4* t_y;
end


BIC_R_mean_no_err = squeeze(median(BIC_R_init(1:n_no_error,:,:,:),1));
BIC_U_mean_no_err = squeeze(median(BIC_U_init(1:n_no_error,:,:,:),1));
BIC_R_mean_err = squeeze(median(BIC_R_init(n_no_error+1:total_trials,:,:,:),1));
BIC_U_mean_err = squeeze(median(BIC_U_init(n_no_error+1:total_trials,:,:,:),1));

figure;

for i = 1:size(data_Error,1)    % chan1
    for j = 1:size(data_Error,1)  % chan2
        
        plot(t_x, BIC_R_mean_no_err(:,i,j), 'b');
        hold on
        plot(t_y, BIC_U_mean_no_err(:,i,j), 'r');
        hold on

    end
end
xlabel('Lag (ms)');
ylabel('BIC');
title('No Error Trials')

figure;
for i = 1:size(data_Error,1)    % chan1
    for j = 1: size(data_Error,1)  % chan2

        plot(t_x, BIC_R_mean_err(:,i,j), 'k');
        hold on
        plot(t_y, BIC_U_mean_err(:,i,j), 'm');
        hold on
        
    end
end
xlabel('Lag (ms)');
ylabel('BIC');
title(' Error Trials')

figure;
hist(x_lag_med(:));
xlabel('Lag (ms)');
ylabel('Count');
title(' Autoregression')

figure;
hist(y_lag_med(:));
xlabel('Lag (ms)');
ylabel('Count');
title(' Causal Entity')

%% Calculating useful features with the help of fisher score
caus_fish_val = zeros(n_chan,n_chan);  % will contain fisher values

labels = [-1*ones(n_no_error,1); 1*ones(n_error,1)];

for i = 1:n_chan
    for j = 1:n_chan
        x = squeeze(caus_mat_init(:,i,j));
        mu = mean(x);
        thresh = std(x) * sqrt(2*log(2));
        indx = (abs(x-mu)<thresh);  % we want to reject outliers
        x = x(indx);
        y = labels(indx);
        [caus_fish_val(i,j),~] = fisherrank(x,y);
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
            caus_mat_final_train(:,i) = caus_mat_init(train_indx,locs_causs(i,1),locs_causs(i,2));
            caus_mat_final_test(:,i) = caus_mat_init(test_indx,locs_causs(i,1),locs_causs(i,2));
        end

        model = trainShrinkLDA(caus_mat_final_train,labels(train_indx,:),lambda);
        [y] = predictShrinkLDA(model,caus_mat_final_test);
        class_error(j,k) = sum(y' ~= labels(test_indx,:));   % summing up the instants when both are not equal
    end
end


avg_class_error = mean(class_error,2);  % doing mean across the k fold 10 values
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
    caus_mat_final(:,i) = caus_mat_init(:,locs_caus(i,1),locs_caus(i,2));
    caus_mat_mean_no_err(i) = squeeze(median(caus_mat_init(1:n_no_error,locs_caus(i,1),locs_caus(i,2)),1));
    caus_mat_mean_err(i) = squeeze(median(caus_mat_init(n_no_error+1:total_trials,locs_caus(i,1),locs_caus(i,2) ),1));
end

figure;
plot(caus_mat_mean_no_err,'b');
hold on;

plot(caus_mat_mean_err,'r');
xlabel('feature number');
ylabel('F value');

caus_mat_final_mean = squeeze(median(caus_mat_final))';  % needed for topoplot


%% Plotting the connection along with their causal strengths
ds.chanPairs = locs_caus;
ds.connectStrength = caus_mat_final_mean;
topoplot_connect(ds, EEG.chanlocs);
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
xlabel('ranked features'); % y-axis label
title('temp')

% training based on only connectivity features
model2 = trainShrinkLDA(conn_mat_final,labels,lambda);  % final model
[y] = predictShrinkLDA(model2,conn_mat_final);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for connectivity features is: %f \n', performance_error);

[v2, rank2] = fisherrank(conn_mat_final, labels);
subplot(3,2,2)
plot(v2);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('conn')

% training based on only connectivity features
model3 = trainShrinkLDA(caus_mat_final,labels,lambda);  % final model
[y] = predictShrinkLDA(model3,caus_mat_final);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for casuality features is: %f \n', performance_error);

[v3, rank3] = fisherrank(caus_mat_final, labels);
subplot(3,2,3)
plot(v2);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('caus')


% training based on temporal + connectivity features
feature_mat_red_comb = [feature_mat_red caus_mat_final];
[v4, rank4] = fisherrank(feature_mat_red_comb, labels);
feature_mat_red_comb4 = zeros(300,30);
for i = 1:30
    feature_mat_red_comb4(:,i) = feature_mat_red_comb(:,rank4(i));        % feature matrix with optimum number of features
end
model4 = trainShrinkLDA(feature_mat_red_comb4,labels,lambda);  % final model
[y] = predictShrinkLDA(model4,feature_mat_red_comb4);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal + casuality features is: %f \n', performance_error);

subplot(3,2,4)
plot(1:30, v4(1:30));
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
EEG_epo_recall = pop_epoch(EEG_recall,{'S  6'},[0.0 1.0]); % Two spaces between S and 6. S6 in this file is the time when the response was made by the user

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

for i = 1:total_trials_recall % there will be a connectivity matrix for every no error trial
    for j = 1: size(data_Error,1)    % chan1
        for k =  1: size(data_Error,1)  % chan2 
            [caus_mat_recall(i,j,k),P_recall(i,j,k),P_corr_recall(i,j,k), x_lag_recall(i,j,k), y_lag_recall(i,j,k)] = granger_cause(squeeze(data_recall(j,:,i)),squeeze(data_recall(k,:,i)), 0.05, x_lag_med(j,k), y_lag_med(j,k), x_max_lag, y_max_lag, 1, downsample); % transpose make channels as columns
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
figure;
subplot(3,2,1)
plot(v11);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('temp')

% Prediction based on only connectivity features
[y_conn] = predictShrinkLDA(model2,conn_mat_final_recall);
performance_error = sum(y_conn ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for connectivity features is: %f \n', performance_error);

[v22, rank22] = fisherrank(conn_mat_final_recall, recall_labels);
subplot(3,2,2)
plot(v22);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('conn')

% prediction based on causality features
[y_caus] = predictShrinkLDA(model3,caus_mat_final_recall);
performance_error = sum(y_caus ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for casuality features is: %f \n', performance_error);

[v33, rank33] = fisherrank(caus_mat_final_recall, recall_labels);
subplot(3,2,3)
plot(v33);
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label
title('caus')


% prediction based on temporal + causality features 
feature_mat_red_comb = [feature_mat_red_recall caus_mat_final_recall];
feature_mat_red_comb4 = zeros(300,30);
for i = 1:30
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


