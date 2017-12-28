%% This code uses fisherrank function to rank the features and then LDA classifier to classify the classes
% I am also using connectivity feature to reach a better classification

%% Loading EEGLAB
clc;
addpath('D:\TUM\3rd Sem\Neuroinspired System Engineering\eeglab14_1_1b');
eeglab;

%% Loading of data and creating epochs
clc;
EEG = pop_loadset('filename','calibration.set','filepath','C:/Users/user/Dropbox/Semester3/Neuro Inspired Engineering/Project/data/');

% S5 are errors and S4 are no_errors. This information is
% in the third row of every trial (EEG.event)

eegplot(EEG.data,'eloc_file',EEG.chanlocs);   % plotting eeg data

EEG_epo_noError = pop_epoch(EEG,{'S  4'},[0.0 0.5]); % retrieving the desired epoch with reference as S4 when there was no error

EEG_epo_Error = pop_epoch(EEG,{'S  5'},[0.0 0.5]); % retrieving the desired epoch with reference as S5 when there was error

n_chan = size(EEG_epo_noError.data, 1);
n_no_error =  size(EEG_epo_noError.data, 3);
n_error = size(EEG_epo_Error.data, 3);
total_trials = n_no_error + n_error; 


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

%% Connectivity features (correlation measure [-1,1])
conn_mat = zeros(total_trials, n_chan,n_chan);
P = zeros(total_trials, n_chan,n_chan); % will indicate the significance of the correlation value 

data_noError = EEG_epo_noError.data;
data_Error = EEG_epo_Error.data;

%for input matrix of corrcoed, columnsrepresent random variables and the rows represent observations
for i = 1:n_no_error % there will be a connectivity matrix for every no error trial
    [conn_mat(i,:,:),P(i,:,:)] = corrcoef(squeeze(data_noError(:,:,i))'); % transpose make channels as columns
end

for i = 1:n_error % there will be a connectivity matrix for every error trial
    [conn_mat(i+n_no_error,:,:),P(i+n_no_error,:,:)] = corrcoef(squeeze(data_Error(:,:,i))'); % 
end

conn_fish_val = zeros(n_chan,n_chan);

labels = [-1*ones(n_no_error,1); 1*ones(n_error,1)];

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

n_conn_feat = 10;

[max, locs] = maxNvalues(conn_fish_val,n_conn_feat);
conn_mat_final = zeros(total_trials,n_conn_feat);
for i = 1: n_conn_feat
    conn_mat_final(:,i) = conn_mat(:,locs(i,1),locs(i,2));
end


%% Creation of feature vector

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

data = [no_error; error];  %  dimension = total_trials x (2*channels)

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
addpath('C:\Users\user\Dropbox\Semester3\Neuro Inspired Engineering\Project\functions');
[v, rank] = fisherrank(feature_mat, labels);
figure;
plot(1:54, v);
xlabel('fisher score'); % x-axis label
ylabel('ranked features'); % y-axis label

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

% choosing best lambda and optimum number of features using Kfold crossvalidation
class_error = zeros(100,round(n_feat/2),10);
for i =1:100   % I am multiplying in the for loop for 0.01. So it is 0.01 change per loop   
    for j = 1:round(n_feat/2) % across all the features
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
[X,Y] = meshgrid(1:round(n_feat/2),lambda_set);
surf(X,Y,avg_class_error);
xlabel('Number of features'); % x-axis label
ylabel('Lambda value'); % y-axis label
zlabel('Average error');


%% Creating final feature matrix which contains connectivity features as well

feature_mat_red = zeros(300,n);
for i = 1:n
    feature_mat_red(:,i) = feature_mat(:,rank(i));        % feature matrix with optimum number of features
end

feature_mat_red = [feature_mat_red conn_mat_final];

model = trainShrinkLDA(feature_mat_red,labels,lambda);  % final model
[y] = predictShrinkLDA(model,feature_mat_red);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate is: %f % \n', performance_error);

