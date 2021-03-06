%% This code uses fisherrank function to rank the features and then LDA classifier to classify the classes
% It compares the old original code developed for temporal features with
% the new one where every channels has separate two time points

%% Loading EEGLAB
clc; 
addpath('D:\TUM\3rd Sem\Neuroinspired System Engineering\eeglab14_1_1b');
addpath('C:\Users\user\Dropbox\Semester3\Neuro Inspired Engineering\Project_Pranshul\functions');
eeglab;

%% Loading of data and creating epochs
EEG = pop_loadset('filename','calibration.set','filepath','../data/');

% S5 are errors and S4 are no_errors. One of these are errors. This information is
% in the third row of every trial (EEG.event)

eegplot(EEG.data,'eloc_file',EEG.chanlocs);   % plotting eeg data

EEG_epo_noError = pop_epoch(EEG,{'S  4'},[0.0 1]); % retrieving the desired epoch with reference as S4 when there was no error

EEG_epo_Error = pop_epoch(EEG,{'S  5'},[0.0 1]); % retrieving the desired epoch with reference as S5 when there was error

%% Time course Visualisation
avg_no_error = mean(EEG_epo_noError.data,3);
cz_avg_no_error = avg_no_error(20,:); %Cz channel is at the center of the head. index is 20

avg_error = mean(EEG_epo_Error.data,3);
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
time_pt_1 = find(abs(EEG_epo_Error.times - 210.9375)<1.0);   % Finding the proximity of the event
time_pt_2 = find(abs(EEG_epo_Error.times - 285.2)<1.0);

avg_tp1_no_error = avg_no_error(:,time_pt_1);
avg_tp2_no_error = avg_no_error(:,time_pt_2);

avg_tp1_error = avg_error(:,time_pt_1);
avg_tp2_error = avg_error(:,time_pt_2);

figure;   
ax1 = subplot(2,2,1);
title(ax1,'no Error N2');
topoplot(avg_tp1_no_error,EEG.chanlocs,'maplimits',[-4 4]);

ax2 = subplot(2,2,2);
title(ax2,'no Error P3');
topoplot(avg_tp2_no_error,EEG.chanlocs,'maplimits',[-4 4]);

ax3 = subplot(2,2,3);
title(ax3,'Error N2');
topoplot(avg_tp1_error,EEG.chanlocs,'maplimits',[-4 4]);

ax4 = subplot(2,2,4);
title(ax4,'Error P3');
topoplot(avg_tp2_error,EEG.chanlocs,'maplimits',[-4 4]);

%% Creation of feature vector
no_error_tp1 = EEG_epo_noError.data(:,time_pt_1,:);
no_error_tp1 = squeeze(no_error_tp1)';
no_error_tp2 = EEG_epo_noError.data(:,time_pt_2,:);
no_error_tp2 = squeeze(no_error_tp2)';
no_error = [no_error_tp1 no_error_tp2];

error_tp1 = EEG_epo_Error.data(:,time_pt_1,:);
error_tp1 = squeeze(error_tp1)';
error_tp2 = EEG_epo_Error.data(:,time_pt_2,:);
error_tp2 = squeeze(error_tp2)';
error = [error_tp1 error_tp2];

feature_mat = [no_error; error];

%standardization
mean_feature_mat = mean(feature_mat);
mean_feature_mat = repmat(mean_feature_mat,300,1);
std_feature_mat = std(feature_mat);
std_feature_mat = repmat(std_feature_mat, 300,1);
feature_mat = (feature_mat - mean_feature_mat)./std_feature_mat;

labels = [-1*ones(190,1); 1*ones(110,1)];

figure;  % just to see if the data is normally distributed or not. Otherwise we could do log normalisation
histogram(feature_mat(:,47));  % It appears that the data is approximately normally distribued. So log normal transformation is not needed. Moreover, it can't be used here because we can't feed negative values to log function
title('Approximatly gaussian data (Log normal transform not required)')


%% plotting rank wise
addpath('C:\Users\user\Dropbox\Semester 3\Neuro Inspired Engineering\NISE_BCI_tutorial\functions');
[v, rank] = fisherrank(feature_mat, labels);
figure;
plot(1:54, v);
xlabel('fisher score'); % x-axis label
ylabel('ranked features'); % y-axis label

%scatter plot of best and second best features
% figure;
% scatter(no_error(:,rank(1)), no_error(:,rank(2)), 'b');
% hold on;
% scatter(error(:,rank(1)), error(:,rank(2)), 'r');
% legend('No error,-1','error,1');
% xlabel('1st best feature'); % x-axis label
% ylabel('2nd best feature'); % y-axis label


%% Creating a classifier

% choosing best lambda and optimum number of features using Kfold crossvalidation
class_error = zeros(100,30,10);
for i =1:100   % I am multiplying in the for loop for 0.01. So it is 0.01 change per loop   
    for j = 1:30 % Run upto 30 to avoid overfitting
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

feature_mat_red = zeros(300,n);
for i = 1:n
    feature_mat_red(:,i) = feature_mat(:,rank(i));        % feature matrix with optimum number of features
end

figure;
[X,Y] = meshgrid(1:30,lambda_set);
surf(X,Y,avg_class_error);
xlabel('Number of features'); % x-axis label
ylabel('Lambda value'); % y-axis label
zlabel('Average error');

model = trainShrinkLDA(feature_mat_red,labels,lambda);  % final model
[y] = predictShrinkLDA(model,feature_mat_red);
performance_error = sum(y' ~= labels)/300;  % This is the misclassification rate
fprintf('The misclassification rate is: %f % \n', performance_error);

%% Model Cross-session validation with the recall set
recall_load = load('recall_ground_truth.mat');
recall_labels = recall_load.labels;

EEG_recall = pop_loadset('filename','recall.set','filepath','../data/');
EEG_epo_recall = pop_epoch(EEG_recall,{'S  6'},[0.0 0.5]); % Two spaces between S and 6. S6 in this file is the time when the response was made by the user

time_pt_1_recall = find(abs(EEG_epo_recall.times - 210.9375)<1.0);
time_pt_2_recall = find(abs(EEG_epo_recall.times - 285.2)<1.0);

% forming feature matrix
EEG_recall_tp1 = EEG_epo_recall.data(:,time_pt_1_recall,:);
EEG_recall_tp1 = squeeze(EEG_recall_tp1)';
EEG_recall_tp2 = EEG_epo_recall.data(:,time_pt_2_recall,:);
EEG_recall_tp2 = squeeze(EEG_recall_tp2)';
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

% Validation on recall set
[y_temp] = predictShrinkLDA(model,feature_mat_red_recall);
performance_error = sum(y_temp ~= recall_labels )/300;  % This is the misclassification rate
fprintf('The misclassification rate for temporal features is: %f \n', performance_error);




%% New method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% figure;
% plot(EEG_epo_noError.times,cz_avg_no_error);
% hold on;
% plot(EEG_epo_Error.times,cz_avg_error);
% legend('No error','error')
% title('ERP time course of channel Cz')
% xlabel('time locked to key press (ms)') % x-axis label
% ylabel('Signal (microvolts') % y-axis label


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


% figure;   % green color would represent 0. Towards red = positive; Towards blue = negative
% ax1 = subplot(2,2,1);
% title(ax1,'no Error time point 1');
% topoplot(avg_tp1_no_error,EEG.chanlocs,'maplimits',[-4 4], 'conv', 'on');
% 
% ax2 = subplot(2,2,2);
% title(ax2,'no Error time point 2');
% topoplot(avg_tp2_no_error,EEG.chanlocs,'maplimits',[-4 4],  'conv', 'on');
% 
% ax3 = subplot(2,2,3);
% title(ax3,'Error time point 1');
% topoplot(avg_tp1_error,EEG.chanlocs, 'maplimits',[-4 4], 'conv', 'on');
% 
% ax4 = subplot(2,2,4);
% title(ax4,'Error time point 2');
% topoplot(avg_tp2_error,EEG.chanlocs,'maplimits',[-4 4], 'conv', 'on');



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

% figure;  % just to see if the data is normally distributed or not. Otherwise we could do log normalisation
% histogram(feature_mat(:,47));  % It appears that the data is approximately normally distribued. So log normal transformation is not needed. Moreover, it can't be used here because we can't feed negative values to log function
% title('Approximatly gaussian data (Log normal transform not required)')



%% plotting rank wise
hold on
[v, rank] = fisherrank(feature_mat, labels);

plot(1:54, v, 'r');
ylabel('fisher score'); % x-axis label
xlabel('ranked features'); % y-axis label

