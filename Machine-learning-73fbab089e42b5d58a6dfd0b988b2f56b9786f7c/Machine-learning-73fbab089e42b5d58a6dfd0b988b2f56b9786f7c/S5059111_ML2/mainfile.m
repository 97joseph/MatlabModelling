clear all
clc 
close all
addpath('functions')
addpath('data')

% Number of models we use IN THIS IS
N_MODELS=3;

% Task 1: load the acquired data
CarsDataset=readtable('mtcarsdata-4features.csv');
TurkishDataset=load('turkish-se-SP500vsMSCI.csv');

%% Task 2
%% 1 ) One-dimensional problem without intercept on the Turkish dataset
% x:data
% t:result
x_Turk=TurkishDataset(:,1);
t_Turk=TurkishDataset(:,2);
% Call  linear regression function
w1=oneDimLinReg(x_Turk,t_Turk);
y1=x_Turk*w1;

% Plot the data
figure; 
plot(x_Turk,t_Turk,'bx')
hold on;
plot(x_Turk,y1,'black')
title('One Dimensional problem without intercept on the dataset(Turkish)')
xlabel('x (data)'); 
ylabel('t (target)'); 

%% 2 ) Compare graphically the solution obtained on different random 
% Plot model
figure; 
hold on;
plot(x_Turk,t_Turk,'bx')
plot(x_Turk,y1,'CYAN')
title('One Dimensional problem of different subsets (10% of dataset)')
xlabel('x (data)'); 
ylabel('t (target)'); 

for i=1:7
    idx = randperm(length(x_Turk));
    subSet = TurkishDataset(idx(1:round(length(x_Turk)/10)),:);
    x2=subSet(:,1);
    t2=subSet(:,2);
    w=oneDimLinReg(x2,t2);
    y2=w*x2;
    
    % Plot the data
    hold on;
    plot(x2,y2,'yellow')
    legend('Data','Model  dataset', ' subsets model');
end


%% 3 ) One-dimensional problem with intercept on the Motor Trends car data, 
x_Car=CarsDataset{:,5};
t_Car=CarsDataset{:,2};
% Call 1D linear regression function with intercept
[w1,w0]=oneDimLinReg_intercept(x_Car,t_Car);
y3=w1*x_Car+w0;

% Plot
figure; 
plot(x_Car,t_Car,'bx')
hold on;
plot(x_Car,y3,'red')
title('1D problem with intercept on the  data, using MPG and WEIGHT')
xlabel('weight (data)'); 
ylabel('mpg (target)'); 

%% 4 ) Multi-dimensional problem on the complete MTcars data, using all 

X_Car=CarsDataset{:,3:5};     
t_Car=CarsDataset{:,2};

% Normalize
X_Car_norm=zeros(size(X_Car));
for i=1:size(X_Car,2) 
    for j=1:size(X_Car,1) 
        X_Car_norm(j,i)=(X_Car(j,i)-mean(X_Car(:,i)))/std(X_Car(:,i));
    end
end

t_Car_norm=zeros(size(t_Car));
for j=1:size(t_Car,1)
    t_Car_norm(j)=(t_Car(j)-mean(t_Car(:)))/std(t_Car(:));
end

% To Call multi-DIMENSIONAL linear regression function
W=multiDimLinReg(X_Car_norm,t_Car_norm);
y4_norm=X_Car_norm*W;
y4=zeros(size(y4_norm));
for j=1:size(y4_norm,1)
    y4(j)=(y4_norm(j)*std(t_Car(:)))+mean(t_Car(:));
end

MultidimResults = table(t_Car, y4);
MultidimResults.Properties.VariableNames = {'Real----Target t' 'Predicted------Target y'};

figure
uitable('Data',MultidimResults{:,:},'ColumnName', MultidimResults.Properties.VariableNames,...
    'Units', 'Normalized', 'Position',[0, 0, 1, 1]);



%% Task 3: re-run 1,3,4 on a training set 
LOOPS=10;
obj.training=zeros(LOOPS,N_MODELS);
obj.test=zeros(LOOPS,N_MODELS);

for i=1:LOOPS
    % Divide the sets 
    idx = randperm(size(CarsDataset,1));
    VAR=round(size(CarsDataset,1)/20);
    CarTrainingSet = CarsDataset{idx(1:VAR),2:end}; 
    CarTestSet = CarsDataset{idx(VAR:end),2:end}; 

    idx2 = randperm(size(TurkishDataset,1));
    VAR2=round(size(TurkishDataset,1)/20);
    TurkTrainingSet = TurkishDataset(idx(1:VAR2),:);
    TurkTestSet = TurkishDataset(idx(VAR2:end),:);
    
    % Print dimensions of training sets
    if i==1
        fprintf('\nTraining set for (CARS)dataset is of size %d', VAR)
        fprintf('\nTraining set for (TURKISH) dataset is of size %d', VAR2)
    end
    
    x_1=TurkTrainingSet(:,1);
    t_1=TurkTrainingSet(:,2);
    w1_1=oneDimLinReg(x_1,t_1);
    obj.training(i,1) = meanSquareError(x_1,t_1,w1_1,0,1);

    %% 3
    x_2=CarTrainingSet(:,4);
    t_2=CarTrainingSet(:,1);
    [w1_2,w0]=oneDimLinReg_intercept(x_2,t_2);
    obj.training(i,2) = meanSquareError(x_2,t_2,w1_2,w0,1);
    
    %% 4
    x_3=CarTrainingSet(:,2:4);
    x_3_norm=zeros(size(x_3));
    for k=1:size(x_3,2) 
        for j=1:size(x_3,1) 
            x_3_norm(j,k)=(x_3(j,k)-mean(x_3(:,k)))/std(x_3(:,k));
        end
    end
    
    t_3=CarTrainingSet(:,1); 
    t_3_norm=zeros(size(t_3));
    for j=1:size(t_3,1)
        t_3_norm(j)=(t_3(j)-mean(t_3(:)))/std(t_3(:));
    end

    w1_3=multiDimLinReg(x_3,t_3);
    obj.training(i,3) = meanSquareError(x_3_norm,t_3_norm,w1_3,0,2);
    
    %% Test results and compute J from training sets
    %% 1
    x=TurkTestSet(:,1);
    t=TurkTestSet(:,2);
    obj.test(i,1) = meanSquareError(x,t,w1_1,0,1);
     
    %% 3
    x=CarTestSet(:,4);
    t=CarTestSet(:,1);
    obj.test(i,2) = meanSquareError(x,t,w1_2,w0,1);
    
    %% 4
    x=CarTestSet(:,2:4);
    x_norm=zeros(size(x));
    for k=1:size(x,2) 
        for j=1:size(x,1) 
            x_norm(j,k)=(x(j,k)-mean(x(:,k)))/std(x(:,k));
        end
    end
    
    t=CarTestSet(:,1); 
    t_norm=zeros(size(t));
    for j=1:size(t,1)
        t_norm(j)=(t(j)-mean(t(:)))/std(t(:));
    end
    obj.test(i,3) = meanSquareError(x_norm,t_norm,w1_3,0,2);
    
end 

average_obj.training=zeros(N_MODELS,1);
average_obj.test=zeros(N_MODELS,1);

for i=1:N_MODELS
    average_obj.training(i)=mean(obj.training(i));
    average_obj.test(i)=mean(obj.test(i));
end

ObjectivesResults = table(average_obj.training, average_obj.test);
ObjectivesResults.Properties.VariableNames = {'Train' 'Test'};
ObjectivesResults.Properties.RowNames = {'one dimension' 'one dimension offset' 'multi-DIMENSIONAL'};

figure
uitable('Data',ObjectivesResults{:,:},'ColumnName',ObjectivesResults.Properties.VariableNames,...
    'RowName',ObjectivesResults.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

