clear all
clc 
close all
addpath('Functions')
addpath('Data')

%% Task1: To Load data 
[TrainSet, TrainLabels] = loadMNIST(0); %nxd; n=60000, d=784 (pixels)
[TRAIN_ROWS,TRAIN_COLS]=size(TrainSet);

[TestSet, TestLabels] = loadMNIST(1); %mxd; m=10000, d=784 (pixels)
[TEST_ROWS,TEST_COLS]=size(TestSet);

%%  testing purposes 
PROPORTION=1/100;
REDUCED_TRAIN_ROWS=TRAIN_ROWS*PROPORTION;
REDUCED_TEST_ROWS=TEST_ROWS*PROPORTION;

%  random lines of the datasets
idx = randperm(size(TrainSet,1));
TrainSet = TrainSet(idx(1:REDUCED_TRAIN_ROWS),:);
TrainLabels=TrainLabels(idx(1:REDUCED_TRAIN_ROWS),:);

idx = randperm(size(TestSet,1));
TestSet = TestSet(idx(1:REDUCED_TEST_ROWS),:);
TestLabels=TestLabels(idx(1:REDUCED_TEST_ROWS),:);

%% Task2: create kNN classifier 
k=[11,21,31,41,51,61,71,81,91,101,111];
Accuracy=ones(size(k,2),REDUCED_TEST_ROWS);

%% kNN classifier and plot of results 
f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
title('visual result Example')
for i=1:size(k,2)
    subplot(4,3,i)
    [classification, Error1]=knnClassifier([TrainSet TrainLabels], TestSet, k(i), TestLabels);
    % Example
    plotImages(TestSet(1,:),classification(1),k(i),0);
    Accuracy(i,:)=Accuracy(i,:)-Error1; 
end
saveas(f, ['Results/' 'Image examples.jpg']);

%% Plot of average accuracy
f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
A=ones(1,size(k,2));
for i=1:size(k,2)
    A(i)=mean(Accuracy(i,:))*100;
end
plot(k,A,'R*')
title('Accuracy task2')
ylim([80 100])
xlabel('k')
ylabel('Accuracy %')
saveas(f, ['Results/' 'Accuracy task2.jpg']);

%% Task3: DATA of testing accuracy
n_digits=10; 
k2=[11,21,31,41,51,61,71,81,91,101,111];

%% KNN
Errors=zeros(n_digits,size(k2,2),REDUCED_TEST_ROWS);
Accuracy=ones(n_digits,size(k2,2),REDUCED_TEST_ROWS);
for d=1:n_digits 
    for k=1:size(k2,2) 
        [classifications, Error] = knnClassifier([TrainSet, TrainLabels == d], TestSet, k2(k), TestLabels == d);
        Errors(d,k,:)=Error;
        Accuracy(d,k,:)=Accuracy(d,k,:)-Errors(d,k,:);
    end
end

%% Output plot accuracy
A=ones(n_digits,size(k2,2));
f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
title('Accuracy task3')
xlabel('k')
ylabel('Accuracy %')
for d=1:n_digits 
    for k=1:size(k2,2) 
        A(d,k)=mean(Accuracy(d,k,:))*100;
    end
    subplot(5,2,d)
    plot(k2,A(d,:),'R*')
    ylim([80 100])
    title(['Accuracy for digit ', num2str(d)])
end
saveas(f, ['Results/' 'Accuracy on each digit.jpg']);

