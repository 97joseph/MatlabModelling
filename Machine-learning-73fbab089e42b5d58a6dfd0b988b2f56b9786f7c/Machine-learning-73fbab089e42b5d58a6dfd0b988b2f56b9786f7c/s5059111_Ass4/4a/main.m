clear all
clc 
close all
addpath('Functions')
addpath('Data')

%% TASK 2 

%% Load MNIST dataset
[MNISTTrainSet, MNISTTrainLabels] = loadMNIST(0); 
[TRAIN_ROWS,TRAIN_COLS]=size(MNISTTrainSet);

% Choose 2 digits to recognize
digitToRecognize1=1;
digitToRecognize2=2;

% Prepare the dataset to have the two digits only
idx1=find(MNISTTrainLabels==digitToRecognize1);
idx2=find(MNISTTrainLabels==digitToRecognize2);
MNISTTrainSetpt1=MNISTTrainSet(idx1,:);
MNISTTrainSetpt2=MNISTTrainSet(idx2,:);
MNISTTrainSet=[MNISTTrainSetpt1;MNISTTrainSetpt2];
MNISTTrainLabelspt1=MNISTTrainLabels(idx1,:);
MNISTTrainLabelspt2=MNISTTrainLabels(idx2,:);
MNISTTrainLabels=[MNISTTrainLabelspt1;MNISTTrainLabelspt2];

% Take a portion of data only
PROPORTION=1/10;
REDUCED_TRAIN_ROWS=size(MNISTTrainSet,1)*PROPORTION;
idx = randperm(size(MNISTTrainSet,1));
MNISTTrainSet = MNISTTrainSet(idx(1:REDUCED_TRAIN_ROWS),:);
MNISTTrainLabels=MNISTTrainLabels(idx(1:REDUCED_TRAIN_ROWS),:);

% Set labels to -1 or +1
for i=1:length(MNISTTrainLabels)
    if MNISTTrainLabels(i)==digitToRecognize1 
        MNISTTrainLabels(i)=-1;
    elseif MNISTTrainLabels(i)==digitToRecognize2
        MNISTTrainLabels(i)=1;
    end
end

%% Load iris dataset
irisDataSet = load("iris-2class.txt");

%% Load XOR dataset
XORDataSet=load('XOR.txt');

%% XOR: modify to make compliant with code
for i=1:size(XORDataSet,1)
    if XORDataSet(i,3)==0
        XORDataSet(i,3)=-1;
    end
end

%% Data
eta=0.1;

%% TASK 2 

%% IRIS
k=[2,10,size(irisDataSet,1)];
for i=1:3
    %% Split dataset, apply perceptron and adaline algorithms
    [x,num_sets]=splitDataSet(irisDataSet,k(i));
    [IRISpercConfMat,~]=perceptron(x,eta,num_sets);
    [IRISadaConfMat,~]=adaline(x,eta,num_sets);
    %% Create and save tables for confusion matrices
    images('Perceptron','iris',IRISpercConfMat,k(i))
    images('Adaline','iris',IRISadaConfMat,k(i))
end

%% MNIST
k2=[2,10,size(MNISTTrainSet,1)];
for i=1:3
    %% Split dataset, apply perceptron and adaline algorithms
    [x,num_sets]=splitDataSet([MNISTTrainSet MNISTTrainLabels],k2(i));
    [MNISTpercConfMat,~]=perceptron(x,eta,num_sets);
    [MNISTadaConfMat,~]=adaline(x,eta,num_sets);
    %% Create and save tables for confusion matrices
    images('Perceptron','MNIST',MNISTpercConfMat,k(i))
    images('Adaline','MNIST',MNISTadaConfMat,k(i))
end

%% XOR
k=2;
%% Split dataset, apply perceptron and adaline algorithms
[x,num_sets]=splitDataSet(XORDataSet,k);
[XORpercConfMat,~]=perceptron(x,eta,num_sets);
[XORadaConfMat,~]=adaline(x,eta,num_sets);

%% Create and save tables for confusion matrices
images('Perceptron','XOR',XORpercConfMat,k)
images('Adaline','XOR',XORadaConfMat,k)
