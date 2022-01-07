clear all; 
close all; 
clc; 
addpath('MNISTdataset')


[Train1, Target1] = loadMNIST(0,3); 
[Train2, Target2] = loadMNIST(0,8); 

%Extract only some classes 
Data1 = [Train1, Target1]; 
Data2 = [Train2, Target2]; 
[n,m] = size(Data1); 
[q,r] = size(Data2); 

random_indexes1 = randperm(n); 
random_indexes2 = randperm(q); 

Subset1 = Data1(random_indexes1(1:500), :); 
Subset2 = Data2(random_indexes2(1:500), :); 

%Create training set 
Training = [Subset1(:,1:end-1)', Subset2(:,1:end-1)']; 
Target = [Subset1(:,end)', Subset2(:,end)']; 


HiddenSize = 2; 
myAutoencoder = trainAutoencoder(Training, HiddenSize); 

%Encode the different classes using the encoder obtained
myEncodedData = encode(myAutoencoder, Training);


plotcl(myEncodedData', Target');

legend(['Class ', num2str(labelData1(1))], ['Class ', num2str(labelData2(1))]);



