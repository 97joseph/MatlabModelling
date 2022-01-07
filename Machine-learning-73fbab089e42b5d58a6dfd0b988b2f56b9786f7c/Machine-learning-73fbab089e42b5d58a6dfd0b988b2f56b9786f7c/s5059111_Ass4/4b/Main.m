close all; 
clear all; 
clc; 

%% Task 0 : Neural networks in Matlab 

load chemical_dataset; 

hiddenLayerSize = 10;


FittingProblemwithNeuralNetwork(chemicalInputs,chemicalTargets, hiddenLayerSize);

%% Task 1 : Feedforward multi-layer networks (multi-layer perceptrons)
load wine_dataset; 


load glass_dataset;

PatternRecognitionProblemwithNeuralNetwork(wineInputs, wineTargets, hiddenLayerSize); 
PatternRecognitionProblemwithNeuralNetwork(glassInputs, glassTargets, hiddenLayerSize); 
