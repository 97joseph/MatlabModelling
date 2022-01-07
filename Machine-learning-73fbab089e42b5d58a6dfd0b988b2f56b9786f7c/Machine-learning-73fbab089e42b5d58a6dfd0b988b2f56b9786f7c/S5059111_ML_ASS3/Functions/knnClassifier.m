function [predictedtestLabels, Error]=knnClassifier(TrainSet, TestSet, k, TestLabels)
% This function applies a kNN classifier
% inputs: TrainSet: training set + training labels
%         TestSet: test set without labels
%         k: parameter of the kNN classifier
%         TestLabels: test labels
% outputs: Prediction: predicted class for each observation
%          Error: total error of the classifier

    [TRAIN_ROWS,TRAIN_COLS]=size(TrainSet);
    [TEST_ROWS,TEST_COLS]=size(TestSet);
    
    %% Check that the number of arguments received (nargin) equals at least the number of mandatory arguments
    if nargin <3
        disp('Error: not enough inputs');
        return
    end
    
    %% Check that the number of columns of the second matrix equals the number of columns of the first matrix
    % Remember that train has 1 col more here
    if TEST_COLS+1 ~= TRAIN_COLS
        disp('Erros: columns of different sizes');
        return
    end
    
    %% Check that k>0 and k<=cardinality of the training set (number of rows, above referred to as n)
    if k < 0 || k > TRAIN_ROWS
        disp('Error: k<0 or k>cardinality of training set');
        return
    end
    
    %% Compute the k nearest neighbours
    % pdist2: Computes the distance using the metric 'euclidean' and returns:
    % - matrix D of the K smallest pairwise distances to observations in X 
    % for each observation in Y in ascending order. D: kxTEST_COLS
    % - matrix TrainIdx of the indices of the observations in X
    % corresponding to the distances in D. I: kxTEST_COLS
    [~ , TrainIdx] = pdist2(TrainSet(:,1:TEST_COLS), TestSet, 'euclidean', 'Smallest', k); % Euclidean distance
    
    %% Get the labels of the selected K entries
    pointLabels = zeros(size(TrainIdx)); % kxTEST_COLS
    for i=1:size((TrainIdx),1)
        for j=1:size((TrainIdx),2)
            pointLabels(i,j) = TrainSet(TrainIdx(i,j),end); %end: column of labels
        end
    end
    
    %% It's a classification problem: return the mode of the K labels
    predictedtestLabels=zeros(size(TrainIdx,2), 1);
    for i=1:size((TrainIdx),2)
        predictedtestLabels(i,1)=mode(pointLabels(:,i));
    end

    %% Compute the error rate
    if nargin > 3
        Error = (sum(predictedtestLabels ~= TestLabels))/TEST_ROWS;
    end
    
end


