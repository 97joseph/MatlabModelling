% Description: test a feature for the investigation and output the
% accuracy, there's some performance variance in the custom knn functions
%
% Inputs: testCat, testData, trainCat, trainData,y (see run_investigation())
function test_feature(testCat, testData, trainCat, trainData,y)
    % train the model
    if y == 0
        model = fitcknn(trainData,trainCat,"NumNeighbors",3);
    else
        model = knn_fit(trainData,trainCat);
    end
    % use the model to predict what label each row of data will have
    if y == 0
        predictions = predict(model, testData);
    else
        predictions = knn_predict(model, testData);
    end
    % create a confusion matrix to aid in performance evaluation
    [results,~] = confusionmat(testCat, predictions);
    % create a percentage accuracy to allow easy comparison
    a = 100 * (sum(diag(results)) / sum(results(:)));
    % declare the accuracy
    disp(['The accuracy is: ',num2str(a),'%'])
end