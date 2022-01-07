function plotImages(TestSet,classification,k,d)
% This function plots images of the dataset
% inputs: TestSet: test set
%         classification: predicted value of the observation
%         k: parameter of the kNN classifier
%         d: digit to analyze
    plotting=reshape(TestSet,[28,28]);
    imshow(plotting)
    if classification==10
        classification=0;
    end
    if d~=0
        title(['Prediction : ', num2str(classification), ' with k=', num2str(k) 'tested on digit ', num2str(d)])
    else
        title(['Prediction : ', num2str(classification), ' with k=', num2str(k)])
    end

end

