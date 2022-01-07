
%% Task 2 - 
%Load Data 
unzip('MerchData.zip'); 
imds = imageDatastore('MerchData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); 
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized'); 
%Display some sample images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
%Load Pretrained Network 
net = resnet18; 
inputSize = net.Layers(1).InputSize; 
analyzeNetwork(net); 
%Extract Image Features 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

whos featuresTrain;

%Extract the class labels from the two dataset 
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
classifier = fitcecoc(featuresTrain,YTrain);

%Classify Test Images using the trained SVM model 
YPred = predict(classifier,featuresTest);

%Display 4 sample test images with thei predicted labels 
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

%accuracy
accuracy = mean(YPred == YTest);

layer = 'res3b_relu';
featuresTrain = activations(net,augimdsTrain,layer);
featuresTest = activations(net,augimdsTest,layer);
whos featuresTrain;

featuresTrain = squeeze(mean(featuresTrain,[1 2]))';
featuresTest = squeeze(mean(featuresTest,[1 2]))';
whos featuresTrain;

%Train an SVM classifier on the shallower features. 
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);

%Calculate accuracy 
accuracy = mean(YPred == YTest)



