% Description: run an investigation on a feature
%
% Inputs: x: the feature number, matches up to the header, y: variable to
% determine which functions to use: 0 - bundled, 1 - custom
function run_investigation(x,y)
    % re-seed the random number for reproducible results
    rng(0);
    % read the file's data and store it in memory
    if x == 1
        data = readcell('MerchData.csv');
    else
        data = imageDatastore('MerchData', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    end
    % create a split of test:train randomised data per classification
    % category, typically 2:3
    switch x
        case 1
            data(1,:) = []; % remove the headers
            data = data(randperm(size(data,1)),:);
            testSize = round(0.4 * size(data,1));
            testData = data(1:testSize,:);
            trainData = data(testSize+1:end,:);
        case 6
            % use a 7:3 split, as recommended by mathworks
            [imdsTrain, imdsTest] = splitEachLabel(data, 0.3, 'randomize');
        otherwise
            [imdsTrain, imdsTest] = splitEachLabel(data, 0.6, 'randomize');
    end
    % gather the labels
    if x == 1
        trainCat = categorical(trainData(:,2));
        testCat = categorical(testData(:,2));
    else
        trainCat = imdsTrain.Labels;
        testCat = imdsTest.Labels;
    end
    % test resnet50 on a sample image
    if x == 6
        % set the requested neural network
        net = resnet50;
        % retrieve the first test image
        im = imdsTest.read();
        % make sure later uses still use all images
        imdsTest.reset;
        % store the required size 
        imSize = net.Layers(1).InputSize; 
        % get the classification result using an image resized to meet
        % the size requirement
        label = classify(net,imresize(im,imSize(1:2)));
        % report classification result
        disp(['Resnet50 classifies this image as: ',label])
    end
    % prepare the data
    switch x
        case 1
            % strip the categories
            trainData(:,2) = [];
            testData(:,2) = [];
            % convert to matrices
            trainData = cell2mat(trainData(:,:));
            testData = cell2mat(testData(:,:));
        case 6
            % resize the images to mkae them compatible 
            imdsTrain = augmentedImageDatastore(imSize, imdsTrain);
            imdsTest = augmentedImageDatastore(imSize, imdsTest);
            % set feature layer
            featureLayer = 'fc1000';
            % collect high level features
            trainData = activations(net, imdsTrain, featureLayer, ...
            'MiniBatchSize', 32, 'OutputAs', 'rows');
            testData = activations(net, imdsTest, featureLayer, ...
            'MiniBatchSize', 32, 'OutputAs', 'rows');
            disp('Investigation using high level features in progress...')
        otherwise
            % initialise arrays
            trainData = [];
            testData = [];
            % one feature needs words, the rest not so much
            if x == 5
                words = bagOfFeatures(imdsTrain);
            else
                words = 0;
            end
            % populate arrays, feature is extracted from greyscale image
            while hasdata(imdsTrain)
                trainData(end+1,:) = get_feature(x, y, imdsTrain.read(), words);
            end
            while hasdata(imdsTest)
                testData(end+1,:) = get_feature(x, y, imdsTest.read(), words);
            end
    end
    % compute an accuracy rating for a feature
    test_feature(testCat, testData, trainCat, trainData,y)
    % this investigation tests multiple kinds of feature
    if x == 6
        % set feature layer
        featureLayer = 'input_1';
        % collect high level features
        trainData = activations(net, imdsTrain, featureLayer, ...
        'MiniBatchSize', 32, 'OutputAs', 'rows');
        testData = activations(net, imdsTest, featureLayer, ...
        'MiniBatchSize', 32, 'OutputAs', 'rows');
        disp('Investigation using low level features in progress...')
        % compute the accuracy
        test_feature(testCat, testData, trainCat, trainData,y)
        % create a graph of the network
        lgraph = layerGraph(net);
        % find layers we need in order to retrain
        [learnableLayer,classLayer] = findLayersToReplace(lgraph);
        % replace with the new learnable layer containing the correct
        % number of classes
        lgraph = replaceLayer(...
            lgraph,...
            learnableLayer.Name,...
            fullyConnectedLayer(...
                numel(categories(trainCat)), ...
                'Name','new_fc', ...
                'WeightLearnRateFactor',10, ...
                'BiasLearnRateFactor',10)...
        );
        % swap the class layer with an empty one, this will be populated
        % during training
        lgraph = replaceLayer(lgraph,classLayer.Name,classificationLayer('Name','new_classoutput'));
        % define training options
        miniBatchSize = 10;
        options = trainingOptions('sgdm', ...
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs',6, ...
            'InitialLearnRate',3e-4, ...
            'Shuffle','every-epoch', ...
            'ValidationData',imdsTest, ...
            'ValidationFrequency',floor(numel(imdsTrain.Files)/miniBatchSize), ...
            'Verbose',false, ...
            'Plots','training-progress');
        % retrain the network
        trainNetwork(imdsTrain,lgraph,options);
    end
end