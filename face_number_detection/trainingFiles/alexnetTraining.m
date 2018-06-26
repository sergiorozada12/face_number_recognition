% This code follows the Matlab example stated in the report, the same
% script is used for training face recognizer, number recognizer and
% expresion recognizer: just change input dataset

%% Load the data

% Load the data
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/n1';
images = imageDatastore(root,... 
    'IncludeSubfolders',true,... 
    'LabelSource','foldernames'); 

% Split into train, validation and test
[training,testImages] = splitEachLabel(images,0.8,'randomized'); 
[trainingImages,validationImages] = splitEachLabel(training,0.8,'randomized'); 

%% Instantiante the AlexNet

% Instantiate the network
net = alexnet; 

% Just train the last three layers
layersTransfer = net.Layers(1:end-3); 
numClasses = numel(categories(trainingImages.Labels));

% Define the MLP for the last three layers
layers = [ layersTransfer 
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];
inputSize = net.Layers(1).InputSize;

%% Data augmentation

% Augment the dataset to get different invariances and avoid overfitting
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-30,30], ...
    'RandXScale',[0.9,1.1], ...
    'RandYScale',[0.9,1.1], ...
    'RandXReflection',false, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% TRAIN SET
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);

% VALIDATION SET
augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationImages);

% TEST SET
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

%% Train the net

% Set training parameters
miniBatchSize = 10; 
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize); 
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'ValidationFrequency',100, ...
    'Verbose',false, ...
    'Plots','training-progress'); 

% Train the network
netTransfer = trainNetwork(augimdsTrain,layers,options); 

%% Evaluation

% Predict on test set
predictedLabels = classify(netTransfer,augimdsTest);
testLabels = testImages.Labels;

% Get the confusion matrix
C = confusionmat(testLabels,predictedLabels);
figure();   plotconfusion(testLabels,predictedLabels);