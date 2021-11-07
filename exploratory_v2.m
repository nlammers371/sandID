% much of this script is copied from:
% https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html

clear
close all
dataFolder = 'C:\Users\nlamm\Dropbox (Personal)\sandClassifier\raw_data\20211028\';

% specify aggregate size to use
size_string = '12mm';

% set writepath 
ReadPath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\built_data\20211028\' size_string filesep];

% set save path
SavePath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\classifiers\20211028\' size_string filesep];
mkdir(SavePath)

% define image augmenter
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-90 90], ...
    'RandScale',[1 2]);

inputSize = [256 256 3]; 
  
% generate datastore object
sandImds = imageDatastore(ReadPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');
                        
% agument
sandImdsAug = augmentedImageDatastore(inputSize, sandImds,...
                      'DataAugmentation', imageAugmenter);

% divide data into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(sandImds,0.7,'randomized');
                        
% define NN 
net = googlenet;

%% adjust architecture to work with our class number
lgraph = layerGraph(net); 
numClasses = numel(categories(imdsTrain.Labels));
% define new layer
newLearnableLayer = fullyConnectedLayer(numClasses, ...
                    'Name','new_fc', ...
                    'WeightLearnRateFactor',10, ...
                    'BiasLearnRateFactor',10);
% replace
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

% define data augmenter
imageAugmenter = imageDataAugmenter( ...
                    'RandXReflection',true, ...
                    'RandRotation',[-90 90], ...
                    'RandScale',[1 2]);
                  
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                'DataAugmentation',imageAugmenter);
              
% augmenter for validation 
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

options = trainingOptions('sgdm', ...
                          'MiniBatchSize',10, ...
                          'MaxEpochs',6, ...
                          'InitialLearnRate',1e-4, ...
                          'Shuffle','every-epoch', ...
                          'ValidationData',augimdsValidation, ...
                          'ValidationFrequency',3, ...
                          'Verbose',false, ...
                          'Plots','training-progress');
  
% TRAIN
netTransfer = trainNetwork(augimdsTrain,lgraph,options);


[YPred,scores] = classify(netTransfer,augimdsValidation);

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

% save this network
save([SavePath 'network_v1.mat'],'netTransfer')