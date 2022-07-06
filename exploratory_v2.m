% much of this script is copied from:
% https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html

clear
close all
dataRoot = 'C:\Users\nlamm\Dropbox (Personal)\sandID';
if ~exist(dataRoot)
    dataRoot = 'S:\Nick\Dropbox (Personal)\sandID\';
end

dataFolder = [dataRoot '\raw_data\20211124\'];

% specify aggregate size to use
snip_size = 128;
load_string = ['sand_' num2str(snip_size)];

% set writepath 
ReadPath = [dataRoot '\built_data\' dateString filesep load_string filesep];   

% set save path
SavePath = [dataRoot '\classifiers\' dateString filesep load_string filesep];
mkdir(SavePath)


% define image augmenter
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-90 90]);

inputSize = [snip_size snip_size 3]; 
  
% generate datastore object
sandImds = imageDatastore(ReadPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');
                        
% agument
sandImdsAug = augmentedImageDatastore(inputSize, sandImds,...
                      'DataAugmentation', imageAugmenter);

% balance classes
tbl = countEachLabel(sandImds);

% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% Use splitEachLabel method to trim the set.
sandImds = splitEachLabel(sandImds, minSetCount, 'randomize');
%%
                    
% divide data into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(sandImds,0.7,'randomized');
                        
% define NN 
net = googlenet;

% adjust architecture to work with our class number
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


% Tabulate the results using a confusion matrix.
confMat = confusionmat(YValidation, YPred);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% save this network
save([SavePath 'network_v1.mat'],'netTransfer')
