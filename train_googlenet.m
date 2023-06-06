% much of this script is copied from:
% https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html

clear
close all
%%%%%%%%%%%%%%%%%
% Use this option to train model that generated results shown in Fig 7 A
grain_size_cell = {'sand_snips'};

%%%%%%%%%%%%%%%%%
% Use this option to train model that generated results shonwin Fig 7 B
% grain_size_cell = {'500_snips'}

% specify aggregate size to use
snip_size = 176; % do not change this

suffix = '';
% suffix ='_no_sap';

for g = 1:length(grain_size_cell)
      
    rng(236); % for reproducibility

    grain_size = grain_size_cell{g};
    load_string = [num2str(grain_size) '_' num2str(snip_size) suffix];
    
    % set writepath 
    ReadPath = ['.\data' filesep  load_string filesep];   
    
    % set save path
    SavePath = ['.\classifiers\googlenet_v3_test\' load_string filesep];
    mkdir(SavePath)
    
    % define image augmenter
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation',[-90 90]);
    
    inputSize = [224 224 3]; 
      
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
                        'RandYReflection',true, ...
                        'RandRotation',[-90 90]);
                      
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    'DataAugmentation',imageAugmenter);
                  
    % augmenter for validation 
    augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);
    
    options = trainingOptions('sgdm', ...
                              'MiniBatchSize',10, ...
                              'MaxEpochs',12, ...
                              'InitialLearnRate',1e-4, ...
                              'Shuffle','every-epoch', ...
                              'ValidationData',augimdsValidation, ...
                              'ValidationFrequency',3, ...
                              'Verbose',false, ...
                              'Plots','training-progress');
      
    % TRAIN
    netTransfer = trainNetwork(augimdsTrain,lgraph,options);
   
    % save this network
    save([SavePath 'network_v1.mat'],'netTransfer')
    % save training and testing datastores
    save([SavePath 'imdsValidation.mat'],'augimdsValidation')
    save([SavePath 'imdsTraining.mat'],'augimdsTrain')
end