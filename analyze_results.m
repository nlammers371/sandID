clear
close all

addpath(genpath('/utilities'));
% dateString = '20211028';
dateString = '20211124';
% grain_size = 'sand';
dataFolder = ['..\raw_data' filesep dateString filesep];

% specify aggregate size to use
snip_size = 176;

% grain_size_cell = {'500','5001','12mm','_2mm'};
grain_size_cell = {'sand'};

for g = 1:length(grain_size_cell)
    grain_size = grain_size_cell{g};

    load_string = [num2str(grain_size) '_' num2str(snip_size)];

    % set writepath 
    DataPath = ['..\built_data' filesep dateString filesep load_string filesep];   
    
    % set save path
    ReadPath = ['..\classifiers\googlenet_v2\' load_string filesep];
        
    % load this network
    load([ReadPath 'network_v1.mat'],'netTransfer')
    
    % load sample data stores 
    load([ReadPath 'imdsValidation.mat'])
    load([ReadPath 'imdsTraining.mat'])
    
      
    % generate datastore object
    sandImds = imageDatastore(DataPath, ...
                              'IncludeSubfolders',true, ...
                              'LabelSource','foldernames');       
       
    % remove training samples from set before generating confusion matrix
    training_flags = ismember(sandImds.Files,augimdsTrain.Files);

    subSandImds = subset(sandImds,~training_flags);   

    % initialize structure to store results
    results_struct = struct;
    
    tic
    [results_struct.YPredicted,results_struct.classScores] = classify(netTransfer,subSandImds);
    toc
    
    results_struct.YTrue = subSandImds.Labels;
%     results_struct.accuracy = mean(results_struct.YPredicted == results_struct.YTrue);
    
    % Generate figures
    % Tabulate the results using a confusion matrix.
    results_struct.confMat = confusionmat(results_struct.YTrue, results_struct.YPredicted);
    
    % Convert confusion matrix into percentage form
    results_struct.confMat = bsxfun(@rdivide,results_struct.confMat,sum(results_struct.confMat,2));
    
    % calculate overall accuracy 
    results_struct.accuracy = mean(diag(results_struct.confMat));
    
    % Extract layer activations (do this for all data points)
    tic
    results_struct.nn_activations = squeeze(activations(netTransfer,sandImds,'pool5-drop_7x7_s1'))';
    toc
    
    rng(335); % for reproducibility
    
    tic
    results_struct.tsne_scores = tsne(results_struct.nn_activations);
    toc

    tic
    [results_struct.umap_scores, umap, results_struct.umap_cluster_ids, extras]=run_umap(results_struct.nn_activations,'verbose','none');
    toc

    save([ReadPath 'results_struct.mat'],'results_struct')
end