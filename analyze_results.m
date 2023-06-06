clear
close all

addpath(genpath('./utilities'));

%%%%%%%%%%%%%%%%%%%%
% Use these parameters for Fig 7A
grain_size_cell = {'sand_snips'};

%%%%%%%%%%%%%%%%%%%%
% Use these parameters for Fig 7B
% grain_size_cell = {'500_snips'};

snip_size = 176; % Do not change
suffix = '';
% suffix = '_no_sap';

for g = 1:length(grain_size_cell)
    grain_size = grain_size_cell{g};

    load_string = [num2str(grain_size) '_' num2str(snip_size)  suffix];

    % set writepath 
    DataPath = ['.\data' filesep load_string filesep];   
    
    % set save path
    ReadPath = ['.\classifiers\googlenet_v3\' load_string filesep];
        
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
    [results_struct.umap_scores, umap, results_struct.umap_cluster_ids, extras] = run_umap(results_struct.nn_activations,'verbose','none');
    toc

%     save([ReadPath 'results_struct.mat'],'results_struct')
end