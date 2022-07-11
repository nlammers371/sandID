clear
close all

dateString = '20211124';
grain_size = 'sand';
dataFolder = ['..\raw_data' filesep dateString filesep];

% specify aggregate size to use
snip_size = 176;

load_string = [num2str(grain_size) '_' num2str(snip_size)];

% set writepath 
DataPath = ['..\built_data' filesep dateString filesep load_string filesep];   

% set save path
ReadPath = ['..\classifiers\googlenet_v1\' load_string filesep];

figPath = ['..\fig' filesep dateString filesep load_string filesep];
mkdir(figPath)

% load this network
load([ReadPath 'network_v1.mat'],'netTransfer')

% set input size
inputSize = [224 224 3]; 
  
% generate datastore object
sandImds = imageDatastore(DataPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');

   
% initialize structure to store results
results_struct = struct;

tic
[results_struct.YPredicted,results_struct.classScores] = classify(netTransfer,sandImds);
toc

results_struct.YTrue = sandImds.Labels;
results_struct.accuracy = mean(results_struct.YPredicted == results_struct.YTrue);

% Generate figures
% Tabulate the results using a confusion matrix.
results_struct.confMat = confusionmat(results_struct.YTrue, results_struct.YPredicted);

% Convert confusion matrix into percentage form
results_struct.confMat = bsxfun(@rdivide,results_struct.confMat,sum(results_struct.confMat,2));

% Extract layer activations

tic
results_struct.act2 = activations(netTransfer,sandImds,'pool5-drop_7x7_s1');
toc

rng(335); % for reproducibility

tic
results_struct.tsne_scores = tsne(squeeze(results_struct.act2)');
toc

save([ReadPath 'results_Struct.mat'],'results_struct')
