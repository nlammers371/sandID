% much of this script is copied from:
% https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html
clear
close all
dataFolder = 'C:\Users\nlamm\Dropbox (Personal)\sandClassifier\raw_data\20211124\';

% specify aggregate size to use
size_string = 'sand';

FigPath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\exploratory\20211124\' size_string filesep];
mkdir(FigPath)

% set save path
ReadPath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\classifiers\20211124\' size_string filesep];


% load this network
load([ReadPath 'network_v1.mat'],'netTransfer')

% set input size
inputSize = [224 224 3]; 
  
% generate datastore object
sandImds = imageDatastore(ReadPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');

%%                        
[YPred,scores] = classify(netTransfer,sandImds);

YValidation = sandImds.Labels;
accuracy = mean(YPred == YValidation);

%% Generate figures
% Tabulate the results using a confusion matrix.
confMat = confusionmat(YValidation, YPred);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% get list of category labels
cat_labels = string(unique(sandImds.Labels)');

close all

fig1 = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);

% hold on
heatmap(cat_labels,cat_labels,round(confMat,3));
ylabel('assigned from')
xlabel('assigned to')

set(gca,'Fontsize',14)

saveas(fig1,[FigPath 'confusion_matrix.png'])

%% Do PCA on the results

[coeff, pca_score, latent] = pca(scores);

close all 
pca_fig = figure;
scatter(pca_score(:,1), pca_score(:,2), [], sandImds.Labels)