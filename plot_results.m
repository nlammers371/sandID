% clear
close all

addpath(genpath('./utilities'))

dateString = '20211124';
grain_size = 'sand';
dataFolder = ['..\raw_data' filesep dateString filesep];   
                        
% specify aggregate size to use
snip_size = 176;

load_string = [num2str(grain_size) '_' num2str(snip_size)];

% set path to data
DataPath = ['..\built_data' filesep dateString filesep load_string filesep];

% set save path
ReadPath = ['..\classifiers\googlenet_v2\' load_string filesep];

figPath = ['..\fig' filesep dateString filesep load_string filesep];
mkdir(figPath)

% load this network
load([ReadPath 'results_struct.mat'],'results_struct')

% load saved datastores
% load([ReadPath 'results_struct.mat'],'results_struct')


% generate datastore object
sandImds = imageDatastore(DataPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');
                        
% Generate figures

% Tabulate the results using a confusion matrix.
confMat = results_struct.confMat;

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% get list of category labels
cat_labels = string(unique(sandImds.Labels)');

% close all

fig1 = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);

% hold on
heatmap(cat_labels,cat_labels,round(confMat,3));
ylabel('assigned from')
xlabel('assigned to')

set(gca,'Fontsize',14)

saveas(fig1,[figPath 'confusion_matrix.png'])

% Make tsne plot
tsne_scores = results_struct.tsne_scores;

% generate finer-grained SA labels
lb_vec = results_struct.YTrue;
sap_flags = contains(sandImds.Files,'POT') | contains(sandImds.Files,'PSS');
lb_vec(sap_flags) = categorical({'SAP'});

%
% close all
tsne_fig = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);

s = gscatter(tsne_scores(:,1),tsne_scores(:,2),lb_vec,[],[],15);
xlabel('tsne component 1')
ylabel('tsne component 2')

saveas(tsne_fig,[figPath 'tsne_plot.png'])
saveas(tsne_fig,[figPath 'tsne_plot.pdf'])
% legend(s,'AB','AR','CRL','CRUH','CRUP','LR','SA');

% icons = findobj(H, 'type', 'patch'); % doesn't work
%icons = findobj(H, '-property', 'Marker', '-and', '-not', 'Marker', 'none'); % also doesn't work

