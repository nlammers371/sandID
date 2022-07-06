clear
close all

dateString = '20211028';
dataFolder = ['..\raw_data' filesep dateString filesep];

% specify aggregate size to use
snip_size = 128;
grain_size = '500_';
load_string = [num2str(grain_size) num2str(snip_size)];

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

%%                        
tic
[YPred,scores] = classify(netTransfer,sandImds);
toc

YValidation = sandImds.Labels;
accuracy = mean(YPred == YValidation);

% Generate figures
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

% %% 
% tic
% act = activations(netTransfer,sandImds,'new_fc');
% toc

%% Extract layer activations

tic
act2 = activations(netTransfer,sandImds,'pool5-drop_7x7_s1');
toc

rng(335); % for reproducibility

tic
tsne_scores = tsne(squeeze(act2)');
toc

% generate finer-grained SA labels
lb_vec = sandImds.Labels;
sap_flags = contains(sandImds.Files,'POT') | contains(sandImds.Files,'PSS');
lb_vec(sap_flags) = categorical({'SAP'});

%
close all
tsne_fig = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);

s = gscatter(tsne_scores(:,1),tsne_scores(:,2),lb_vec,[],[],15);
xlabel('tsne component 1')
ylabel('tsne component 2')

saveas(tsne_fig,'tsne_plot.png')
saveas(tsne_fig,'tsne_plot.pdf')
% legend(s,'AB','AR','CRL','CRUH','CRUP','LR','SA');

% icons = findobj(H, 'type', 'patch'); % doesn't work
%icons = findobj(H, '-property', 'Marker', '-and', '-not', 'Marker', 'none'); % also doesn't work

