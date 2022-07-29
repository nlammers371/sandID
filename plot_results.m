% clear
close all

addpath(genpath('./utilities'))

dateString = '20211124';
% dateString = '20211028';
grain_size = 'sand';
% grain_size = '500';
dataFolder = ['..\raw_data' filesep dateString filesep];   

% Add colors
color_lb_cell = {'SAP','SA','AB','AR','CRL','CRUP','CRUH','LR'};
% sap_orig = 84.03,22.59,100,8.69;
color_array = [84.03,22.59,100,8.69;
               22.5, 20.12,88.75,0;
               27.04,21.31,21.78,0;
               34.47,94.38,24.14,1.9;
               77.83, 48.34,12.51,0.37;
               52.95,0.09,3.1,0;
               15.09,0,3.5,0;
               0.61,75.47,37.82,0];
            
% color_array_rgb = [37 179 0;
%                   196,204,28;
%                   186,206,199;
%               34.47,94.38,24.14,1.9;
%               77.83, 48.34,12.51,0.37;
%               52.95,0.09,3.1,0;
%               15.09,0,3.5,0;
%               0.61,75.47,37.82,0];            

C = makecform('cmyk2srgb');
color_array_rgb = applycform(color_array/100,C);
% sap_orig = color_array_rgb(1,:);            
% sapss = brighten(sap_orig,0.25);
% sapot = brighten(sap_orig,-0.25);
% color_array_rgb(1,:) = sapss;
% color_array_rgb = [sapot ;  color_array_rgb];

% specify aggregate size to use
snip_size = 176;

load_string = [num2str(grain_size) '_' num2str(snip_size)];

% set path to data
DataPath = ['..\built_data_v2' filesep dateString filesep load_string filesep];

% set save path
ReadPath = ['..\classifiers\googlenet_v3\' load_string filesep];

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
ylabel('true provenance')
xlabel('predicted provenance')

set(gca,'Fontsize',14)

saveas(fig1,[figPath 'confusion_matrix.png'])
saveas(fig1,[figPath 'confusion_matrix.pdf'])

% Make tsne plot
tsne_scores = results_struct.tsne_scores;

% generate finer-grained SA labels
lb_vec = sandImds.Labels;
% sapot_flags = contains(sandImds.Files,'POT');
% sapss_flags = contains(sandImds.Files,'PSS');
% lb_vec(sapot_flags) = categorical({'SAPOT'});
% lb_vec(sapss_flags) = categorical({'SAPSS'});

%%
% close all
tsne_fig = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);
hold on
s = [];
for i = 1:size(color_array_rgb,1)    
    lb = categorical(color_lb_cell(i));
    lb_filter = lb_vec==lb;
    s(end+1) = scatter(tsne_scores(lb_filter,1),tsne_scores(lb_filter,2),20,'MarkerFaceColor',...
              color_array_rgb(i,:),'MarkerEdgeAlpha',0.1,'MarkerEdgeColor','k','MarkerFaceAlpha',0.75);
end    
xlabel('tsne component 1')
ylabel('tsne component 2')
set(gca,'Fontsize',14)
h = legend(s,color_lb_cell{:},'Location','southwest');
set(h,'FontSize',8);

grid on
% ylim([-50 50])
% xlim([-50 50])

saveas(tsne_fig,[figPath 'tsne_plot.png'])
saveas(tsne_fig,[figPath 'tsne_plot.pdf'])
% legend(s,'AB','AR','CRL','CRUH','CRUP','LR','SA');

% icons = findobj(H, 'type', 'patch'); % doesn't work
%icons = findobj(H, '-property', 'Marker', '-and', '-not', 'Marker', 'none'); % also doesn't work

%%

umap_fig = figure;
cmap1 = flipud(brewermap([],'Spectral'));
colormap(cmap1);
hold on
s = [];
for i = 1:size(color_array_rgb,1)    
    lb = categorical(color_lb_cell(i));
    lb_filter = lb_vec==lb;
    s(end+1) = scatter(results_struct.umap_scores(lb_filter,1),results_struct.umap_scores(lb_filter,2),20,'MarkerFaceColor',...
              color_array_rgb(i,:),'MarkerEdgeAlpha',0.1,'MarkerEdgeColor','k','MarkerFaceAlpha',0.75);
end    
xlabel('UMAP component 1')
ylabel('UMAP component 2')
set(gca,'Fontsize',14)
h = legend(s,color_lb_cell{:},'Location','southwest');
set(h,'FontSize',8);

grid on
% ylim([-50 50])
xlim([-15 15])

saveas(umap_fig,[figPath 'UMAP_plot.png'])
saveas(umap_fig,[figPath 'UMAP_plot.pdf'])
% 
% umap_fig = figure;
% cmap1 = flipud(brewermap([],'Spectral'));
% colormap(cmap1);
% 
% s = gscatter(results_struct.umap_scores(:,1),results_struct.umap_scores(:,2),lb_vec,[],[],15);
% xlabel('UMAP component 1')
% ylabel('UMAP component 2')
% 
% saveas(tsne_fig,[figPath 'umap_plot.png'])
% saveas(tsne_fig,[figPath 'umap_plot.pdf'])