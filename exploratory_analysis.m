clear
close all
dataFolder = 'C:\Users\nlamm\Dropbox (Personal)\sandClassifier\raw_data\20220419\';
file_list = dir([dataFolder '*jpg']);
data_struct = struct;
FigPath = 'C:\Users\nlamm\Dropbox (Personal)\sandClassifier\exploratory\';
mkdir(FigPath)
% identify region suitable for classification
filter_size = 15;

for f = 1:length(file_list)
    data_struct(f).im_raw_bg = imread([dataFolder file_list(f).name]);
    data_struct(f).im_raw = imresize(data_struct(f).im_raw_bg,.1);
    data_struct(f).im_sm = imgaussfilt(data_struct(f).im_raw,filter_size);
    data_struct(f).im_sm_db = imgaussfilt(double(data_struct(f).im_raw),filter_size);
    % perform basic rgb segmentation
    data_struct(f).pixel_labels = imsegkmeans(data_struct(f).im_sm,2,'NumAttempts',3);
    
    % we want to identify "core" of sand well-removed from surounding paper    
    im_stats = regionprops(data_struct(f).pixel_labels==2,'Area','Centroid','MajorAxisLength');
    
    % find largest region
    [~,mi] = max([im_stats.Area]);
    
    % generate mask
    centroid = round(im_stats(mi).Centroid);
    diameter = im_stats(mi).MajorAxisLength;
    im_mask = zeros(size(data_struct(f).pixel_labels));
    im_mask(centroid(2),centroid(1)) = 1;
    dist_mat = bwdist(im_mask);
    im_mask(dist_mat<=0.25*diameter) = 1;
    
    data_struct(f).im_mask = im_mask;
end    

%% Experiment with PCA to see how image colors spread

% extract pixel values
image_id_vec = [];
pixel_val_array = [];
for d = 1:length(data_struct)
    px_temp = [];
    im_temp = data_struct(d).im_sm_db;
    for i = 1:3
        slice = im_temp(:,:,i);
        px_temp = [px_temp double(slice(data_struct(d).im_mask==1))];
    end
    pixel_val_array = vertcat(px_temp,pixel_val_array);
    image_id_vec = vertcat(repelem(d,size(px_temp,1))',image_id_vec);
end    

% run PCA
% rng(124);
% n_dp = 5e4;
% rs_ids = randsample(1:length(pixel_val_array),n_dp);
[coeff,score,latent] = pca(pixel_val_array);

close all

pca_fig12 = figure;
hold on
cmap1 = brewermap(length(data_struct),'Spectral');
colormap(cmap1);

for i = fliplr(1:length(data_struct))
    scatter(score(image_id_vec==i,1),score(image_id_vec==i,2),[],'MarkerFaceColor',cmap1(i,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0)
end
inc = 1/length(data_struct)/2;
h = colorbar('YTick',linspace(inc,1-inc,length(data_struct)),'YTickLabel', 1:length(data_struct));
set(gca,'Fontsize',14)
grid on
xlabel('principle component 1')
ylabel('principle component 2')
ylabel(h,'sample ID')
saveas(pca_fig12,[FigPath 'pca_12.png'])
saveas(pca_fig12,[FigPath 'pca_12.pdf'])

pca_fig13 = figure;
hold on
cmap1 = brewermap(length(data_struct),'Spectral');
colormap(cmap1);

for i = fliplr(1:length(data_struct))
    scatter(score(image_id_vec==i,1),score(image_id_vec==i,3),[],'MarkerFaceColor',cmap1(i,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0)
end
inc = 1/length(data_struct)/2;
h = colorbar('YTick',linspace(inc,1-inc,length(data_struct)),'YTickLabel', 1:length(data_struct));
set(gca,'Fontsize',14)
grid on
xlabel('principle component 1')
ylabel('principle component 3')
ylabel(h,'sample ID')
saveas(pca_fig13,[FigPath 'pca_13.png'])
saveas(pca_fig13,[FigPath 'pca_13.pdf'])

pca_fig23 = figure;
hold on
cmap1 = brewermap(length(data_struct),'Spectral');
colormap(cmap1);

for i = fliplr(1:length(data_struct))
    scatter(score(image_id_vec==i,2),score(image_id_vec==i,3),[],'MarkerFaceColor',cmap1(i,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0)
end
inc = 1/length(data_struct)/2;
h = colorbar('YTick',linspace(inc,1-inc,length(data_struct)),'YTickLabel', 1:length(data_struct));
set(gca,'Fontsize',14)
grid on
xlabel('principle component 2')
ylabel('principle component 3')
ylabel(h,'sample ID')
saveas(pca_fig23,[FigPath 'pca_23.png'])
saveas(pca_fig23,[FigPath 'pca_23.pdf'])

%%
close all

pca_fig = figure;
hold on
cmap1 = brewermap(length(data_struct),'Spectral');
colormap(cmap1);

for i = fliplr(1:length(data_struct))
    scatter3(score(image_id_vec==i,1),score(image_id_vec==i,2),score(image_id_vec==i,3),...
          [],'MarkerFaceColor',cmap1(i,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.15,'MarkerEdgeAlpha',0)
end
inc = 1/length(data_struct)/2;
h = colorbar('YTick',linspace(inc,1-inc,length(data_struct)),'YTickLabel', 1:length(data_struct));
set(gca,'Fontsize',14)
grid on
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
ylabel(h,'sample ID')
view(30,15)
saveas(pca_fig,[FigPath 'pca_3D.png'])


pca_fig = figure;
hold on
cmap1 = brewermap(length(data_struct),'Spectral');
colormap(cmap1);

for i = fliplr(1:length(data_struct))
    scatter3(pixel_val_array(image_id_vec==i,1),pixel_val_array(image_id_vec==i,2),pixel_val_array(image_id_vec==i,3),...
          [],'MarkerFaceColor',cmap1(i,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.15,'MarkerEdgeAlpha',0)
end
inc = 1/length(data_struct)/2;
h = colorbar('YTick',linspace(inc,1-inc,length(data_struct)),'YTickLabel', 1:length(data_struct));
set(gca,'Fontsize',14)
grid on
xlabel('red')
ylabel('green')
zlabel('blue')
ylabel(h,'sample ID')
view(30,15)
saveas(pca_fig,[FigPath 'rgb_3D.png'])