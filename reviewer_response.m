clear
close all

addpath(genpath('./utilities'));

% specify aggregate size to use
snip_size = 176;

% grain_size_cell = {'500_snips'};
grain_size_cell = {'sand_snips'};

suffix = '';
g = 1;
grain_size = grain_size_cell{g};

load_string = [num2str(grain_size) '_' num2str(snip_size) suffix];

% set path to trainin data
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
f_list_train = augimdsTrain.Files;
f_train_short = cell(size(f_list_train));
for f = 1:length(f_list_train)
    f_long = f_list_train{f};
    slashes = strfind(f_long,'\');
    f_short = f_long(slashes(end)+1:end);
    f_train_short{f} = f_short;
end

f_list_all = sandImds.Files;
f_all_short = cell(size(f_list_all));
for f = 1:length(f_list_all)
    f_long = f_list_all{f};
    slashes = strfind(f_long,'\');
    f_short = f_long(slashes(end)+1:end);
    f_all_short{f} = f_short;
end


training_flags = ismember(f_all_short,f_train_short);

testSandImds = subset(sandImds,~training_flags);   
trainSandImds = subset(sandImds,training_flags);  


tic
[YPredictedTest,classScoresTest] = classify(netTransfer,testSandImds);
[YPredictedTrain,classScoresTrain] = classify(netTransfer,trainSandImds);
toc

YTrueTest = testSandImds.Labels;
YTrueTrain = trainSandImds.Labels;
%     results_struct.accuracy = mean(results_struct.YPredicted == results_struct.YTrue);

% Generate figures
% Tabulate the results using a confusion matrix.
confMatTest = confusionmat(YTrueTest, YPredictedTest);
confMatTrain = confusionmat(YTrueTrain, YPredictedTrain);

% Convert confusion matrix into percentage form
confMatTest = bsxfun(@rdivide,confMatTest,sum(confMatTest,2));
confMatTrain = bsxfun(@rdivide,confMatTrain,sum(confMatTrain,2));

% calculate overall accuracy 
accuracyTest = mean(diag(confMatTest))
accuracyTrain = mean(diag(confMatTrain))


correct_fractions_test = diag(confMatTest);
% combine CR sources
bulk_accuracy_cr = correct_fractions_test;
bulk_accuracy_cr(4) = bulk_accuracy_cr(4) + confMatTest(4,5);
bulk_accuracy_cr(5) = bulk_accuracy_cr(5) + confMatTest(5,4);
bulk_accuracy_cr(4) = mean([bulk_accuracy_cr(4) bulk_accuracy_cr(5)]);
bulk_accuracy_cr = [bulk_accuracy_cr(1:4) ; bulk_accuracy_cr(6:end)];
bulk_accuracy_cr_test = mean(bulk_accuracy_cr)


correct_fractions_train = diag(confMatTrain);
% combine CR sources
bulk_accuracy_cr = correct_fractions_train;
bulk_accuracy_cr(4) = bulk_accuracy_cr(4) + confMatTrain(4,5);
bulk_accuracy_cr(5) = bulk_accuracy_cr(5) + confMatTrain(5,4);
bulk_accuracy_cr(4) = mean([bulk_accuracy_cr(4) bulk_accuracy_cr(5)]);
bulk_accuracy_cr = [bulk_accuracy_cr(1:4) ; bulk_accuracy_cr(6:end)];
bulk_accuracy_cr_train = mean(bulk_accuracy_cr)



