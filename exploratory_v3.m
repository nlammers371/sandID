% much of this script is copied from:
% https://www.mathworks.com/help/vision/ug/image-category-classification-using-deep-learning.html

clear
close all
dataFolder = 'C:\Users\nlamm\Dropbox (Personal)\sandClassifier\raw_data\20211028\';

% specify aggregate size to use
size_string = '5001mm';

% set writepath 
ReadPath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\built_data\20211028\' size_string filesep];

% set save path
SavePath = ['C:\Users\nlamm\Dropbox (Personal)\sandClassifier\classifiers\20211028\' size_string filesep];
mkdir(SavePath)

% Load pretrained network
net = resnet50();

%% define image augmenter

inputSize = [256 256 3]; 
  
% generate datastore object
sandImds = imageDatastore(ReadPath, ...
                          'IncludeSubfolders',true, ...
                          'LabelSource','foldernames');                        

% balance classes
tbl = countEachLabel(sandImds)

% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% Use splitEachLabel method to trim the set.
sandImds = splitEachLabel(sandImds, minSetCount, 'randomize');

% divide data into training and validation sets
[trainingSet, testSet] = splitEachLabel(sandImds,0.3,'randomize');
                        
% agument
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet);
augmentedTestSet = augmentedImageDatastore(imageSize, testSet);

%% adjust architecture to work with our class number
% Get the network weights for the second convolutional layer
% w1 = net.Layers(2).Weights;
% 
% % Scale and resize the weights for visualization
% w1 = mat2gray(w1);
% w1 = imresize(w1,5); 
% 
% % Display a montage of network weights. There are 96 individual sets of
% % weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')

featureLayer = 'fc1000';
tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
toc  
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
  
%%
                  
% Extract test features using the CNN
tic
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
toc

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))
%%
[coeff,score,latent] = pca(testFeatures');
