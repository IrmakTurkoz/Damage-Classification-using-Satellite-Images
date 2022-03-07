%Customizing matlab
clear;
clc;
close all;
warning('off','all')
warning
set(0,'DefaultFigureWindowStyle','docked')

load('data.mat')
firstN = 100;
% Divide data into test and train
% 70% train 30% test
train_data = data(3,1:200) ;
test_data = data(:,201:end);
test_flat = reshape(test_data.',1,[]);
train_flat = reshape(train_data.',1,[]);
% Create a classifier.
train_labels = zeros(1, length(train_flat));
train_labels(1:length(train_flat)/4) = 1;
train_labels(length(train_flat)/4+1:length(train_flat)/2) = 2;
train_labels(length(train_flat)/2+1:length(train_flat)*3/4) = 3;
train_labels(length(train_flat)*3/4+1:end) = 4;


test_labels = zeros(1, length(test_flat));
test_labels(1:length(test_flat)/4) = 1;
test_labels(length(test_flat)/4+1:length(test_flat)/2) = 2;
test_labels(length(test_flat)/2+1:length(test_flat)*3/4) = 3;
test_labels(length(test_flat)*3/4+1:end) = 4;

% features = zeros(length(train_flat),135);
% features_test = zeros(length(test_flat),135);
index = 1;
features = zeros( 10000,65);
features_test = zeros(10000,65);
for i = 1 : length(train_flat)
    I = rgb2gray(train_flat{i});
%     features(i,:) = extractLBPFeatures(I,'NumNeighbors',12);
    mser_features = detectMSERFeatures(I,'RegionAreaRange',[30 1000]);
    [ft, regions]  = extractFeatures(I, mser_features);
    ft(:,65) = train_labels(i);
    features(index : index-1 + length(ft(:,1)),:) = ft;
    index = index+ length(ft(:,1));
%     total_pixels = 0;
%     for pix = 1:mser_features.Count
%         total_pixels = total_pixels + length(mser_features.PixelList(pix));
%     end

    
%     features(i,1) = total_pixels;
%     features(i,2) = mser_features.Count;
    
%     figure;
%     imshow(I); hold on;
%     plot(mser_features,'showPixelList',true,'showEllipses',false);
%     drawnow();
%     pause(1);
end
index = 1;
for i = 1 : length(test_flat)
    I = rgb2gray(test_flat{i});

%     features_test(i,:) = extractLBPFeatures(I,'NumNeighbors',12);
    mser_features = detectMSERFeatures(I,'RegionAreaRange',[30 1000]);
    [ft, regions]  = extractFeatures(I, mser_features);
    ft(:,65) = test_labels(i);
    features_test(index : index-1 + length(ft(:,1)),:) = ft;
    index = index+ length(ft(:,1));
%     total_pixels = 0;
%     for pix = 1:mser_features.Count
%         total_pixels = total_pixels + length(mser_features.PixelList(pix));
%     end
%     features_test(i,1) = total_pixels;
%     features_test(i,2) = mser_features.Count;
   
end
features = features(any(features,2),:);
features_test = features_test(any(features_test,2),:);
mdl = fitcdiscr(features(:,1:64),features(:,65),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'))
predicted = predict(mdl,features_test(:,1:64));
% Mdl =  fitcsvm(features(:,1),train_labels,'CrossVal','on');

% predicted = predict(Mdl,features_test);
% Model=svm.train(features,train_labels);
% predicted=svm.predict(Model,features_test);
confusion_matrix =  confusionchart(features_test(:,65),predicted);
% confusion_matrix =  plotconfusion(test_labels,predicted);

% trues = confusion_matrix(1,1)+ confusion_matrix(2,2) + confusion_matrix(3,3) + confusion_matrix(4,4);
% accuracy = trues / length(test_labels)
% classErr1 = kfoldLoss(mdl,'LossFun','ClassifErr')
