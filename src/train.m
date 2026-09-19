%% ================================================================
%  ARCFACE FACE RECOGNITION
%  TRAINING SCRIPT
%
%  Model : InsightFace w600k_r50.onnx
%  MATLAB: R2026a
% ================================================================

clear;
clc;
close all;


%% ------------------------------------------------
% PROJECT PATH
% -------------------------------------------------

projectRoot = fileparts(mfilename('fullpath'));

datasetPath = fullfile(projectRoot,'dataset');

trainPath = fullfile(datasetPath,'train_augmented');

modelPath = fullfile( ...
    projectRoot, ...
    'models', ...
    'arcface.onnx');

resultPath = fullfile( ...
    projectRoot, ...
    'results');

trainedModel = fullfile( ...
    resultPath, ...
    'ArcFaceRecognitionModel.mat');


%% ------------------------------------------------
% DISPLAY
% -------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf('              ARCFACE FACE RECOGNITION\n');
fprintf('                     TRAINING\n');
fprintf('============================================================\n');

fprintf('Project folder:\n%s\n\n',projectRoot);

fprintf('Training folder:\n%s\n\n',trainPath);

fprintf('ArcFace model:\n%s\n\n',modelPath);


%% ------------------------------------------------
% CHECK REQUIRED FILES/FOLDERS
% -------------------------------------------------

if ~isfolder(trainPath)

    error([ ...
        '\nTraining folder not found.\n\n' ...
        'Expected:\n%s\n'], ...
        trainPath);

end


if ~isfile(modelPath)

    error([ ...
        '\nArcFace ONNX model not found.\n\n' ...
        'Expected:\n%s\n\n' ...
        'Put w600k_r50.onnx inside the models folder ' ...
        'and rename it to arcface.onnx.\n'], ...
        modelPath);

end


%% ------------------------------------------------
% CREATE RESULTS FOLDER
% -------------------------------------------------

if ~isfolder(resultPath)

    mkdir(resultPath);

end


%% ------------------------------------------------
% CHECK ONNX IMPORTER
% -------------------------------------------------

if exist('importNetworkFromONNX','file') ~= 2

    error([ ...
        '\nONNX importer is not available.\n\n' ...
        'Install:\n' ...
        'Deep Learning Toolbox Converter for ONNX Model Format\n\n' ...
        'MATLAB:\n' ...
        'Home -> Add-Ons -> Get Add-Ons\n']);

end


%% ------------------------------------------------
% CHECK DATASET
% -------------------------------------------------

imds = imageDatastore( ...
    trainPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


if isempty(imds.Files)

    error('No training images found.');

end


classes = categories(imds.Labels);


fprintf('Training images : %d\n', ...
    numel(imds.Files));

fprintf('Classes         : %d\n\n', ...
    numel(classes));


fprintf('Classes:\n');


for k = 1:numel(classes)

    fprintf( ...
        '  %d. %s\n', ...
        k,classes{k});

end


if numel(classes) < 2

    error( ...
        'At least TWO person folders are required.');

end


%% ------------------------------------------------
% LOAD ARCFACE
% -------------------------------------------------

fprintf('\n');
fprintf('Loading ArcFace...\n');

recognizer = ...
    ArcFaceRecognition(modelPath);


%% ------------------------------------------------
% TRAIN
% -------------------------------------------------

fprintf('\n');
fprintf('Starting feature extraction and training...\n');

tic;

recognizer.train(trainPath);

trainingTime = toc;


%% ------------------------------------------------
% SAVE
% -------------------------------------------------

recognizer.saveModel(trainedModel);


%% ------------------------------------------------
% FINISHED
% -------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 TRAINING COMPLETE\n');
fprintf('============================================================\n');

fprintf( ...
    'Training time : %.2f seconds\n', ...
    trainingTime);

fprintf('\nSaved trained model:\n');

fprintf( ...
    '%s\n', ...
    trainedModel);

fprintf('\nNext step:\n');

fprintf('Run test.m\n');

fprintf('============================================================\n');