%% ========================================================================
%  AUGMENT DATASET - OCCLUSION-AWARE FACE RECOGNITION
%
%  Project:
%  C:\228004030\occulationfacerecongnition
%
%  INPUT:
%      dataset\train\person 1
%      dataset\train\person 2
%
%  OUTPUT:
%      dataset\train_augmented\person 1
%      dataset\train_augmented\person 2
%
%  IMPORTANT:
%      The TEST dataset is NOT modified.
%
%  Augmentations:
%      1. Horizontal flip
%      2. Small rotation
%      3. Brightness variation
%      4. Contrast variation
%      5. Small translation
%      6. Gaussian noise
%      7. Mild Gaussian blur
%      8. Eye-region occlusion
%      9. Mouth-region occlusion
%     10. Left-side occlusion
%     11. Right-side occlusion
%
%  Each original image produces multiple augmented images.
% ========================================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       OCCLUSION-AWARE FACE RECOGNITION\n');
fprintf('                 DATA AUGMENTATION\n');
fprintf('============================================================\n\n');

%% ========================================================================
%  1. PROJECT PATHS
% ========================================================================

projectPath = 'C:\228004030\occulationfacerecongnition';

trainPath = fullfile(projectPath, 'dataset', 'train');

% IMPORTANT:
% Augmented images are stored separately from the original training set.
augmentedPath = fullfile(projectPath, 'dataset', 'train_augmented');

fprintf('Project path:\n%s\n\n', projectPath);

fprintf('Original training dataset:\n%s\n\n', trainPath);

fprintf('Augmented training dataset:\n%s\n\n', augmentedPath);


%% ========================================================================
%  2. CHECK TRAINING DATASET
% ========================================================================

if ~isfolder(trainPath)
    error('Training folder does not exist:\n%s', trainPath);
end


%% ========================================================================
%  3. CREATE OUTPUT DIRECTORY
% ========================================================================

if isfolder(augmentedPath)

    fprintf('Existing augmented dataset found.\n');
    fprintf('It will be deleted and recreated.\n\n');

    try
        rmdir(augmentedPath, 's');
    catch ME
        error(['Unable to delete the existing augmented dataset.\n' ...
               'Close any files using it and try again.\n\n' ...
               'MATLAB error: %s'], ME.message);
    end
end

mkdir(augmentedPath);


%% ========================================================================
%  4. GET PERSON FOLDERS
% ========================================================================

folderInfo = dir(trainPath);

folderInfo = folderInfo([folderInfo.isdir]);

folderNames = {};

for i = 1:numel(folderInfo)

    name = folderInfo(i).name;

    if strcmp(name, '.') || strcmp(name, '..')
        continue;
    end

    folderNames{end+1} = name; %#ok<SAGROW>
end

if isempty(folderNames)
    error('No person folders were found inside:\n%s', trainPath);
end


fprintf('Person classes found:\n');

for i = 1:numel(folderNames)
    fprintf('  %d. %s\n', i, folderNames{i});
end

fprintf('\n');


%% ========================================================================
%  5. AUGMENTATION SETTINGS
% ========================================================================

% Number of augmented images generated per original image.

numAugmentedPerImage = 10;

fprintf('Augmented images per original image : %d\n', ...
    numAugmentedPerImage);

fprintf('\n');


%% ========================================================================
%  6. IMAGE EXTENSIONS
% ========================================================================

extensions = { ...
    '*.jpg', ...
    '*.jpeg', ...
    '*.png', ...
    '*.bmp', ...
    '*.tif', ...
    '*.tiff'};


%% ========================================================================
%  7. STATISTICS
% ========================================================================

totalOriginal = 0;
totalGenerated = 0;

classOriginalCounts = zeros(numel(folderNames),1);
classAugmentedCounts = zeros(numel(folderNames),1);


%% ========================================================================
%  8. PROCESS EACH PERSON
% ========================================================================

for classIndex = 1:numel(folderNames)

    personName = folderNames{classIndex};

    sourceFolder = fullfile(trainPath, personName);
    outputFolder = fullfile(augmentedPath, personName);

    if ~isfolder(outputFolder)
        mkdir(outputFolder);
    end

    fprintf('------------------------------------------------------------\n');
    fprintf('Processing class: %s\n', personName);
    fprintf('------------------------------------------------------------\n');

    %% --------------------------------------------------------------------
    % Find images
    % ---------------------------------------------------------------------

    imageFiles = [];

    for e = 1:numel(extensions)

        tempFiles = dir(fullfile(sourceFolder, extensions{e}));

        if ~isempty(tempFiles)
            imageFiles = [imageFiles; tempFiles]; %#ok<AGROW>
        end
    end

    if isempty(imageFiles)

        warning('No images found in:\n%s', sourceFolder);
        continue;
    end

    fprintf('Original images found : %d\n', numel(imageFiles));

    classOriginalCounts(classIndex) = numel(imageFiles);

    %% --------------------------------------------------------------------
    % Process each original image
    % ---------------------------------------------------------------------

    for imageIndex = 1:numel(imageFiles)

        fileName = imageFiles(imageIndex).name;
        inputFile = fullfile(sourceFolder, fileName);

        try
            originalImage = imread(inputFile);
        catch ME

            warning('Unable to read image: %s\n%s', ...
                inputFile, ME.message);

            continue;
        end

        % Convert grayscale to RGB
        if ndims(originalImage) == 2
            originalImage = repmat(originalImage, [1 1 3]);
        end

        % Remove alpha channel if present
        if size(originalImage,3) > 3
            originalImage = originalImage(:,:,1:3);
        end

        % Convert to uint8
        if ~isa(originalImage,'uint8')

            if isfloat(originalImage)

                if max(originalImage(:)) <= 1
                    originalImage = uint8(255 * originalImage);
                else
                    originalImage = uint8(originalImage);
                end

            else
                originalImage = uint8(originalImage);
            end
        end

        % Save original image into augmented training folder.
        %
        % This means train_augmented contains both:
        %   - original training images
        %   - augmented training images
        %
        [~, baseName, ext] = fileparts(fileName);

        originalOutputName = sprintf( ...
            '%s_original%s', baseName, ext);

        originalOutputFile = fullfile( ...
            outputFolder, originalOutputName);

        imwrite(originalImage, originalOutputFile);

        totalOriginal = totalOriginal + 1;


        %% ================================================================
        %  Generate augmented images
        % ================================================================

        for augIndex = 1:numAugmentedPerImage

            augmentedImage = createAugmentedImage( ...
                originalImage, augIndex);


            %% ------------------------------------------------------------
            % Output filename
            % ------------------------------------------------------------

            augmentedName = sprintf( ...
                '%s_aug_%02d.jpg', ...
                baseName, augIndex);

            outputFile = fullfile( ...
                outputFolder, augmentedName);


            %% ------------------------------------------------------------
            % Save image
            % ------------------------------------------------------------

            imwrite(augmentedImage, outputFile, ...
                'Quality', 95);

            totalGenerated = totalGenerated + 1;

            classAugmentedCounts(classIndex) = ...
                classAugmentedCounts(classIndex) + 1;

        end


        %% ---------------------------------------------------------------
        % Progress
        % ---------------------------------------------------------------

        fprintf('  %4d / %4d completed: %s\n', ...
            imageIndex, ...
            numel(imageFiles), ...
            fileName);

    end

    fprintf('\n');

end


%% ========================================================================
%  9. FINAL SUMMARY
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 AUGMENTATION COMPLETED\n');
fprintf('============================================================\n\n');

fprintf('Original images processed : %d\n', totalOriginal);
fprintf('Augmented images created  : %d\n', totalGenerated);

fprintf('Total training images     : %d\n', ...
    totalOriginal + totalGenerated);

fprintf('\n');

fprintf('Class-wise results:\n\n');

for classIndex = 1:numel(folderNames)

    fprintf('%-15s : Original = %4d | Augmented = %4d | Total = %4d\n', ...
        folderNames{classIndex}, ...
        classOriginalCounts(classIndex), ...
        classAugmentedCounts(classIndex), ...
        classOriginalCounts(classIndex) + ...
        classAugmentedCounts(classIndex));

end

fprintf('\n');

fprintf('Augmented dataset:\n%s\n', augmentedPath);

fprintf('\n');
fprintf('TEST DATASET WAS NOT MODIFIED.\n');

fprintf('\n');
fprintf('============================================================\n');
fprintf('                         NEXT STEP\n');
fprintf('============================================================\n\n');

fprintf('Your new training folder is:\n');
fprintf('dataset\\train_augmented\\\n\n');

fprintf('Change train.m from:\n');
fprintf('dataset\\train\\\n\n');

fprintf('to:\n');
fprintf('dataset\\train_augmented\\\n\n');

fprintf('Then retrain the model.\n');

fprintf('============================================================\n\n');


%% ========================================================================
%  LOCAL FUNCTION
%  CREATE AUGMENTED IMAGE
% ========================================================================

function outputImage = createAugmentedImage(inputImage, augIndex)

    % Convert image to double for processing
    img = im2double(inputImage);

    [H, W, ~] = size(img);


    %% ====================================================================
    %  DIFFERENT AUGMENTATION COMBINATIONS
    %
    %  Each augIndex intentionally uses a different realistic variation.
    % ====================================================================

    switch augIndex

        %% ----------------------------------------------------------------
        % 1. Horizontal flip + small rotation
        % -----------------------------------------------------------------

        case 1

            img = fliplr(img);

            angle = -8 + 16*rand();

            img = imrotate(img, angle, 'bilinear', 'crop');


        %% ----------------------------------------------------------------
        % 2. Small positive rotation
        % -----------------------------------------------------------------

        case 2

            angle = 5 + 5*rand();

            img = imrotate(img, angle, 'bilinear', 'crop');


        %% ----------------------------------------------------------------
        % 3. Small negative rotation
        % -----------------------------------------------------------------

        case 3

            angle = -(5 + 5*rand());

            img = imrotate(img, angle, 'bilinear', 'crop');


        %% ----------------------------------------------------------------
        % 4. Brightness variation
        % -----------------------------------------------------------------

        case 4

            brightnessChange = -0.15 + 0.30*rand();

            img = img + brightnessChange;


        %% ----------------------------------------------------------------
        % 5. Contrast variation
        % -----------------------------------------------------------------

        case 5

            contrastFactor = 0.75 + 0.50*rand();

            meanValue = mean(img(:));

            img = (img - meanValue) * contrastFactor ...
                + meanValue;


        %% ----------------------------------------------------------------
        % 6. Gaussian noise
        % -----------------------------------------------------------------

        case 6

            noiseSigma = 0.015 + 0.025*rand();

            noise = noiseSigma * randn(size(img));

            img = img + noise;


        %% ----------------------------------------------------------------
        % 7. Mild blur
        % -----------------------------------------------------------------

        case 7

            sigma = 0.6 + 0.8*rand();

            img = imgaussfilt(img, sigma);


        %% ----------------------------------------------------------------
        % 8. Eye-region occlusion
        % -----------------------------------------------------------------

        case 8

            img = addEyeOcclusion(img);


        %% ----------------------------------------------------------------
        % 9. Mouth-region occlusion
        % -----------------------------------------------------------------

        case 9

            img = addMouthOcclusion(img);


        %% ----------------------------------------------------------------
        % 10. Random side occlusion
        % -----------------------------------------------------------------

        case 10

            if rand < 0.5
                img = addLeftOcclusion(img);
            else
                img = addRightOcclusion(img);
            end


        otherwise

            % Random mild augmentation
            angle = -5 + 10*rand();

            img = imrotate(img, angle, ...
                'bilinear', 'crop');

    end


    %% ====================================================================
    %  RANDOM SMALL TRANSLATION
    % ====================================================================

    if augIndex <= 7

        tx = randi([-4 4]);
        ty = randi([-4 4]);

        img = imtranslate(img, ...
            [tx ty], ...
            'FillValues', 0.5);

    end


    %% ====================================================================
    %  CLAMP PIXEL VALUES
    % ====================================================================

    img = min(max(img,0),1);


    %% ====================================================================
    %  CONVERT BACK TO UINT8
    % ====================================================================

    outputImage = im2uint8(img);

end


%% ========================================================================
%  EYE OCCLUSION
% ========================================================================

function img = addEyeOcclusion(img)

    [H,W,~] = size(img);

    % Approximate eye region
    y1 = round(0.28 * H);
    y2 = round(0.48 * H);

    x1 = round(0.18 * W);
    x2 = round(0.82 * W);

    % Random vertical position
    shift = randi([-round(0.03*H), round(0.03*H)]);

    y1 = max(1,y1+shift);
    y2 = min(H,y2+shift);

    % Dark semi-realistic rectangular occlusion
    occlusionHeight = max(3,round(0.10*H));

    centerY = round((y1+y2)/2);

    topY = max(1,centerY-round(occlusionHeight/2));
    bottomY = min(H,centerY+round(occlusionHeight/2));

    % Slight random width
    leftX = max(1,x1+randi([-round(0.03*W),0]));
    rightX = min(W,x2+randi([0,round(0.03*W)]));

    % Use dark neutral value with slight variation
    value = 0.05 + 0.15*rand();

    img(topY:bottomY,leftX:rightX,:) = value;

end


%% ========================================================================
%  MOUTH OCCLUSION
% ========================================================================

function img = addMouthOcclusion(img)

    [H,W,~] = size(img);

    % Approximate mouth/lower-face region
    centerY = round(0.72 * H);
    centerX = round(0.50 * W);

    occWidth = round((0.38 + 0.12*rand()) * W);
    occHeight = round((0.10 + 0.05*rand()) * H);

    x1 = centerX - round(occWidth/2);
    x2 = centerX + round(occWidth/2);

    y1 = centerY - round(occHeight/2);
    y2 = centerY + round(occHeight/2);

    x1 = max(1,x1);
    x2 = min(W,x2);
    y1 = max(1,y1);
    y2 = min(H,y2);

    value = 0.05 + 0.15*rand();

    img(y1:y2,x1:x2,:) = value;

end


%% ========================================================================
%  LEFT-SIDE OCCLUSION
% ========================================================================

function img = addLeftOcclusion(img)

    [H,W,~] = size(img);

    occWidth = round((0.18 + 0.12*rand()) * W);

    x1 = 1;
    x2 = occWidth;

    % Avoid completely hiding the eye region
    y1 = round(0.15*H);
    y2 = round(0.90*H);

    value = 0.05 + 0.15*rand();

    img(y1:y2,x1:x2,:) = value;

end


%% ========================================================================
%  RIGHT-SIDE OCCLUSION
% ========================================================================

function img = addRightOcclusion(img)

    [H,W,~] = size(img);

    occWidth = round((0.18 + 0.12*rand()) * W);

    x1 = W - occWidth + 1;
    x2 = W;

    % Avoid completely hiding the eye region
    y1 = round(0.15*H);
    y2 = round(0.90*H);

    value = 0.05 + 0.15*rand();

    img(y1:y2,x1:x2,:) = value;

end