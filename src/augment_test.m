%% ================================================================
%  augment_test.m
%
%  TEST DATA AUGMENTATION FOR OCCLUSION ROBUSTNESS EVALUATION
%
%  Original test dataset is NOT modified.
%
%  Input:
%      dataset\test\person 1
%      dataset\test\person 2
%
%  Output:
%      dataset\test_augmented\person 1
%      dataset\test_augmented\person 2
%
%  Each original image:
%      1 original + 10 augmented images
%
%  Augmentations:
%      1. Flip + rotation
%      2. Positive rotation
%      3. Negative rotation
%      4. Brightness variation
%      5. Contrast variation
%      6. Gaussian noise
%      7. Gaussian blur
%      8. Eye-region occlusion
%      9. Mouth-region occlusion
%     10. Random side occlusion
%
%  MATLAB R2026a compatible
%% ================================================================

clc;
clear;
close all;

%% ------------------------------------------------
%  PROJECT PATH
%% ------------------------------------------------

projectPath = 'C:\228004030\occulationfacerecongnition';

datasetPath = fullfile(projectPath, 'dataset');

testPath = fullfile(datasetPath, 'test');

outputPath = fullfile(datasetPath, 'test_augmented');

%% ------------------------------------------------
%  RANDOM SEED
%
%  Fixed seed makes the generated dataset
%  reproducible.
%% ------------------------------------------------

rng(42);

%% ------------------------------------------------
%  CHECK INPUT DIRECTORY
%% ------------------------------------------------

if ~isfolder(testPath)

    error('Test folder not found:\n%s', testPath);

end

%% ------------------------------------------------
%  CREATE OUTPUT DIRECTORY
%% ------------------------------------------------

if isfolder(outputPath)

    fprintf('\nExisting test_augmented folder found.\n');
    fprintf('It will be replaced.\n\n');

    rmdir(outputPath, 's');

end

mkdir(outputPath);

%% ------------------------------------------------
%  PERSON FOLDERS
%% ------------------------------------------------

classes = {'person 1', 'person 2'};

%% ------------------------------------------------
%  IMAGE EXTENSIONS
%% ------------------------------------------------

extensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};

%% ------------------------------------------------
%  COUNTERS
%% ------------------------------------------------

totalOriginal = 0;
totalGenerated = 0;

%% ================================================================
%  PROCESS EACH PERSON
%% ================================================================

for c = 1:numel(classes)

    className = classes{c};

    inputClassPath = fullfile(testPath, className);
    outputClassPath = fullfile(outputPath, className);

    %% Check class folder

    if ~isfolder(inputClassPath)

        warning('Folder not found: %s', inputClassPath);
        continue;

    end

    %% Create output class folder

    mkdir(outputClassPath);

    %% ------------------------------------------------------------
    %  COLLECT IMAGES
    %% ------------------------------------------------------------

    files = [];

    for e = 1:numel(extensions)

        tempFiles = dir(fullfile(inputClassPath, extensions{e}));

        files = [files; tempFiles]; %#ok<AGROW>

    end

    if isempty(files)

        warning('No images found in %s', inputClassPath);
        continue;

    end

    fprintf('\n============================================\n');
    fprintf('Processing: %s\n', className);
    fprintf('Original images: %d\n', numel(files));
    fprintf('============================================\n');

    %% ------------------------------------------------------------
    %  PROCESS EACH IMAGE
    %% ------------------------------------------------------------

    for i = 1:numel(files)

        inputFile = fullfile(inputClassPath, files(i).name);

        %% Read image

        try

            I = imread(inputFile);

        catch ME

            warning('Could not read %s\n%s', inputFile, ME.message);
            continue;

        end

        %% Convert grayscale to RGB

        if ndims(I) == 2

            I = cat(3, I, I, I);

        end

        %% Remove alpha channel if present

        if size(I,3) > 3

            I = I(:,:,1:3);

        end

        %% Convert to uint8

        if ~isa(I,'uint8')

            I = im2uint8(I);

        end

        %% Base filename

        [~, baseName, ~] = fileparts(files(i).name);

        %% --------------------------------------------------------
        %  SAVE ORIGINAL COPY
        %% --------------------------------------------------------

        originalName = sprintf('%s_original.jpg', baseName);

        imwrite(I, ...
            fullfile(outputClassPath, originalName), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 1
        %  HORIZONTAL FLIP + SMALL ROTATION
        %% ========================================================

        A1 = fliplr(I);

        angle = -8 + 16*rand;

        A1 = imrotate(A1, angle, 'bilinear', 'crop');

        filename = sprintf('%s_aug01_flip_rotate.jpg', baseName);

        imwrite(A1, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 2
        %  POSITIVE ROTATION
        %% ========================================================

        angle = 4 + 4*rand;

        A2 = imrotate(I, angle, 'bilinear', 'crop');

        filename = sprintf('%s_aug02_rotate_positive.jpg', baseName);

        imwrite(A2, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 3
        %  NEGATIVE ROTATION
        %% ========================================================

        angle = -(4 + 4*rand);

        A3 = imrotate(I, angle, 'bilinear', 'crop');

        filename = sprintf('%s_aug03_rotate_negative.jpg', baseName);

        imwrite(A3, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 4
        %  BRIGHTNESS VARIATION
        %% ========================================================

        A4 = im2double(I);

        brightnessFactor = 0.75 + 0.50*rand;

        A4 = A4 * brightnessFactor;

        A4 = min(max(A4,0),1);

        A4 = im2uint8(A4);

        filename = sprintf('%s_aug04_brightness.jpg', baseName);

        imwrite(A4, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 5
        %  CONTRAST VARIATION
        %% ========================================================

        A5 = im2double(I);

        contrastFactor = 0.75 + 0.50*rand;

        A5 = (A5 - 0.5) * contrastFactor + 0.5;

        A5 = min(max(A5,0),1);

        A5 = im2uint8(A5);

        filename = sprintf('%s_aug05_contrast.jpg', baseName);

        imwrite(A5, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 6
        %  GAUSSIAN NOISE
        %% ========================================================

        A6 = im2double(I);

        noiseSigma = 0.015 + 0.025*rand;

        A6 = imnoise(A6, 'gaussian', 0, noiseSigma^2);

        A6 = im2uint8(A6);

        filename = sprintf('%s_aug06_noise.jpg', baseName);

        imwrite(A6, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 7
        %  MILD GAUSSIAN BLUR
        %% ========================================================

        sigma = 0.7 + 0.8*rand;

        kernelSize = 2*ceil(2*sigma)+1;

        A7 = imgaussfilt(I, sigma, ...
            'FilterSize', kernelSize);

        filename = sprintf('%s_aug07_blur.jpg', baseName);

        imwrite(A7, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 8
        %  EYE REGION OCCLUSION
        %
        %  Dark semi-realistic rectangular obstruction
        %  across the upper-middle face.
        %% ========================================================

        A8 = I;

        [H,W,~] = size(A8);

        x1 = round(0.12*W);
        x2 = round(0.88*W);

        y1 = round(0.28*H);
        y2 = round(0.46*H);

        % Slight random movement

        shiftX = randi([-round(0.04*W), round(0.04*W)]);
        shiftY = randi([-round(0.025*H), round(0.025*H)]);

        x1 = max(1,min(W,x1+shiftX));
        x2 = max(1,min(W,x2+shiftX));

        y1 = max(1,min(H,y1+shiftY));
        y2 = max(1,min(H,y2+shiftY));

        % Create occlusion mask

        mask = false(H,W);

        mask(y1:y2,x1:x2) = true;

        % Add slight feathering

        mask = imgaussfilt(double(mask),2);

        mask = mask > 0.25;

        % Occlusion intensity

        occlusionValue = uint8(45 + 30*rand);

        for ch = 1:3

            channel = A8(:,:,ch);

            channel(mask) = occlusionValue;

            A8(:,:,ch) = channel;

        end

        filename = sprintf('%s_aug08_eye_occlusion.jpg', baseName);

        imwrite(A8, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 9
        %  MOUTH REGION OCCLUSION
        %% ========================================================

        A9 = I;

        [H,W,~] = size(A9);

        x1 = round(0.15*W);
        x2 = round(0.85*W);

        y1 = round(0.62*H);
        y2 = round(0.80*H);

        shiftX = randi([-round(0.04*W), round(0.04*W)]);
        shiftY = randi([-round(0.025*H), round(0.025*H)]);

        x1 = max(1,min(W,x1+shiftX));
        x2 = max(1,min(W,x2+shiftX));

        y1 = max(1,min(H,y1+shiftY));
        y2 = max(1,min(H,y2+shiftY));

        mask = false(H,W);

        mask(y1:y2,x1:x2) = true;

        mask = imgaussfilt(double(mask),2);

        mask = mask > 0.25;

        occlusionValue = uint8(50 + 35*rand);

        for ch = 1:3

            channel = A9(:,:,ch);

            channel(mask) = occlusionValue;

            A9(:,:,ch) = channel;

        end

        filename = sprintf('%s_aug09_mouth_occlusion.jpg', baseName);

        imwrite(A9, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% ========================================================
        %  AUGMENTATION 10
        %  RANDOM SIDE OCCLUSION
        %% ========================================================

        A10 = I;

        [H,W,~] = size(A10);

        side = randi([1 2]);

        occlusionWidth = round((0.20 + 0.10*rand)*W);

        y1 = round((0.15 + 0.10*rand)*H);

        y2 = round((0.75 + 0.10*rand)*H);

        y1 = max(1,min(H,y1));
        y2 = max(1,min(H,y2));

        if side == 1

            % Left-side occlusion

            x1 = 1;
            x2 = occlusionWidth;

        else

            % Right-side occlusion

            x1 = W - occlusionWidth + 1;
            x2 = W;

        end

        x1 = max(1,min(W,x1));
        x2 = max(1,min(W,x2));

        mask = false(H,W);

        mask(y1:y2,x1:x2) = true;

        mask = imgaussfilt(double(mask),2);

        mask = mask > 0.25;

        occlusionValue = uint8(45 + 40*rand);

        for ch = 1:3

            channel = A10(:,:,ch);

            channel(mask) = occlusionValue;

            A10(:,:,ch) = channel;

        end

        if side == 1

            sideName = 'left';

        else

            sideName = 'right';

        end

        filename = sprintf( ...
            '%s_aug10_%s_occlusion.jpg', ...
            baseName, sideName);

        imwrite(A10, ...
            fullfile(outputClassPath, filename), ...
            'Quality', 95);

        totalGenerated = totalGenerated + 1;

        %% --------------------------------------------------------
        %  PROGRESS
        %% --------------------------------------------------------

        totalOriginal = totalOriginal + 1;

        fprintf( ...
            '%s: %d/%d processed\n', ...
            className, i, numel(files));

    end

end

%% ================================================================
%  FINAL SUMMARY
%% ================================================================

fprintf('\n\n');
fprintf('============================================================\n');
fprintf('       TEST DATA AUGMENTATION COMPLETED\n');
fprintf('============================================================\n');

fprintf('Original images processed : %d\n', totalOriginal);
fprintf('Generated images          : %d\n', totalGenerated);

if totalOriginal > 0

    fprintf('Images per original       : %d\n', ...
        round(totalGenerated/totalOriginal));

end

fprintf('\nOutput folder:\n%s\n', outputPath);

fprintf('\nOriginal test dataset was NOT modified.\n');

fprintf('\nDataset structure:\n');
fprintf('dataset\\test\\\n');
fprintf('dataset\\test_augmented\\\n');

fprintf('\n============================================================\n');
fprintf('IMPORTANT:\n');
fprintf('Use dataset\\test for your primary result.\n');
fprintf('Use dataset\\test_augmented for occlusion robustness testing.\n');
fprintf('Do NOT combine both sets into one accuracy calculation.\n');
fprintf('============================================================\n');