classdef ArcFaceRecognition < handle
    % ================================================================
    % OCCLUSION-AWARE FACE RECOGNITION
    %
    % Model: InsightFace w600k_r50.onnx
    % MATLAB: R2026a
    %
    % Pipeline:
    % Image
    %   -> Face Detection
    %   -> Face Crop
    %   -> ArcFace Embedding
    %   -> Regional Feature Extraction
    %   -> Weighted Feature Fusion
    %   -> L2 Normalization
    %   -> Standardization
    %   -> PCA
    %   -> RBF SVM
    %   -> Identity
    %
    % predictDetailed() additionally returns the actual
    % RBF-SVM decision score for Person 2.
    % ================================================================

    properties
        Net
        FaceDetector

        InputSize = [112 112]
        EmbeddingSize = 512

        TrainMean
        TrainStd

        PCAMean
        PCACoeff
        PCAExplained
        NumPCAComponents

        Classifier
        ClassNames

        RegionWeights = [1.00 0.90 0.85 0.80 0.65 0.65]
    end

    methods

        %% ============================================================
        % Constructor
        % ============================================================

        function obj = ArcFaceRecognition(modelPath)

            if nargin < 1 || isempty(modelPath)

                root = fileparts(mfilename('fullpath'));

                modelPath = fullfile( ...
                    root, ...
                    'models', ...
                    'arcface.onnx');

            end

            if ~isfile(modelPath)

                error( ...
                    'ArcFace model not found:\n%s\n\n', ...
                    modelPath);

            end

            fprintf('\n');
            fprintf('================================================\n');
            fprintf('       ARCFACE INITIALIZATION\n');
            fprintf('================================================\n');
            fprintf('MATLAB R2026a\n');
            fprintf('Model: %s\n\n', modelPath);

            if exist('importNetworkFromONNX','file') ~= 2

                error([ ...
                    'importNetworkFromONNX is not available.\n\n' ...
                    'Install:\n' ...
                    'Deep Learning Toolbox Converter for ONNX Model Format\n']);

            end

            fprintf('Importing ONNX model...\n');

            try

                obj.Net = importNetworkFromONNX(modelPath);

            catch ME

                error([ ...
                    'Could not import ArcFace ONNX model.\n\n' ...
                    'MATLAB error:\n%s\n\n' ...
                    'Check that the file is the InsightFace ' ...
                    'w600k_r50.onnx model.'], ...
                    ME.message);

            end

            fprintf('ONNX model imported successfully.\n');

            obj.FaceDetector = vision.CascadeObjectDetector( ...
                'FrontalFaceCART');

            fprintf('Face detector initialized.\n');
            fprintf('Input size : 112 x 112 x 3\n');
            fprintf('Embedding  : 512-D\n');
            fprintf('================================================\n\n');

        end


        %% ============================================================
        % Detect face
        % ============================================================

        function [box,face] = detectFace(obj,img)

            if isempty(img)

                box = [];
                face = [];

                return;

            end

            if ndims(img) == 2

                img = repmat(img,[1 1 3]);

            end

            boxes = step(obj.FaceDetector,img);

            if isempty(boxes)

                box = [];
                face = [];

                return;

            end

            area = boxes(:,3).*boxes(:,4);

            [~,idx] = max(area);

            box = boxes(idx,:);

            face = imcrop(img,box);

            if isempty(face)

                box = [];
                face = [];

            end

        end


        %% ============================================================
        % ArcFace preprocessing
        % ============================================================

        function X = preprocessArcFace(obj,img)

            if ndims(img) == 2

                img = repmat(img,[1 1 3]);

            end

            img = imresize(img,obj.InputSize);

            img = single(img);

            % RGB -> BGR
            img = img(:,:,[3 2 1]);

            % Normalize to approximately -1 ... +1
            img = (img - 127.5) / 127.5;

            X = reshape( ...
                img, ...
                [112 112 3 1]);

        end


        %% ============================================================
        % Extract ArcFace embedding
        % ============================================================

        function embedding = extractEmbedding(obj,img)

            X = obj.preprocessArcFace(img);

            try

                output = predict(obj.Net,X);

            catch

                try

                    dlX = dlarray(X,'SSCB');

                    output = predict(obj.Net,dlX);

                catch ME

                    error([ ...
                        'ArcFace prediction failed.\n\n' ...
                        '%s'], ...
                        ME.message);

                end

            end

            if iscell(output)

                output = output{1};

            end

            if isa(output,'dlarray')

                output = extractdata(output);

            end

            output = gather(output);

            embedding = double(output);

            embedding = squeeze(embedding);

            embedding = embedding(:)';

            if numel(embedding) ~= 512

                warning( ...
                    'Expected 512-D embedding, but received %d-D.', ...
                    numel(embedding));

                obj.EmbeddingSize = numel(embedding);

            end

            n = norm(embedding);

            if n > 0

                embedding = embedding / n;

            end

        end


        %% ============================================================
        % Extract regional features
        % ============================================================

        function feature = extractRegionFeatures(obj,face)

            face = imresize(face,obj.InputSize);

            H = size(face,1);
            W = size(face,2);

            % Full face
            fullFace = face;

            e1 = obj.extractEmbedding(fullFace);

            % Upper face
            upperFace = face( ...
                1:round(0.58*H), ...
                :, :);

            e2 = obj.extractEmbedding(upperFace);

            % Middle face
            middleFace = face( ...
                round(0.25*H):round(0.78*H), ...
                round(0.10*W):round(0.90*W), :);

            e3 = obj.extractEmbedding(middleFace);

            % Lower face
            lowerFace = face( ...
                round(0.48*H):H, ...
                :, :);

            e4 = obj.extractEmbedding(lowerFace);

            % Left face
            leftFace = face( ...
                :, ...
                1:round(0.65*W), :);

            e5 = obj.extractEmbedding(leftFace);

            % Right face
            rightFace = face( ...
                :, ...
                round(0.35*W):W, :);

            e6 = obj.extractEmbedding(rightFace);

            % Weighted feature fusion
            feature = [ ...
                obj.RegionWeights(1)*e1, ...
                obj.RegionWeights(2)*e2, ...
                obj.RegionWeights(3)*e3, ...
                obj.RegionWeights(4)*e4, ...
                obj.RegionWeights(5)*e5, ...
                obj.RegionWeights(6)*e6];

            % Final L2 normalization
            n = norm(feature);

            if n > 0

                feature = feature / n;

            end

        end


        %% ============================================================
        % Build training dataset
        % ============================================================

        function [X,Y] = buildDataset(obj,folder)

            fprintf('\n');
            fprintf('================================================\n');
            fprintf('       BUILDING DATASET\n');
            fprintf('================================================\n');
            fprintf('%s\n\n',folder);

            imds = imageDatastore( ...
                folder, ...
                'IncludeSubfolders',true, ...
                'LabelSource','foldernames');

            total = numel(imds.Files);

            X = [];
            Y = categorical();

            count = 0;

            for i = 1:total

                file = imds.Files{i};

                fprintf( ...
                    '[%d/%d] %s\n', ...
                    i,total,file);

                try

                    img = imread(file);

                    [box,face] = obj.detectFace(img);

                    if isempty(box)

                        fprintf( ...
                            '    No face detected -> skipped\n');

                        continue;

                    end

                    feature = ...
                        obj.extractRegionFeatures(face);

                    count = count + 1;

                    X(count,:) = feature;

                    Y(count,1) = imds.Labels(i);

                    fprintf( ...
                        '    Face detected -> OK\n');

                catch ME

                    fprintf( ...
                        '    ERROR -> %s\n', ...
                        ME.message);

                end

            end

            fprintf('\n');

            fprintf( ...
                'Successfully processed: %d / %d\n', ...
                count,total);

            if count == 0

                error( ...
                    'No usable images were found in %s', ...
                    folder);

            end

        end


        %% ============================================================
        % Train classifier
        % ============================================================

        function train(obj,folder)

            [X,Y] = obj.buildDataset(folder);

            fprintf('\n');
            fprintf('================================================\n');
            fprintf('       TRAINING\n');
            fprintf('================================================\n');

            obj.ClassNames = categories(Y);

            fprintf('\nClasses:\n');

            for i = 1:numel(obj.ClassNames)

                fprintf( ...
                    '  %d -> %s\n', ...
                    i,obj.ClassNames{i});

            end

            % Standardization
            obj.TrainMean = mean(X,1);

            obj.TrainStd = std(X,0,1);

            obj.TrainStd(obj.TrainStd < 1e-8) = 1;

            Xstd = ...
                (X - obj.TrainMean) ./ obj.TrainStd;

            % PCA
            fprintf('\nRunning PCA...\n');

            [coeff,~,~,~,explained,mu] = pca(Xstd);

            cumulative = cumsum(explained);

            k = find( ...
                cumulative >= 98, ...
                1, ...
                'first');

            if isempty(k)

                k = size(coeff,2);

            end

            k = min( ...
                k, ...
                size(Xstd,1)-1);

            k = max(k,1);

            obj.PCAMean = mu;

            obj.PCACoeff = coeff(:,1:k);

            obj.PCAExplained = explained;

            obj.NumPCAComponents = k;

            XPCA = ...
                (Xstd - obj.PCAMean) * obj.PCACoeff;

            fprintf( ...
                'Original feature size : %d\n', ...
                size(X,2));

            fprintf( ...
                'PCA feature size      : %d\n', ...
                k);

            fprintf( ...
                'Variance retained     : %.2f %%\n', ...
                sum(explained(1:k)));

            % RBF SVM
            fprintf('\nTraining RBF SVM...\n');

            numClasses = numel(categories(Y));

            if numClasses == 2

                obj.Classifier = fitcsvm( ...
                    XPCA, ...
                    Y, ...
                    'KernelFunction','rbf', ...
                    'KernelScale','auto', ...
                    'BoxConstraint',1, ...
                    'Standardize',false);

            else

                template = templateSVM( ...
                    'KernelFunction','rbf', ...
                    'KernelScale','auto', ...
                    'BoxConstraint',1, ...
                    'Standardize',false);

                obj.Classifier = fitcecoc( ...
                    XPCA, ...
                    Y, ...
                    'Learners',template, ...
                    'Coding','onevsone');

            end

            fprintf('\nTraining completed.\n');

        end


        %% ============================================================
        % Original prediction method
        % ============================================================

        function [label,confidence,box] = predict(obj,img)

            [label,confidence,box,~] = ...
                obj.predictDetailed(img);

        end


        %% ============================================================
        % Detailed prediction
        %
        % positiveScore = actual SVM decision score for Person 2
        % ============================================================

        function [label,confidence,box,positiveScore] = ...
                predictDetailed(obj,img)

            positiveClass = "person2";

            % Face detection
            [box,face] = obj.detectFace(img);

            if isempty(box)

                label = categorical("Unknown");

                confidence = 0;

                positiveScore = NaN;

                return;

            end

            % Feature extraction
            feature = ...
                obj.extractRegionFeatures(face);

            % Standardization
            Xstd = ...
                (feature - obj.TrainMean) ./ obj.TrainStd;

            % PCA
            XPCA = ...
                (Xstd - obj.PCAMean) * obj.PCACoeff;

            % SVM prediction
            [label,scores] = ...
                predict(obj.Classifier,XPCA);

            scores = double(scores(:)');

            % Find actual Person 2 decision score
            positiveScore = NaN;

            if isa(obj.Classifier,'ClassificationSVM')

                svmClasses = ...
                    string(obj.Classifier.ClassNames);

                positiveIndex = find( ...
                    strcmpi(svmClasses,positiveClass), ...
                    1);

                if isempty(positiveIndex)

                    error([ ...
                        'Positive class "%s" was not found ' ...
                        'in the trained SVM classes.'], ...
                        positiveClass);

                end

                if numel(scores) ~= numel(svmClasses)

                    error([ ...
                        'Unexpected SVM score dimension. ' ...
                        'Expected %d scores, received %d.'], ...
                        numel(svmClasses), ...
                        numel(scores));

                end

                positiveScore = ...
                    scores(positiveIndex);

            elseif isa(obj.Classifier,'ClassificationECOC')

                classNames = ...
                    string(obj.Classifier.ClassNames);

                positiveIndex = find( ...
                    strcmpi(classNames,positiveClass), ...
                    1);

                if isempty(positiveIndex)

                    error([ ...
                        'Positive class "%s" was not found ' ...
                        'in ECOC classifier.'], ...
                        positiveClass);

                end

                positiveScore = ...
                    scores(positiveIndex);

            else

                error( ...
                    'Unsupported classifier type: %s', ...
                    class(obj.Classifier));

            end

            % Display confidence only.
            % NOT used for ROC/AUC.
            if numel(scores) == 2

                margin = ...
                    abs(scores(1)-scores(2));

            else

                sortedScores = ...
                    sort(scores,'descend');

                if numel(sortedScores) >= 2

                    margin = ...
                        sortedScores(1)-sortedScores(2);

                else

                    margin = ...
                        abs(sortedScores(1));

                end

            end

            confidence = ...
                100*(1-exp(-abs(margin)));

            confidence = ...
                min(max(confidence,0),100);

        end


        %% ============================================================
        % Save trained model
        % ============================================================

        function saveModel(obj,file)

            if nargin < 2

                root = ...
                    fileparts(mfilename('fullpath'));

                file = fullfile( ...
                    root, ...
                    'results', ...
                    'ArcFaceRecognitionModel.mat');

            end

            save(file,'obj','-v7.3');

            fprintf('\n');
            fprintf('Trained model saved:\n');
            fprintf('%s\n',file);

        end

    end
end