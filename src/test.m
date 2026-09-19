%% ========================================================================
%   OCCLUSION-AWARE FACE RECOGNITION
%   COMPLETE TESTING AND EVALUATION
%
%   IMPORTANT:
%   ROC/AUC is calculated using the ACTUAL RBF-SVM decision score.
%
%   Positive class:
%       person2
%
% ========================================================================

clc;
clear;
close all;


%% ========================================================================
% 1. PROJECT PATHS
% ========================================================================

projectFolder = ...
    'C:\228004030\occulationfacerecongnition';

modelFile = fullfile( ...
    projectFolder, ...
    'results', ...
    'ArcFaceRecognitionModel.mat');

testFolder = fullfile( ...
    projectFolder, ...
    'dataset', ...
    'test_augmented');

reportFolder = fullfile( ...
    projectFolder, ...
    'results', ...
    'report');


%% ========================================================================
% 2. CHECK PATHS
% ========================================================================

if ~isfolder(projectFolder)

    error( ...
        'Project folder not found:\n%s', ...
        projectFolder);

end


if ~isfile(modelFile)

    error([ ...
        'Trained model not found:\n%s\n\n' ...
        'Run train.m first.'], ...
        modelFile);

end


if ~isfolder(testFolder)

    error( ...
        'Test dataset folder not found:\n%s', ...
        testFolder);

end


if ~isfolder(reportFolder)

    mkdir(reportFolder);

end


%% ========================================================================
% 3. LOAD TRAINED MODEL
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('      OCCLUSION-AWARE FACE RECOGNITION - TESTING\n');
fprintf('============================================================\n\n');

fprintf('Loading trained recognition model...\n');

S = load(modelFile);


if isfield(S,'recognizer')

    recognizer = S.recognizer;

elseif isfield(S,'model')

    recognizer = S.model;

else

    names = fieldnames(S);

    if numel(names) == 1

        recognizer = S.(names{1});

    else

        error([ ...
            'Could not identify the trained recognizer inside:\n' ...
            '%s'], ...
            modelFile);

    end

end


fprintf('Model loaded successfully.\n');


%% ========================================================================
% 4. VERIFY CLASSIFIER
% ========================================================================

if isempty(recognizer.Classifier)

    error( ...
        'Loaded recognizer does not contain a trained classifier.');

end


fprintf('\nClassifier type : %s\n', ...
    class(recognizer.Classifier));

fprintf('Classes:\n');

disp(string(recognizer.Classifier.ClassNames));


%% ========================================================================
% 5. LOAD TEST DATASET
% ========================================================================

imdsTest = imageDatastore( ...
    testFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


numImages = ...
    numel(imdsTest.Files);


fprintf('\nTest dataset loaded.\n');

fprintf( ...
    'Total test images : %d\n\n', ...
    numImages);


if numImages == 0

    error('No test images were found.');

end


%% ========================================================================
% 6. INITIALIZE STORAGE
% ========================================================================

actualLabels = ...
    strings(numImages,1);

predictedLabels = ...
    strings(numImages,1);

confidenceValues = ...
    zeros(numImages,1);

% ACTUAL RBF-SVM decision score for Person 2
svmPositiveScores = ...
    NaN(numImages,1);


%% ========================================================================
% 7. TEST EVERY IMAGE
% ========================================================================

fprintf('============================================================\n');
fprintf('                     TESTING IMAGES\n');
fprintf('============================================================\n\n');


for i = 1:numImages

    %% ------------------------------------------------------------
    % Read image
    % ------------------------------------------------------------

    img = ...
        readimage(imdsTest,i);


    %% ------------------------------------------------------------
    % Actual label
    % ------------------------------------------------------------

    actualLabel = ...
        string(imdsTest.Labels(i));


    actualLabelNormalized = ...
        normalizeLabel(actualLabel);


    %% ------------------------------------------------------------
    % Detailed prediction
    % ------------------------------------------------------------

    try

        [predictedLabel, ...
         confidence, ...
         ~, ...
         positiveScore] = ...
            recognizer.predictDetailed(img);

    catch ME

        fprintf('\n');
        fprintf( ...
            'Error while processing image %d:\n', ...
            i);

        fprintf( ...
            '%s\n', ...
            ME.message);

        rethrow(ME);

    end


    %% ------------------------------------------------------------
    % Normalize predicted label
    % ------------------------------------------------------------

    predictedLabel = ...
        string(predictedLabel);


    predictedLabelNormalized = ...
        normalizeLabel(predictedLabel);


    %% ------------------------------------------------------------
    % Store
    % ------------------------------------------------------------

    actualLabels(i) = ...
        actualLabelNormalized;

    predictedLabels(i) = ...
        predictedLabelNormalized;

    confidenceValues(i) = ...
        double(confidence);

    svmPositiveScores(i) = ...
        double(positiveScore);


    %% ------------------------------------------------------------
    % Correct / Wrong
    % ------------------------------------------------------------

    isCorrect = strcmp( ...
        actualLabelNormalized, ...
        predictedLabelNormalized);


    if isCorrect

        resultText = 'CORRECT';

    else

        resultText = 'WRONG';

    end


    %% ------------------------------------------------------------
    % Print result
    % ------------------------------------------------------------

    fprintf( ...
        '%2d/%2d: Actual %-10s | Predicted %-10s | %s | Conf %.2f%% | SVM %.6f\n', ...
        i, ...
        numImages, ...
        actualLabelNormalized, ...
        predictedLabelNormalized, ...
        resultText, ...
        confidenceValues(i), ...
        svmPositiveScores(i));

end


%% ========================================================================
% 8. BASIC ACCURACY
% ========================================================================

correctMask = ...
    strcmp(actualLabels,predictedLabels);


numCorrect = ...
    sum(correctMask);


numWrong = ...
    numImages-numCorrect;


accuracy = ...
    numCorrect/numImages;


fprintf('\n');
fprintf('============================================================\n');
fprintf('                     FINAL RESULTS\n');
fprintf('============================================================\n');

fprintf( ...
    'Total test images : %d\n', ...
    numImages);

fprintf( ...
    'Correct           : %d\n', ...
    numCorrect);

fprintf( ...
    'Wrong             : %d\n', ...
    numWrong);

fprintf( ...
    'Accuracy          : %.2f%%\n', ...
    accuracy*100);


%% ========================================================================
% 9. CLASS-WISE ACCURACY
% ========================================================================

allClasses = ...
    unique(actualLabels,'stable');


numClasses = ...
    numel(allClasses);


classTotal = ...
    zeros(numClasses,1);


classCorrect = ...
    zeros(numClasses,1);


classAccuracy = ...
    zeros(numClasses,1);


for c = 1:numClasses

    classMask = ...
        strcmp(actualLabels,allClasses(c));


    classTotal(c) = ...
        sum(classMask);


    classCorrect(c) = ...
        sum(classMask & correctMask);


    if classTotal(c) > 0

        classAccuracy(c) = ...
            classCorrect(c)/classTotal(c);

    end

end


fprintf('\n');
fprintf('============================================================\n');
fprintf('                   CLASS-WISE ACCURACY\n');
fprintf('============================================================\n');


for c = 1:numClasses

    fprintf( ...
        '%-10s : %d / %d = %.2f%%\n', ...
        allClasses(c), ...
        classCorrect(c), ...
        classTotal(c), ...
        classAccuracy(c)*100);

end


%% ========================================================================
% 10. CONFUSION MATRIX
% ========================================================================

actualCategorical = ...
    categorical(actualLabels,allClasses);


predictedCategorical = ...
    categorical(predictedLabels,allClasses);


confMatrix = ...
    confusionmat( ...
        actualCategorical, ...
        predictedCategorical);


fprintf('\n');
fprintf('============================================================\n');
fprintf('                    CONFUSION MATRIX\n');
fprintf('============================================================\n');

disp(confMatrix);


%% ========================================================================
% 11. BINARY CLASSIFICATION METRICS
%
% Positive class = person2
%
%                    Predicted
%                 Person 1  Person 2
%
% Actual Person 1    TN        FP
%        Person 2    FN        TP
%
% ========================================================================

positiveClass = ...
    "person2";


actualPositive = ...
    strcmp(actualLabels,positiveClass);


predictedPositive = ...
    strcmp(predictedLabels,positiveClass);


TP = sum( ...
    actualPositive == 1 & ...
    predictedPositive == 1);


TN = sum( ...
    actualPositive == 0 & ...
    predictedPositive == 0);


FP = sum( ...
    actualPositive == 0 & ...
    predictedPositive == 1);


FN = sum( ...
    actualPositive == 1 & ...
    predictedPositive == 0);


%% ========================================================================
% 12. SENSITIVITY
% ========================================================================

if (TP+FN) > 0

    sensitivity = ...
        TP/(TP+FN);

else

    sensitivity = NaN;

end


%% ========================================================================
% 13. SPECIFICITY
% ========================================================================

if (TN+FP) > 0

    specificity = ...
        TN/(TN+FP);

else

    specificity = NaN;

end


%% ========================================================================
% 14. PRECISION
% ========================================================================

if (TP+FP) > 0

    precision = ...
        TP/(TP+FP);

else

    precision = NaN;

end


%% ========================================================================
% 15. F1 SCORE
% ========================================================================

if (precision+sensitivity) > 0

    F1 = ...
        2*(precision*sensitivity)/ ...
        (precision+sensitivity);

else

    F1 = NaN;

end


%% ========================================================================
% 16. PRINT CLASSIFICATION METRICS
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              CLASSIFICATION METRICS\n');
fprintf('============================================================\n');

fprintf( ...
    'Positive Class     : %s\n', ...
    positiveClass);

fprintf( ...
    'True Positive (TP) : %d\n', ...
    TP);

fprintf( ...
    'True Negative (TN) : %d\n', ...
    TN);

fprintf( ...
    'False Positive(FP) : %d\n', ...
    FP);

fprintf( ...
    'False Negative(FN) : %d\n', ...
    FN);

fprintf('\n');

fprintf( ...
    'Sensitivity        : %.2f%%\n', ...
    sensitivity*100);

fprintf( ...
    'Specificity        : %.2f%%\n', ...
    specificity*100);

fprintf( ...
    'Selectivity        : %.2f%%\n', ...
    specificity*100);

fprintf( ...
    'Precision          : %.2f%%\n', ...
    precision*100);

fprintf( ...
    'F1 Score           : %.2f%%\n', ...
    F1*100);


%% ========================================================================
% 17. PROPER ROC / AUC
%
% IMPORTANT:
%
% DO NOT use confidenceValues here.
%
% DO NOT construct:
%
%     +confidence for Person 2
%     -confidence for Person 1
%
% Instead use:
%
%     Actual RBF-SVM decision score
%
% ========================================================================

rocScores = ...
    svmPositiveScores;


rocLabels = ...
    actualPositive;


%% ========================================================================
% 18. VALID ROC DATA
% ========================================================================

validROC = ...
    isfinite(rocScores);


if sum(validROC) < 2

    warning( ...
        'Not enough valid SVM scores for ROC/AUC.');

    falsePositiveRate = [];
    truePositiveRate = [];
    thresholdValues = [];
    AUC = NaN;

else

    try

        [falsePositiveRate, ...
         truePositiveRate, ...
         thresholdValues, ...
         AUC] = perfcurve( ...
            rocLabels(validROC), ...
            rocScores(validROC), ...
            true);

    catch ME

        warning( ...
            'ROC/AUC calculation failed: %s', ...
            ME.message);

        falsePositiveRate = [];
        truePositiveRate = [];
        thresholdValues = [];
        AUC = NaN;

    end

end


%% ========================================================================
% 19. PRINT AUC
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    ROC ANALYSIS\n');
fprintf('============================================================\n');


if ~isnan(AUC)

    fprintf( ...
        'AUC                : %.4f\n', ...
        AUC);

    fprintf( ...
        'AUC Percentage     : %.2f%%\n', ...
        AUC*100);

else

    fprintf( ...
        'AUC                : Not available\n');

end


%% ========================================================================
% 20. GRAPH - OVERALL ACCURACY
% ========================================================================

figure( ...
    'Name','Overall Accuracy', ...
    'Color','w');


bar(accuracy*100);

ylim([0 100]);

ylabel('Accuracy (%)');

title('Overall Face Recognition Accuracy');

xticks(1);

xticklabels({'Overall'});

grid on;


text( ...
    1, ...
    accuracy*100+3, ...
    sprintf('%.2f%%',accuracy*100), ...
    'HorizontalAlignment','center', ...
    'FontWeight','bold');


overallAccuracyFile = ...
    fullfile( ...
        reportFolder, ...
        'overall_accuracy.png');


exportgraphics( ...
    gcf, ...
    overallAccuracyFile, ...
    'Resolution',300);


%% ========================================================================
% 21. GRAPH - CLASS-WISE ACCURACY
% ========================================================================

figure( ...
    'Name','Class-wise Accuracy', ...
    'Color','w');


bar(classAccuracy*100);

ylim([0 100]);

ylabel('Accuracy (%)');

title('Class-wise Face Recognition Accuracy');

xticks(1:numClasses);

xticklabels(allClasses);

grid on;


for c = 1:numClasses

    text( ...
        c, ...
        classAccuracy(c)*100+3, ...
        sprintf('%.2f%%',classAccuracy(c)*100), ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold');

end


classAccuracyFile = ...
    fullfile( ...
        reportFolder, ...
        'classwise_accuracy.png');


exportgraphics( ...
    gcf, ...
    classAccuracyFile, ...
    'Resolution',300);


%% ========================================================================
% 22. GRAPH - CORRECT VS WRONG
% ========================================================================

figure( ...
    'Name','Correct vs Wrong', ...
    'Color','w');


bar([numCorrect numWrong]);

ylabel('Number of Images');

title('Correct vs Wrong Predictions');

xticks([1 2]);

xticklabels({'Correct','Wrong'});

grid on;


text( ...
    1, ...
    numCorrect+0.3, ...
    sprintf('%d',numCorrect), ...
    'HorizontalAlignment','center', ...
    'FontWeight','bold');


text( ...
    2, ...
    numWrong+0.3, ...
    sprintf('%d',numWrong), ...
    'HorizontalAlignment','center', ...
    'FontWeight','bold');


correctWrongFile = ...
    fullfile( ...
        reportFolder, ...
        'correct_vs_wrong.png');


exportgraphics( ...
    gcf, ...
    correctWrongFile, ...
    'Resolution',300);


%% ========================================================================
% 23. GRAPH - CONFUSION MATRIX
% ========================================================================

figure( ...
    'Name','Confusion Matrix', ...
    'Color','w');


cm = confusionchart( ...
    actualCategorical, ...
    predictedCategorical);


cm.Title = ...
    'Face Recognition Confusion Matrix';


cm.RowSummary = ...
    'row-normalized';


cm.ColumnSummary = ...
    'column-normalized';


confusionMatrixFile = ...
    fullfile( ...
        reportFolder, ...
        'confusion_matrix.png');


exportgraphics( ...
    gcf, ...
    confusionMatrixFile, ...
    'Resolution',300);


%% ========================================================================
% 24. GRAPH - ROC CURVE
% ========================================================================

if ~isnan(AUC)

    figure( ...
        'Name','ROC Curve', ...
        'Color','w');


    plot( ...
        falsePositiveRate, ...
        truePositiveRate, ...
        'LineWidth',2);


    hold on;


    plot( ...
        [0 1], ...
        [0 1], ...
        '--', ...
        'LineWidth',1.2);


    xlabel('False Positive Rate');

    ylabel('True Positive Rate');


    title( ...
        sprintf( ...
            'ROC Curve - AUC = %.4f', ...
            AUC));


    legend( ...
        'ROC Curve', ...
        'Random Classifier', ...
        'Location','southeast');


    grid on;

    axis square;

    xlim([0 1]);

    ylim([0 1]);


    rocFile = ...
        fullfile( ...
            reportFolder, ...
            'roc_auc_curve.png');


    exportgraphics( ...
        gcf, ...
        rocFile, ...
        'Resolution',300);

else

    rocFile = '';

end


%% ========================================================================
% 25. GRAPH - PERFORMANCE METRICS
% ========================================================================

metricValues = [ ...
    accuracy*100, ...
    sensitivity*100, ...
    specificity*100, ...
    precision*100, ...
    F1*100];


metricNames = { ...
    'Accuracy', ...
    'Sensitivity', ...
    'Specificity', ...
    'Precision', ...
    'F1 Score'};


figure( ...
    'Name','Performance Metrics', ...
    'Color','w');


bar(metricValues);

ylim([0 100]);

ylabel('Percentage (%)');

title('Classification Performance Metrics');

xticks(1:numel(metricNames));

xticklabels(metricNames);

grid on;


for k = 1:numel(metricValues)

    text( ...
        k, ...
        metricValues(k)+3, ...
        sprintf('%.2f%%',metricValues(k)), ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold');

end


metricsGraphFile = ...
    fullfile( ...
        reportFolder, ...
        'performance_metrics.png');


exportgraphics( ...
    gcf, ...
    metricsGraphFile, ...
    'Resolution',300);


%% ========================================================================
% 26. SAVE RECOGNITION CSV
% ========================================================================

imageNumbers = ...
    (1:numImages)';


correctStatus = ...
    strings(numImages,1);


for i = 1:numImages

    if correctMask(i)

        correctStatus(i) = "Correct";

    else

        correctStatus(i) = "Wrong";

    end

end


reportTable = table( ...
    imageNumbers, ...
    actualLabels, ...
    predictedLabels, ...
    confidenceValues, ...
    svmPositiveScores, ...
    correctStatus, ...
    'VariableNames',{ ...
        'ImageNumber', ...
        'ActualLabel', ...
        'PredictedLabel', ...
        'ConfidencePercent', ...
        'Person2_SVM_DecisionScore', ...
        'Result'});


csvFile = ...
    fullfile( ...
        reportFolder, ...
        'recognition_report.csv');


writetable( ...
    reportTable, ...
    csvFile);


%% ========================================================================
% 27. SAVE CLASS-WISE CSV
% ========================================================================

classTable = table( ...
    allClasses, ...
    classCorrect, ...
    classTotal, ...
    classAccuracy*100, ...
    'VariableNames',{ ...
        'Class', ...
        'Correct', ...
        'Total', ...
        'AccuracyPercent'});


classCSVFile = ...
    fullfile( ...
        reportFolder, ...
        'classwise_accuracy.csv');


writetable( ...
    classTable, ...
    classCSVFile);


%% ========================================================================
% 28. SAVE METRICS TEXT FILE
% ========================================================================

metricsFile = ...
    fullfile( ...
        reportFolder, ...
        'classification_metrics.txt');


fid = fopen(metricsFile,'w');


if fid == -1

    warning( ...
        'Could not create metrics text file.');

else

    fprintf( ...
        fid, ...
        'OCCLUSION-AWARE FACE RECOGNITION\n');

    fprintf( ...
        fid, ...
        '============================================\n\n');


    fprintf( ...
        fid, ...
        'TEST DATASET\n');

    fprintf( ...
        fid, ...
        '------------\n');

    fprintf( ...
        fid, ...
        'Total Test Images : %d\n', ...
        numImages);

    fprintf( ...
        fid, ...
        'Correct           : %d\n', ...
        numCorrect);

    fprintf( ...
        fid, ...
        'Wrong             : %d\n', ...
        numWrong);

    fprintf( ...
        fid, ...
        'Accuracy          : %.2f%%\n\n', ...
        accuracy*100);


    fprintf( ...
        fid, ...
        'BINARY CLASSIFICATION\n');

    fprintf( ...
        fid, ...
        '---------------------\n');

    fprintf( ...
        fid, ...
        'Positive Class    : %s\n', ...
        positiveClass);

    fprintf( ...
        fid, ...
        'True Positive     : %d\n', ...
        TP);

    fprintf( ...
        fid, ...
        'True Negative     : %d\n', ...
        TN);

    fprintf( ...
        fid, ...
        'False Positive    : %d\n', ...
        FP);

    fprintf( ...
        fid, ...
        'False Negative    : %d\n\n', ...
        FN);


    fprintf( ...
        fid, ...
        'Sensitivity       : %.2f%%\n', ...
        sensitivity*100);

    fprintf( ...
        fid, ...
        'Specificity       : %.2f%%\n', ...
        specificity*100);

    fprintf( ...
        fid, ...
        'Selectivity       : %.2f%%\n', ...
        specificity*100);

    fprintf( ...
        fid, ...
        'Precision         : %.2f%%\n', ...
        precision*100);

    fprintf( ...
        fid, ...
        'F1 Score          : %.2f%%\n', ...
        F1*100);


    if ~isnan(AUC)

        fprintf( ...
            fid, ...
            'AUC               : %.4f\n', ...
            AUC);

    else

        fprintf( ...
            fid, ...
            'AUC               : Not available\n');

    end


    fprintf( ...
        fid, ...
        '\nROC SCORE\n');

    fprintf( ...
        fid, ...
        '---------\n');

    fprintf( ...
        fid, ...
        'ROC score source   : Actual RBF-SVM decision score\n');

    fprintf( ...
        fid, ...
        'Positive class     : person2\n');

    fprintf( ...
        fid, ...
        'Confidence used    : NO\n');

    fprintf( ...
        fid, ...
        'Artificial +/- confidence : NO\n');


    fprintf( ...
        fid, ...
        '\nCLASS-WISE ACCURACY\n');

    fprintf( ...
        fid, ...
        '-------------------\n');


    for c = 1:numClasses

        fprintf( ...
            fid, ...
            '%s : %d / %d = %.2f%%\n', ...
            allClasses(c), ...
            classCorrect(c), ...
            classTotal(c), ...
            classAccuracy(c)*100);

    end


    fclose(fid);

end


%% ========================================================================
% 29. SAVE ROC DATA
% ========================================================================

if ~isnan(AUC)

    rocTable = table( ...
        falsePositiveRate, ...
        truePositiveRate, ...
        thresholdValues, ...
        'VariableNames',{ ...
            'FalsePositiveRate', ...
            'TruePositiveRate', ...
            'Threshold'});


    rocCSVFile = ...
        fullfile( ...
            reportFolder, ...
            'roc_data.csv');


    writetable( ...
        rocTable, ...
        rocCSVFile);

end


%% ========================================================================
% 30. FINAL PERFORMANCE
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  FINAL PERFORMANCE\n');
fprintf('============================================================\n');


fprintf( ...
    'Accuracy          : %.2f%%\n', ...
    accuracy*100);


fprintf( ...
    'Sensitivity       : %.2f%%\n', ...
    sensitivity*100);


fprintf( ...
    'Specificity       : %.2f%%\n', ...
    specificity*100);


fprintf( ...
    'Selectivity       : %.2f%%\n', ...
    specificity*100);


fprintf( ...
    'Precision         : %.2f%%\n', ...
    precision*100);


fprintf( ...
    'F1 Score          : %.2f%%\n', ...
    F1*100);


if ~isnan(AUC)

    fprintf( ...
        'AUC               : %.4f\n', ...
        AUC);

end


%% ========================================================================
% 31. REPORT FILES
% ========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    REPORT FILES\n');
fprintf('============================================================\n');


fprintf( ...
    'Report folder:\n%s\n\n', ...
    reportFolder);


fprintf( ...
    '1. Overall Accuracy:\n   %s\n\n', ...
    overallAccuracyFile);


fprintf( ...
    '2. Class-wise Accuracy:\n   %s\n\n', ...
    classAccuracyFile);


fprintf( ...
    '3. Correct vs Wrong:\n   %s\n\n', ...
    correctWrongFile);


fprintf( ...
    '4. Confusion Matrix:\n   %s\n\n', ...
    confusionMatrixFile);


if ~isempty(rocFile)

    fprintf( ...
        '5. ROC-AUC Curve:\n   %s\n\n', ...
        rocFile);

end


fprintf( ...
    '6. Performance Metrics:\n   %s\n\n', ...
    metricsGraphFile);


fprintf( ...
    '7. Recognition CSV:\n   %s\n\n', ...
    csvFile);


fprintf( ...
    '8. Class-wise CSV:\n   %s\n\n', ...
    classCSVFile);


fprintf( ...
    '9. Classification Metrics:\n   %s\n\n', ...
    metricsFile);


if ~isnan(AUC)

    fprintf( ...
        '10. ROC Data:\n   %s\n\n', ...
        rocCSVFile);

end


%% ========================================================================
% 32. FINAL MESSAGE
% ========================================================================

fprintf('============================================================\n');
fprintf('                  TESTING COMPLETED\n');
fprintf('============================================================\n');


fprintf('\nYour model has been evaluated successfully.\n');


fprintf('\nPoster values:\n');


fprintf( ...
    'Accuracy     = %.2f%%\n', ...
    accuracy*100);


fprintf( ...
    'Sensitivity  = %.2f%%\n', ...
    sensitivity*100);


fprintf( ...
    'Specificity  = %.2f%%\n', ...
    specificity*100);


fprintf( ...
    'Selectivity  = %.2f%%\n', ...
    specificity*100);


fprintf( ...
    'Precision    = %.2f%%\n', ...
    precision*100);


fprintf( ...
    'F1 Score     = %.2f%%\n', ...
    F1*100);


if ~isnan(AUC)

    fprintf( ...
        'AUC          = %.4f\n', ...
        AUC);

end


fprintf('\n');


%% ========================================================================
% LOCAL FUNCTION
% ========================================================================

function label = normalizeLabel(label)

    label = string(label);

    label = replace(label," ","");

    label = replace(label,"_","");

    label = replace(label,"-","");

    label = lower(label);

end