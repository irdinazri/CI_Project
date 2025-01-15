function createHeartRiskPredictionGUI()
    % Create MATLAB GUI for ANFIS Heart Disease Prediction

    % Load data
    dataFile = 'heart_attack_prediction_dataset_Modified.xlsx';
    opts = detectImportOptions(dataFile);
    opts.VariableNamingRule = 'preserve';
    data = readtable(dataFile, opts);

    % Preprocess data (map values, one-hot encoding, etc.)
    [input, output] = preprocessData(data);

    % GUI components
    fig = uifigure('Name', 'Heart Disease Prediction', 'Position', [100, 100, 700, 500]);

    % Train Button
    trainButton = uibutton(fig, 'Position', [50, 450, 100, 30], 'Text', 'Train ANFIS', ...
                           'ButtonPushedFcn', @(btn, event) trainANFISCallback(input, output));

    % Axes for displaying results
    ax = uiaxes(fig, 'Position', [50, 50, 300, 300]);
    title(ax, 'Prediction Results');

    % Confusion Matrix Display Area
    confusionMatrixPanel = uipanel(fig, 'Position', [400, 50, 250, 250], 'Title', 'Confusion Matrix');

    % Status Label
    statusLabel = uilabel(fig, 'Position', [400, 450, 200, 30], 'Text', 'Status: Ready');

    % Callback functions
    function trainANFISCallback(input, output)
    try
        % Split data into train and test
        [XTrain, YTrain, XTest, YTest] = splitData(input, output);

        % Train ANFIS
        fis = anfis([XTrain YTrain], genfis1([XTrain YTrain]));

        % Save model
        assignin('base', 'fisModel', fis);
        statusLabel.Text = 'Status: Model Trained';

        % Evaluate on test data
        predictions = evalfis(fis, XTest);

        % Binarize predictions
        predictionsBinary = predictions >= 0.5;
        YTestBinary = YTest >= 0.5;

        % Calculate confusion matrix
        confMat = confusionmat(YTestBinary, predictionsBinary);

        % Display confusion matrix as a plot
        axesConfMat = uiaxes(confusionMatrixPanel, 'Position', [10, 10, 230, 200]);
        imagesc(axesConfMat, confMat);
        colormap(axesConfMat, 'cool');
        colorbar(axesConfMat);

        % Add annotations
        [rows, cols] = size(confMat);
        for i = 1:rows
            for j = 1:cols
                text(axesConfMat, j, i, num2str(confMat(i, j)), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', ...
                     'FontSize', 12, 'Color', 'black');
            end
        end

        % Set axes labels
        axesConfMat.XTick = 1:cols;
        axesConfMat.YTick = 1:rows;
        axesConfMat.XTickLabels = {'Predicted 0', 'Predicted 1'};
        axesConfMat.YTickLabels = {'Actual 0', 'Actual 1'};
        xlabel(axesConfMat, 'Predicted Class');
        ylabel(axesConfMat, 'True Class');
        title(axesConfMat, 'Confusion Matrix');

        % Plot training results
        scatter(ax, YTest, predictions);
        xlabel(ax, 'True Values'); ylabel(ax, 'Predicted Values');

        % Calculate and display metrics
        mse = mean((YTest - predictions).^2);
        statusLabel.Text = sprintf('Status: Training Completed (MSE: %.2f)', mse);
    catch ME
        disp(ME.message);
        statusLabel.Text = 'Status: Training Failed';
    end
    end
end

function [input, output] = preprocessData(data)
    % Extract features
    Age = data{:, 1};
    Gender = dummyvar(categorical(data{:, 2}));
    Cholesterol = data{:, 3};
    BloodPressure = calculateMAP(data{:, 4});
    HeartRate = data{:, 5}; % Include Heart Rate

    % Combine features into input matrix
    input = [Age, Gender, Cholesterol, BloodPressure, HeartRate];
    output = data{:, 25}; % Target variable (Heart Attack Risk)
end

function mapValues = calculateMAP(bpValues)
    % Process blood pressure data
    mapValues = zeros(length(bpValues), 1);
    for i = 1:length(bpValues)
        bpSplit = strsplit(bpValues{i}, '/');
        systolic = str2double(bpSplit{1});
        diastolic = str2double(bpSplit{2});
        mapValues(i) = (2 * diastolic + systolic) / 3;
    end
end

function [XTrain, YTrain, XTest, YTest] = splitData(input, output)
    % Split data into train and test sets
    cv = cvpartition(size(input, 1), 'HoldOut', 0.3);
    idx = cv.test;
    XTrain = input(~idx, :);
    YTrain = output(~idx);
    XTest = input(idx, :);
    YTest = output(idx);
end
