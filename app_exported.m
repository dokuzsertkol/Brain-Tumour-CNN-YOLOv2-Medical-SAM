classdef app_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        PredictionPer                  matlab.ui.control.Label
        Prediction                     matlab.ui.control.Label
        ClassifyButton                 matlab.ui.control.Button
        SelectedNameLabel              matlab.ui.control.Label
        SelectImageButton              matlab.ui.control.Button
        ModelDropDown                  matlab.ui.control.DropDown
        ModelDropDownLabel             matlab.ui.control.Label
        PREDICTLabel                   matlab.ui.control.Label
        TRAINLabel                     matlab.ui.control.Label
        Image                          matlab.ui.control.Image
        EpochCountEditField            matlab.ui.control.NumericEditField
        EpochCountEditFieldLabel       matlab.ui.control.Label
        LearningRateEditField          matlab.ui.control.NumericEditField
        LearningRateEditFieldLabel     matlab.ui.control.Label
        BatchSizeDropDown              matlab.ui.control.DropDown
        BatchSizeDropDownLabel         matlab.ui.control.Label
        OptimizationAlgorithmDropDown  matlab.ui.control.DropDown
        OptimizationAlgorithmLabel     matlab.ui.control.Label
        ModelTypeDropDown              matlab.ui.control.DropDown
        ModelTypeDropDownLabel         matlab.ui.control.Label
        TrainButton                    matlab.ui.control.Button
    end

    
    properties (Access = private)
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Get current folder and parent folder
            currentFolder = fileparts(mfilename('fullpath'));
        
            % Define prefixes
            prefixes = {'googlenet_', 'mycnn_', 'alexnet_', 'resnet50_'};
        
            % Initialize an empty array to hold file info
            allFiles = [];
        
            % Loop through each prefix and collect matching .mat files
            for i = 1:length(prefixes)
                files = dir(fullfile(currentFolder, [prefixes{i} '*.mat']));
                allFiles = [allFiles; files];
            end
        
            % Extract file names and remove extensions
            modelNames = {allFiles.name};
            cleanNames = erase(modelNames, '.mat');
        
            % Assign to dropdown once
            app.ModelDropDown.Items = cleanNames;
        end

        % Button pushed function: TrainButton
        function Train(app, event)
            app.TrainButton.Enable = false;
            
            % input variables
            i_model = lower(app.ModelTypeDropDown.Value);
            i_opt_alg = lower(app.OptimizationAlgorithmDropDown.Value);
            i_batch = str2double(app.BatchSizeDropDown.Value);
            i_lr = app.LearningRateEditField.Value;
            i_epoch = app.EpochCountEditField.Value;
            
            % dataset
            currentFolder = fileparts(mfilename('fullpath'));
            rootFolder = fullfile(currentFolder, 'brain_tumor_dataset');

            imds = imageDatastore(fullfile(rootFolder, {'no', 'yes'}), ...
                'LabelSource', 'foldernames', ...
                'IncludeSubfolders', true);
            

            % %70 of dataset will be used for training, %15 for validation,
            % %15 for testing
            [imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, 'randomize');
            [imdsTest, imdsValidation] = splitEachLabel(imdsTemp, 0.5,'randomize');
            
            % class count (yes / no)
            numClasses = numel(categories(imdsTrain.Labels));

            if i_model == "mycnn"
                lgraph = [
                    imageInputLayer([224 224 3], 'Name', 'input')
                
                    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
                    batchNormalizationLayer('Name', 'batchnorm_1')
                    reluLayer('Name', 'relu_1')
                
                    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
                
                    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
                    batchNormalizationLayer('Name', 'batchnorm_2')
                    reluLayer('Name', 'relu_2')
                
                    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
                
                    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
                    batchNormalizationLayer('Name', 'batchnorm_3')
                    reluLayer('Name', 'relu_3')
                
                    fullyConnectedLayer(numClasses, 'Name', 'fc')
                    softmaxLayer('Name', 'softmax')
                    classificationLayer('Name', 'output')
                ];
                imageSize = [224, 224];
            else
                switch i_model
                    case "alexnet"
                        net = alexnet;
                        lastLayer = 2;
                        imageSize = [227, 227];
                    case "googlenet"
                        net = googlenet;
                        lastLayer = 1;
                        imageSize = [224, 224];
                    case "resnet50"
                        net = resnet50;
                        lastLayer = 2;
                        imageSize = [224, 224];
                end
                
                lgraph = layerGraph(net);
                
                learnableLayer = lgraph.Layers(end-lastLayer);  % last fully connected layer
                classLayer = lgraph.Layers(end);  % last classification layer
                
                % new fully connected layer
                newLearnableLayer = fullyConnectedLayer(numClasses, ...
                    'Name','new_fc', ...
                    'WeightLearnRateFactor',10, ...
                    'BiasLearnRateFactor',10);
                
                % new classification layer
                newClassLayer = classificationLayer('Name','new_classoutput');
                
                % replacing layers
                lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
                lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);
            end

            YTest = imdsTest.Labels;  % Labels bilgisini augment etmeden önce alıyoruz


            % resizing images
            imdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
            imdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');
            imdsTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');
            
            options = trainingOptions(i_opt_alg, ...
                'InitialLearnRate', i_lr, ...
                'MaxEpochs', i_epoch, ...
                'MiniBatchSize', i_batch, ...
                'ValidationData', imdsValidation, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'training-progress'); % for plotting
            
            % training model
            trainedNet = trainNetwork(imdsTrain, lgraph, options);
            
            % save the trained model to a file
            filename = i_model + "_" + i_opt_alg + "_B" + string(i_batch) + "_L" + string(i_lr) + "_E" + string(i_epoch) +".mat";
            save(filename, 'trainedNet');

            % add trained model to the model dropdown
            items = app.ModelDropDown.Items;
            items = [items, filename];
            app.ModelDropDown.Items = items;
            
            YPred = classify(trainedNet, imdsTest);
            
            % Confusion Matrix
            confusionchart(YTest, YPred);
            cm = confusionmat(YTest, YPred);
            TP = cm(2,2);
            TN = cm(1,1);
            FP = cm(1,2);
            FN = cm(2,1);
            
            % Hesaplamalar:
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            precision = TP / (TP + FP);
            recall = TP / (TP + FN);  % sensitivity diye de geçiyor
            f1 = 2 * (precision * recall) / (precision + recall);


            f = uifigure('Name', 'Model Evaluation', 'Position', [100 100 350 250]);

            uilabel(f, 'Position', [20 180 300 30], 'Text', sprintf('Accuracy: %.2f%%', accuracy*100), 'FontSize', 14);
            uilabel(f, 'Position', [20 140 300 30], 'Text', sprintf('Precision: %.2f', precision), 'FontSize', 14);
            uilabel(f, 'Position', [20 100 300 30], 'Text', sprintf('Recall (Sensitivity): %.2f', recall), 'FontSize', 14);
            uilabel(f, 'Position', [20 60 300 30], 'Text', sprintf('F1-Score: %.2f', f1), 'FontSize', 14);



            app.TrainButton.Enable = true;
        end

        % Button pushed function: SelectImageButton
        function SelectImage(app, event)
            [file,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'});
            figure(app.UIFigure); % to prevent the window from getting minimized

            % check if the user canceled the file selection
            if isequal(file, 0)
                return;
            end

            % display selected image
            app.Image.ImageSource = fullfile(path, file);
            app.SelectedNameLabel.Text = file;

            app.Prediction.Text = "";
        end

        % Button pushed function: ClassifyButton
        function Classify(app, event)
            app.ClassifyButton.Enable = false;
            
            imagePath = app.Image.ImageSource;

            
            % Load the model selected by the user from the dropdown
            selectedModel = app.ModelDropDown.Value;
            loadedModel = load(selectedModel);
            trainedNet = loadedModel.trainedNet;

            % Load and preprocess the selected image
            img = imread(imagePath); % Read the selected image
            img = imresize(img, [224, 224]); % Resize the image to 224x224
            if size(img, 3) == 1
                % If grayscale, convert it to RGB by replicating the single channel
                img = repmat(img, [1 1 3]);
            end
            
            
            % app.Label.Text = inputImage

            % Predict the class of the image using the trained model
            [label, scores] = classify(trainedNet, img);
            
            % Display the predicted label
            if label == "yes"
                label = "Has Tumour";
            else
                label = "Clean";
            end
            app.Prediction.Text = label;
            % app.PredictionPer.Text = scores;

            app.ClassifyButton.Enable = true;
        end

        % Value changed function: ModelDropDown
        function ModelValueChanged(app, event)
            app.Prediction.Text = "";
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create TrainButton
            app.TrainButton = uibutton(app.UIFigure, 'push');
            app.TrainButton.ButtonPushedFcn = createCallbackFcn(app, @Train, true);
            app.TrainButton.FontName = 'Consolas';
            app.TrainButton.Position = [131 128 100 23];
            app.TrainButton.Text = 'Train';

            % Create ModelTypeDropDownLabel
            app.ModelTypeDropDownLabel = uilabel(app.UIFigure);
            app.ModelTypeDropDownLabel.HorizontalAlignment = 'center';
            app.ModelTypeDropDownLabel.FontName = 'Consolas';
            app.ModelTypeDropDownLabel.Position = [85 331 71 22];
            app.ModelTypeDropDownLabel.Text = 'Model Type';

            % Create ModelTypeDropDown
            app.ModelTypeDropDown = uidropdown(app.UIFigure);
            app.ModelTypeDropDown.Items = {'MyCNN', 'GoogLeNet', 'AlexNet', 'ResNet50'};
            app.ModelTypeDropDown.FontName = 'Consolas';
            app.ModelTypeDropDown.Position = [174 331 100 22];
            app.ModelTypeDropDown.Value = 'MyCNN';

            % Create OptimizationAlgorithmLabel
            app.OptimizationAlgorithmLabel = uilabel(app.UIFigure);
            app.OptimizationAlgorithmLabel.HorizontalAlignment = 'center';
            app.OptimizationAlgorithmLabel.FontName = 'Consolas';
            app.OptimizationAlgorithmLabel.Position = [75 283 84 30];
            app.OptimizationAlgorithmLabel.Text = {'Optimization'; 'Algorithm'};

            % Create OptimizationAlgorithmDropDown
            app.OptimizationAlgorithmDropDown = uidropdown(app.UIFigure);
            app.OptimizationAlgorithmDropDown.Items = {'Adam', 'SGDM', 'RMSProp'};
            app.OptimizationAlgorithmDropDown.FontName = 'Consolas';
            app.OptimizationAlgorithmDropDown.Position = [174 291 100 22];
            app.OptimizationAlgorithmDropDown.Value = 'Adam';

            % Create BatchSizeDropDownLabel
            app.BatchSizeDropDownLabel = uilabel(app.UIFigure);
            app.BatchSizeDropDownLabel.HorizontalAlignment = 'center';
            app.BatchSizeDropDownLabel.FontName = 'Consolas';
            app.BatchSizeDropDownLabel.Position = [88 252 71 22];
            app.BatchSizeDropDownLabel.Text = 'Batch Size';

            % Create BatchSizeDropDown
            app.BatchSizeDropDown = uidropdown(app.UIFigure);
            app.BatchSizeDropDown.Items = {'8', '16', '32', '64'};
            app.BatchSizeDropDown.FontName = 'Consolas';
            app.BatchSizeDropDown.Position = [174 252 100 22];
            app.BatchSizeDropDown.Value = '16';

            % Create LearningRateEditFieldLabel
            app.LearningRateEditFieldLabel = uilabel(app.UIFigure);
            app.LearningRateEditFieldLabel.HorizontalAlignment = 'center';
            app.LearningRateEditFieldLabel.FontName = 'Consolas';
            app.LearningRateEditFieldLabel.Position = [68 212 91 22];
            app.LearningRateEditFieldLabel.Text = 'Learning Rate';

            % Create LearningRateEditField
            app.LearningRateEditField = uieditfield(app.UIFigure, 'numeric');
            app.LearningRateEditField.Limits = [0.0001 0.02];
            app.LearningRateEditField.FontName = 'Consolas';
            app.LearningRateEditField.Position = [174 212 100 22];
            app.LearningRateEditField.Value = 0.01;

            % Create EpochCountEditFieldLabel
            app.EpochCountEditFieldLabel = uilabel(app.UIFigure);
            app.EpochCountEditFieldLabel.HorizontalAlignment = 'center';
            app.EpochCountEditFieldLabel.FontName = 'Consolas';
            app.EpochCountEditFieldLabel.Position = [81 171 78 22];
            app.EpochCountEditFieldLabel.Text = 'Epoch Count';

            % Create EpochCountEditField
            app.EpochCountEditField = uieditfield(app.UIFigure, 'numeric');
            app.EpochCountEditField.Limits = [10 Inf];
            app.EpochCountEditField.FontName = 'Consolas';
            app.EpochCountEditField.Position = [174 171 100 22];
            app.EpochCountEditField.Value = 10;

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.Position = [397 252 123 101];

            % Create TRAINLabel
            app.TRAINLabel = uilabel(app.UIFigure);
            app.TRAINLabel.HorizontalAlignment = 'center';
            app.TRAINLabel.FontName = 'Consolas';
            app.TRAINLabel.FontSize = 36;
            app.TRAINLabel.FontWeight = 'bold';
            app.TRAINLabel.Position = [126 385 104 47];
            app.TRAINLabel.Text = 'TRAIN';

            % Create PREDICTLabel
            app.PREDICTLabel = uilabel(app.UIFigure);
            app.PREDICTLabel.HorizontalAlignment = 'center';
            app.PREDICTLabel.FontName = 'Consolas';
            app.PREDICTLabel.FontSize = 36;
            app.PREDICTLabel.FontWeight = 'bold';
            app.PREDICTLabel.Position = [387 385 144 47];
            app.PREDICTLabel.Text = 'PREDICT';

            % Create ModelDropDownLabel
            app.ModelDropDownLabel = uilabel(app.UIFigure);
            app.ModelDropDownLabel.HorizontalAlignment = 'center';
            app.ModelDropDownLabel.FontName = 'Consolas';
            app.ModelDropDownLabel.Position = [382 128 38 22];
            app.ModelDropDownLabel.Text = 'Model';

            % Create ModelDropDown
            app.ModelDropDown = uidropdown(app.UIFigure);
            app.ModelDropDown.Items = {};
            app.ModelDropDown.ValueChangedFcn = createCallbackFcn(app, @ModelValueChanged, true);
            app.ModelDropDown.FontName = 'Consolas';
            app.ModelDropDown.Position = [435 128 100 22];
            app.ModelDropDown.Value = {};

            % Create SelectImageButton
            app.SelectImageButton = uibutton(app.UIFigure, 'push');
            app.SelectImageButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImage, true);
            app.SelectImageButton.FontName = 'Consolas';
            app.SelectImageButton.Position = [409 171 100 23];
            app.SelectImageButton.Text = 'Select Image';

            % Create SelectedNameLabel
            app.SelectedNameLabel = uilabel(app.UIFigure);
            app.SelectedNameLabel.HorizontalAlignment = 'center';
            app.SelectedNameLabel.FontName = 'Consolas';
            app.SelectedNameLabel.Position = [382 212 153 22];
            app.SelectedNameLabel.Text = '';

            % Create ClassifyButton
            app.ClassifyButton = uibutton(app.UIFigure, 'push');
            app.ClassifyButton.ButtonPushedFcn = createCallbackFcn(app, @Classify, true);
            app.ClassifyButton.FontName = 'Consolas';
            app.ClassifyButton.Position = [409 88 100 23];
            app.ClassifyButton.Text = 'Classify';

            % Create Prediction
            app.Prediction = uilabel(app.UIFigure);
            app.Prediction.HorizontalAlignment = 'center';
            app.Prediction.FontName = 'Consolas';
            app.Prediction.Position = [387 64 148 22];
            app.Prediction.Text = '';

            % Create PredictionPer
            app.PredictionPer = uilabel(app.UIFigure);
            app.PredictionPer.HorizontalAlignment = 'center';
            app.PredictionPer.FontName = 'Consolas';
            app.PredictionPer.Position = [387 31 148 22];
            app.PredictionPer.Text = '';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = app_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end