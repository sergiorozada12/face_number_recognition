% The code was developed following the Matlab examples stated in the report

%% LBP from faces

% Load the dataset
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/faces';
imgs = imageSet(root,'recursive');

% Extract LBP features from the dataset
X = []; y = [];
for i=1:size(imgs,2)
    for j=1:50
        % Get label and LBP features from image
        label = imgs(i).Description;
        image = read(imgs(i),j);
        image = imresize(image, [100 100]);
        feature = extractLBPFeatures(rgb2gray(image));
        
        % Store them
        X = [X;feature];
        y = [y;label];
    end
end

%% Split training, test

% Get training, validation and test set
[trainInd,valInd,testInd] = divideint(size(X,1),0.7,0.2,0.1);
X_train = X(trainInd,:);    y_train = y(trainInd,:);
X_val = X(valInd,:);    y_val = y(valInd,:);
X_test = X(testInd,:);    y_test = y(testInd,:);

%% SVM training

% Fit the SVM model
t = templateLinear();
svmModelLBP = fitcecoc(X_train, y_train,'Learners',t);

% Use validation set to tune parameters
y_val_svm = predict(svmModelLBP, X_val);
accuracy_val_svm = sum(str2num(y_val)==str2num(y_val_svm))/length(y_val)

%% Evaluate on test set

% Make class predictions using the test features.
y_pred_svm = predict(svmModelLBP, X_test);

% Tabulate the results using a confusion matrix.
accuracy_lbp_svm = sum(str2num(y_test)==str2num(y_pred_svm))/length(y_test)

%% Decision Tree training

% Fit the decision tree model
dtModelLBP = fitctree(X_train,y_train,'MaxNumCategories',150);

% Use validation set to tune parameters
y_val_dt = predict(dtModelLBP, X_val);
accuracy_val_dt = sum(str2num(y_val)==str2num(y_val_dt))/length(y_val)

%% Evaluate model in test set

% Make class predictions using the test features.
y_pred_dt = predict(dtModelLBP, X_test);

% Tabulate the results using a confusion matrix.
accuracy_lbp_dt = sum(str2num(y_test)==str2num(y_pred_dt))/length(y_test)