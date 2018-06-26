% The code was developed following the Matlab examples stated in the report

%% Single Example

% Get the image
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/faces/001';
I = imread(fullfile(root,'1.png'));

% Extract HOG features
[featureVector,hogVisualization] = extractHOGFeatures(I);

% Show image and hog features
figure;
imshow(I); 
hold on;
plot(hogVisualization);

%% HOG from faces

% Read the dataset
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/faces';
imgs = imageSet(root,'recursive');

% Extract HOG features from the dataset
X = []; y = [];
for i=1:size(imgs,2)
    for j=1:50
        % Get label and image, extract features
       label = imgs(i).Description;
       image = read(imgs(i),j);
       image = imresize(image, [100 100]);
       feature = extractHOGFeatures(image);

       % Store HOG pictures and labels
       X = [X;feature];
       y = [y;label];
    end
end

%% Split training, test

% Get training, test and validation set
[trainInd,valInd,testInd] = divideint(size(X,1),0.7,0.2,0.1);
X_train = X(trainInd,:);    y_train = y(trainInd,:);
X_val = X(valInd,:);    y_val = y(valInd,:);
X_test = X(testInd,:);    y_test = y(testInd,:);

%% SVM training

% Train svm on training set
t = templateLinear();
svmModelHOG = fitcecoc(X_train, y_train,'Learners',t);

% Use validation set to tune parameters
y_val_svm = predict(svmModelHOG, X_val);
accuracy_val_svm = sum(str2num(y_val)==str2num(y_val_svm))/length(y_val)

%% Evaluate in test set

% Make class predictions using the test features.
y_pred_svm = predict(svmModelHOG, X_test);

% Evaluate results on test set
accuracy_hog_svm = sum(str2num(y_test)==str2num(y_pred_svm))/length(y_test)

%% Decision Tree training

% Train decision tree on training set
dtModelHOG = fitctree(X_train,y_train,'MaxNumCategories',100);

% Make class predictions using the test features.
y_val_dt = predict(dtModelHOG, X_val);
accuracy_val_dt = sum(str2num(y_val)==str2num(y_val_dt))/length(y_val)

%% Evaluate in test set

% Make class predictions using the test features.
y_pred_dt = predict(dtModelHOG, X_test);

% Evaluate results in test set
accuracy_hog_dt = sum(str2num(y_test)==str2num(y_pred_dt))/length(y_test)