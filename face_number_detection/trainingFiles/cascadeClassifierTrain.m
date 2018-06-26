% The code was developed following the Matlab examples stated in the report

%% Negative labels

% The root of the folder that contains background images is set
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework';
negativeFolder = fullfile(root,'Data','negative');

%% Train the cascade classifier

% Cascade classifier training: parameters were set  by trial and error
trainCascadeObjectDetector('faces.xml',roi(:,[1 2]), ...
    negativeFolder,'FalseAlarmRate',0.0001,'TruePositiveRate',0.99995,'NumCascadeStages',20,...
    'ObjectTrainingSize',[24 24],'FeatureType','HOG','NegativeSamplesFactor',2);
    
%% Testing the detector

% Test the cascade detector in an image
I = rgb2gray(imread('C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers/001/IMG_0668.JPG'));
figure();   imshow(I);

detector = vision.CascadeObjectDetector('faces.xml');
bbox = step(detector,I);

detectedImg = insertObjectAnnotation(I,'rectangle',bbox,'number');
figure; imshow(detectedImg);