%% Read data

% Load videos from the class
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/Group11of11';
vid = VideoReader(fullfile(root,'IMG_0612.m4v'));

%% Iterate images

% Instantiate the cascade classifier
detector = vision.CascadeObjectDetector;
detector.MergeThreshold = 5;
detector.MinSize = [50 50];

for i=1:10:vid.Duration*vid.FrameRate
    % Read frames and detect faces
    I = read(vid,i);
    bbox = step(detector,I);
    
    for j=1:size(bbox,1)
        % Crop the picture
        cd('C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/facesGroup');
        face = imcrop(I,bbox(j,:));
        
        % Pre-organize the faces using AlexNet and store them
        label = cellstr(classify(faceNet,imresize(face,[227 227])));
        mkdir(label{1});  cd(label{1});
        imwrite(face,strcat(num2str(i),num2str(j),'.png'));
    end
end