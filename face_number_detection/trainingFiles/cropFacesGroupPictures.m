%% Read images from group

% Load all the group pictures 
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/Group11of11';
imgs = imageSet(root,'recursive');

%% Read each image on the folder

% Detect the faces with the Cascade classifier
detector = vision.CascadeObjectDetector;
detector.MergeThreshold = 5;
detector.MinSize = [50 50];

for i=1:imgs.Count
    
    % Load picture and detect face
    I = read(imgs(1),i);
    bbox = step(detector,I);
    
    for j=1:size(bbox,1)
        % Crop the image and store it 
        cd('C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/facesGroup');
        face = imcrop(I,bbox(j,:));
        
        % Use AlexNet to pre-organize detected faces
        label = cellstr(classify(faceNet,imresize(face,[227 227])));
        
        % Store each crop in the proper folder
        mkdir(label{1});  cd(label{1});
        imwrite(face,strcat(num2str(i),num2str(j),'.png'));
    end
end