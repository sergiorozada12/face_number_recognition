%% Read images from group

% Load all the individual pictures
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers';
imgs = imageSet(root,'recursive');

%% Read each image on the folder

% Instantiate the cascade detector
detector = vision.CascadeObjectDetector;
detector.MergeThreshold = 5;
detector.MinSize = [50 50];

for i=1:size(imgs,2)
    % Get the label of the person
    label = imgs(i).Description
    
    for j=1:imgs(i).Count
        % Load picture and detect face
        I = read(imgs(i),j);
        bbox = step(detector,I);

        for k=1:size(bbox,1)
            % Crop the image and store it 
            cd('C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/faces');
            face = imcrop(I,bbox(k,:));

            % Store each crop in the proper folder
            mkdir(label);  cd(label);
            imwrite(face,strcat(num2str(i),num2str(j),'.png'));
        end
    end
end