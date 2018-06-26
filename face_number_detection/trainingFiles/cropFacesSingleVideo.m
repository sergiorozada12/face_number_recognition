%% Load the data

% Data is find in this location
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers';
imgSets = imageSet(root,'recursive');

%% Crop images from videos

% For all the persons
for i=1:size(imgSets,2)
    % Label of the persons
    label = imgs(i).Description;
    
    % Load all the videos
    files = dir(fullfile(imgFolder, label,'*.mov'));
    
    % If there are videos
    if(~isempty(files))
        
        % Set output root
        out = fullfile(root,'Data','created','169','faces');
        
        % For each video
        for j=1:size(files,1)
            
            % Open the video and read frames
            cd(files(j,1).folder)           
            vid = VideoReader(files(j,1).name);
            images = read(vid);
            
            % For all the frames in the video
            for k=1:size(images,4)
                
                %Instantiate people detector
                FaceDetector = vision.CascadeObjectDetector;
                FaceDetector.MinSize = [100 100];
                FaceDetector.MergeThreshold = 15;
                
                % Rotate images if needed
                image = rgb2gray(images(:,:,:,k));
                if size(image,1)<size(image,2)&&(~((strcmp(imgSets(i).Description,'009'))||...
                        strcmp(imgSets(i).Description,'010')||strcmp(imgSets(i).Description,'108')||...
                        strcmp(imgSets(i).Description,'011')||strcmp(imgSets(i).Description,'012')))
                    image = imrotate(image,-90);
                end
                
                % Detect faces
                bboxes = step(FaceDetector,image);
                
                % Store the crops
                if(~isempty(bboxes))
                    personCrop = imcrop(image,bboxes(1,:));
                    cd(out);
                    imwrite(personCrop,strcat('mov',num2str(j),num2str(k),'.png'));
                end         
            end
        end
    end
end