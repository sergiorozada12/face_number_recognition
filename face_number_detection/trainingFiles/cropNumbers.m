%% Load the images

% Load the data and set the output folder
root = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/n';
out = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/n1';
imgs = imageSet(root,'recursive');

%% Iterate the dataset

% Instantiate number detector
minNumber = 100;  maxNumber = 750;
detector = vision.CascadeObjectDetector('numbers.xml');
detector.MergeThreshold = 1;
detector.MinSize = [minNumber minNumber];
detector.MaxSize = [maxNumber maxNumber];

% Iterate all persons
for i=1:size(imgs,2)
    % Iterate all images
    for j=1:imgs(i).Count
        %Load the image and detect rois
        try
            I = rgb2gray(read(imgs(i),j));
        catch
            I = read(imgs(i),j);
        end
        
        bboxes = step(detector,I);
        
        for k=1:size(bboxes,1)
            % Now binarize the image and check if there are numbers inside
            crop = imcrop(I,bboxes(k,:));
            binaryCrop = logical(1-imbinarize(crop,0.65));

            % Filter out non-number regions
            stats = regionprops(binaryCrop, 'Area','Extent','Orientation');
            minDig = 15;  maxDig = 120; 
            minExt = 0.25;  maxExt = 0.51;
            minOrient = 70; maxOrient = 90;
            n = stats(([stats.Area] > (minDig*minDig))&([stats.Area] < (maxDig*maxDig))&...
                ([stats.Area] > size(crop,1)*size(crop,2)/100) & ...
                ([stats.Extent] > minExt) & ([stats.Extent] < maxExt)&...
            (   abs([stats.Orientation]) < maxOrient) & (abs([stats.Orientation]) > minOrient));

            % If there are two numbers, store crop
            if size(n,1) == 2
               % Store the crop
                o = fullfile(out,imgs(i).Description);
                imwrite(crop,fullfile(o,strcat(num2str(j),'.png'))); 
            end   
        end
    end
end