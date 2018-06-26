function labels = findNumbers(image,models)
    % The function detects numbers in an image. The function is prepared to
    % detect more than one number at once. It deals with multiple numbers
    
    % Image to grayscale to speed performance
    try
        image = rgb2gray(image);
    end
    
    % Instantiate the number detector and detect possible numbers. The
    % detector was trained for this task.
    disp('Proposing Regions Of Interest');
    minNumber = 100;  maxNumber = 750;
    detector = vision.CascadeObjectDetector(fullfile(pwd,'numbers.xml'));
    detector.MergeThreshold = 1;
    detector.MinSize = [minNumber minNumber];
    detector.MaxSize = [maxNumber maxNumber];
    bboxes = step(detector,image);
    
    numbers = [];
    for i=1:size(bboxes,1)
        disp('Filtering regions');  
        
        % Now binarize the image and check if there are numbers inside
        crop = imcrop(image,bboxes(i,:));
        binaryCrop = logical(1-imbinarize(crop,0.65));
        
        % Number if orientation mainly vertical and extent low, there
        % is a minimum area for numbers and the crop has to be
        % relatively small in comparison with the number
        stats = regionprops(binaryCrop, 'Area','Extent','Orientation');
        minDig = 15;  maxDig = 120; 
        minExt = 0.25;  maxExt = 0.51;
        minOrient = 75; maxOrient = 90;
        n = stats(([stats.Area] > (minDig*minDig))&([stats.Area] < (maxDig*maxDig))&...
            ([stats.Area] > size(crop,1)*size(crop,2)/100) & ...
            ([stats.Extent] > minExt) & ([stats.Extent] < maxExt)&...
        (   abs([stats.Orientation]) < maxOrient) & (abs([stats.Orientation]) > minOrient));
            
        % If there are two numbers, append it to number proposals
        if size(n,1) == 2
           numbers = [numbers;bboxes(i,:)];
        end
    end
    
    % Once we have all possible numbers, check that none of them are
    % too near between them ( This would mean multiple detection of the
    % same number)
    candidates = ones(length(size(numbers,1)));
    minDist = 300;
    for i=1:size(numbers,1)
        for j=i+1:size(numbers,1)
            dist = sqrt((numbers(i,1)-numbers(j,1))^2+(numbers(i,2)-numbers(j,2))^2);
            if dist < minDist
                areaI = numbers(i,3)*numbers(i,4);
                areaJ = numbers(j,3)*numbers(j,4);
                if areaI > areaJ
                    candidates(j) = 0;
                else
                    candidates(i) = 0;
                end
            end
        end      
    end
    
    % return those candidates which are true
    numbers(candidates == 0,:) = [];
    
    % Detect the label of each number
    disp('Recognizing numbers');
    labels = [];
    for i=1:size(numbers,1)
         % Detect what number is using a CNN
        number = imcrop(image,numbers(i,:));
        number = cat(3,number,number,number);
        labelNumber = cellstr(classify(models.numberNet,imresize(number,[227,227])));
        labels = [labels,str2num(labelNumber{1})];
    end
end