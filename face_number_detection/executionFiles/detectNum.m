function numbers = detectNum(filename)

    % Load the models
    models = load(fullfile(pwd,'models.mat'));
    disp('Models loaded');
    
    % Try reading the video, if it does not work we have an image
    try
        % Load the video
        vid = VideoReader(filename);
        
        % Detect numbers on the frames of the video
        ns = [];
        disp('Recognizing numbers in video');
        
        while hasFrame(vid)
            % Read the frame
            image = readFrame(vid);
            
            % Detect numbers on the image and compile them.
            n = findNumbers(image,models);
            
            if ~isempty(n)
                ns = [ns,n(1)];
            end
        end
        
        % Select most recognized numbers
        numbers = mode(ns);
        disp('Done');
        
    catch
        % Load the image and check that the orientation is appropriate
        image = orientedImage(filename);
        disp('Recognizing numbers in picture');
        
        % Detect numbers on the image
        numbers = findNumbers(image,models);
        disp('Done');
    end
    
end