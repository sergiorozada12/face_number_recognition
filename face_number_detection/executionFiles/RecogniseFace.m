function P = RecogniseFace(image,featureType,classifierName)

    % First of all load the models to use
    models = load(fullfile(pwd,'models.mat'));
    disp('Models loaded');
    
    % Detect the faces in the image, labels, bounding box and feeling
    [faces,labels,expresions] = detectFaces(image,featureType,classifierName,models);
    
    % Calculate center of the bounding box
    cx = faces(:,1) + round(faces(:,3)./2);
    cy = faces(:,2) + round(faces(:,4)./2);

    % Return matrix with all values
    P = [];
    
    if ~isempty(labels)
        P = [str2num(labels),cx,cy,str2num(expresions)];
    end

end