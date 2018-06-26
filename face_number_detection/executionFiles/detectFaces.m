function [faces,labels,expresions] = detectFaces(image,feature,type,models)
    % The function detects faces in an image. It returns the box of the
    % face, the label associated to the face and the expresion of the person
    
    % Image to grayscale to speed performance
    try
        image = rgb2gray(image);
    end
    
    % Initial face detect: We use the matlab face detector to find the
    % rough number of faces on the image.
    detector = vision.CascadeObjectDetector;
    bb = step(detector,image);
    disp('Number of faces counted');
    
    % Now we are detecting faces again but params will vary from individual
    % to group pictures
    if size(bb,1) > 6
        % Sharpen the image to get rid of false positives
        imageSharp = imsharpen(image,'Radius',2,'Amount',1);
        
        % Detect the faces in group pictures
        detector.MergeThreshold = 5;
        bboxes = step(detector,imageSharp);
    else
        % Crop the lowest part of the image
        y = size(image,1);
        image = image(1:y-round(y*0.4),:);
        
        % Detect the faces
        minFace = 140;  maxFace = 800;
        detector.MergeThreshold = 7;    detector.ScaleFactor = 1.05;
        detector.MinSize = [minFace,minFace];   detector.MaxSize = [maxFace,maxFace];
        bboxes = step(detector,image);
    end
    
    disp('Faces detected');
    
    % Store faces, labels and feelings
    faces = bboxes;
    labels = [];
    expresions= [];
    
    for i=1:size(faces,1)
        % Detect what face is using the proper model
        face = imcrop(image,faces(i,:));
        
        % Recognice the face
        labelFace = getLabel(face,models,feature,type);
        labels = [labels;labelFace];
        
        % Recognice the expresion
        faceI = cat(3,face,face,face);
        expresion = cellstr(classify(models.expresionNet,imresize(faceI,[227 227])));
        expresion = expresion{1};
        expresions = [expresions;expresion];
    end
    
    disp('Expresion and face recognition done');
end