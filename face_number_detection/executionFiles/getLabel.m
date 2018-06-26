function label = getLabel(image,models,feature,type)
            
    % HOG features
    if feature == 'HOG'
        % Resize and extract hog features
        hog = extractHOGFeatures(imresize(image,[100 100]));
        
        % Use the classifier (SVM or DT)
        if type == 'SVM'
            label = predict(models.svmModelHOG,hog);
        else
            label = predict(models.dtModelHOG,hog);
        end

    % LBP features
    elseif feature == 'LBP' 
        % Resize and extract lbp features
        lbp = extractLBPFeatures(imresize(image,[100 100]));
        
        % Use the classifier (SVM or DT)
        if type == 'SVM'
            label = predict(models.svmModelLBP,lbp);
        else
            label = predict(models.dtModelLBP,lbp);
        end
    
    % CNN model
    else
        % Create rgb image AlexNet
        image = cat(3,image,image,image);
        
        % Resize and classify
        label = cellstr(classify(models.faceNet,imresize(image,[227 227])));
        label = label{1};
    end
end