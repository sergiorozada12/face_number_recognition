** ALL FILES MUST BE STORED IN THE SAME FOLDER AND PATH OF THE FOLDER
HAS TO BE ADDED TO MATLAB ENVIRONMENT. IDEALLY, TEST SCRIPT SHOULD BE
RUN IN THE SAME FOLDER**

**WARNING: MATLAB CURRENT FOLDER HAS TO BE SET THE FOLDER WHERE THE FILES
TO RUN ARE. IF NOT, MODELS.MAT AND NUMBERS.XML WILL NOT BE FIND BY MATLAB**

**MODELS.MAT HAS TO BE PLACES IN FOLDER 'TORUN'

**MATLAB VERSION USED: R2018a**

**PACKAGES USED:**
- Statistics and Machine Learning Toolbox
- Computer Vision System Toolbox
- Deep Neural Network
- Neural Network Toolbox Model for AlexNet Network

****************************************************************************************
**FACE RECOGNITION**

- RecogniseFace.m
- detectFaces.m
- getLabel.m

**NUMBER RECOGNITION**

- detectNum.m
- orientedImage.m
- findNumbers.m

**MODELS**

- models.mat
- numbers.xml

**TEST**
- test.m: Shows the possible uses of the functions. If instructions are followed, this script have
to run. If it does not work, please, contact back.

**TRAINING**

Training models:
- alexnetTraining.m
- hogFeatures.m
- lbpFeatures.m
- cascadeClassifierTrain.m

Preparing datasets:
- cropFacesGroupPictures.m
- cropFacesGroupVideo.m
- cropFacesSinglePictures.m
- cropFacesSingleVideo.m
- cropNumbers.m

****************************************************************************************
** DEFINITION OF THE FUNCTIONS **

1---[label,cx,cy,expresion] = RecogniseFace(image,feature,model)

- image: image must be vertical and can be rgb or grayscale.
- feature: 'LBP'/'HOG' when model is 'DTR'/'SVM', if model is 'CNN', feature must be 'nil'
- model: 'CNN' for AlexNet, 'SVM' for Support Vector Machine and 'DTR' for Decision Tree

- output: matrix P with each row being compose of label, center and expresion of each individual

E.g. [label,cx,cy,expresion] = RecogniseFace(image,'nil','CNN')


2---[numbers] = detectNum(root)

- root: absolute path to the image or video

- output: numbers is an array of the numbers detected in the file

E.g. [45,63] = RecogniseFace('C:/student/pictures/number45and63.png')
