%% Test individual face recognition
I1 = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers/045/IMG_0888.jpg';
P1 = RecogniseFace(imread(I1),'nil','CNN');
P2 = RecogniseFace(imread(I1),'LBP','DTR');
P3 = RecogniseFace(imread(I1),'HOG','DTR');
P4 = RecogniseFace(imread(I1),'LBP','SVM');
P5 = RecogniseFace(imread(I1),'HOG','SVM');

%% Test group face recognition
I2 = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/Group11of11/IMG_0628.jpg';
P6 = RecogniseFace(imread(I2),'nil','CNN');
P7 = RecogniseFace(imread(I2),'LBP','DTR');
P8 = RecogniseFace(imread(I2),'HOG','DTR');
P9 = RecogniseFace(imread(I2),'LBP','SVM');
P10 = RecogniseFace(imread(I2),'HOG','SVM');

%% Test numbers
I3 = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers/045/IMG_0888.jpg';
numbers1 = detectNum(I3);

%% Test numbers in videos
I4 = 'C:/Users/Sergio/Documents/Msc Data Science/6-Computer Vision/Coursework/Data/numbers/045/IMG_0883.mov';
numbers2 = detectNum(I4);