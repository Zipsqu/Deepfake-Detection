**Honours Project:** Developing & Testing a CNN based deepfake detection. <br /> 
**Hardware:** NVIDIA GeForce GTX 1660 Ti, Intel Core i7-10750H @2.60Ghz, 32GB RAM <br />
**Dataset:** DFDC Sample dataset (400 videos) <br /> <br />
*Once pre-processed, Testing & Validation Datasets contain rougly 13k files each, while training dataset contains around 190k files. Half of these files are frames, and the other half are JSON files.*  <br />



1. Pre-processing: <sup>#done by using Pre-Processing.py</sup> <br />
   -Cropped Faces & Landmark extraction (MTCNN) <br />
   -Adjusting frame size (to 224x224 as required for EfficientNetB0)  <br />
   -Splitting dataset into Trainig/Testing/Evaluation (80%,10%,10%) <br /> 



2. CNN Detector <br />
   -Augumentation ( <br />
   -EfficientNetB0 Based (smallest EfficientNet) <br />
   
   -
