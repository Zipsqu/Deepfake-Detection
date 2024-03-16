**Honours Project:** *Developing & Testing a CNN based deepfake detection.* <br /> 
**Hardware:** *NVIDIA GeForce GTX 1660 Ti, Intel Core i7-10750H @2.60Ghz, 32GB RAM* <br />
**Dataset:** *DFDC Sample dataset (400 videos)* <br /> <br />



 <sub>#Use Pre-Processing.py</sub>
1. Pre-processing: <br />
   -Cropped Faces & Landmark extraction (MTCNN) <br />
   -Adjusting frame size to 224x224 (native for EfficientNetB0)  <br />
   -Splitting dataset into Trainig/Testing/Evaluation (80%,10%,10%) <br /> 



2. CNN Detector <br />
   -Augumentation (Albumentations: XXX) <br />
   -EfficientNetB0 Based (smallest EfficientNet) <br />
   -Batch size:
   -
