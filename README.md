**Honours Project:** *Developing & Testing a CNN based deepfake detection.* <br /> 
**Hardware:** *NVIDIA GeForce GTX 1660 Ti, Intel Core i7-10750H @2.60Ghz, 32GB RAM* <br />
**Dataset:** *DFDC Sample dataset (400 videos)* <br /> <br />



 <sub>#Use Pre-Processing.py</sub>
1. Pre-processing: <br />
   -Cropped Faces & Landmark extraction (MTCNN) <br />
   -Adjusting frame size to 224x224 (native for EfficientNetB0)  <br />
   -Splitting dataset into Trainig/Testing/Evaluation (80%,10%,10%) <br /> 



2. CNN Detector: <br />
- EfficientNetB0 with pre-trained wieghts (Imagenet) <br />
*#Overfitting was the biggest problem due to the size of EfficientNetB0 and relatively small dataset (to fight this, I used heavier augmentation by torchvision transforms, weight decay, droput rates & early stop if validation loss hasn't improved for 3 consecutive epochs).*
