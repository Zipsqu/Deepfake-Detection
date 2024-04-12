**Project:** *Developing & Testing a CNN based deepfake detection.* <br /> 
**Hardware:** *NVIDIA GeForce GTX 1660 Ti, Intel Core i7-10750H @2.60Ghz, 32GB RAM* <br />
**Dataset:** *DFDC Dataset Packages 00-02 (4,782 Videos; 424 Real; 4357 Fake)* <br /> 
**Pre-processed dataset:** *Roughly 1,300,000 etracted frames split into training/testing/evaluation.* <br />


 <sub>#Check Pre-Processing.py </sub>  <br />
 <sub>#Adjust variables in pre-processing directory if necessary. </sub> 
1. Pre-processing: <br /> 
   -Cropped Faces & Landmark extraction (MTCNN) <br />
   -Splitting dataset into Trainig/Testing/Evaluation (80%,10%,10%) <br /> 


 <sub>#Check EfficientNetb0 directory</sub>  
 <sub>#Train.py is used for training the model & makes uses of the first data loader. Adjust variables in both if willing to run it on your own dataset. </sub>  
 <sub>#Testing.py is used for evaluation of the model using previously created weights (again, adjust variables). This code make a use of the data_loader2.py, which is nearly the exact same version of previous data_loader. </sub>  
 
2. EfficientNetB0 with pre-trained weights (Imagenet): <br />
-Overfitting was the biggest problem due to the size of EfficientNetB0 and relatively small dataset (to fight this, I used heavier augmentation by torchvision transforms, weight decay, droput rates & early stop if validation loss hasn't improved for 3 consecutive epochs). <br />
-Extraced frames are resized, normalized, converted to RGB and fed to the training model using tesnsors.
-During evaluation, the model thre the respective accuracy of 85% (worth taking into note that this number might vary when training/evaluating on different datasets). With that being said, model definietly had troubles classyfing REAL frames (only 6% of REAL frames are corretcly classified).

 <sub>#Check GANs directory</sub>  

3. GANs Detector:
- 
   
