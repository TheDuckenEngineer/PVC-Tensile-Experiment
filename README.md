# PVC Tensile Experiment 
PVC gels, plasticized polyvinyl chloride gels, are popular  dielectric elastomer actuators known for their unique actuation mechanism. In the analysis of PVC gels, material models are necessary to facilitate accurate physics simulations. 

![alt text](1114.jpg)

This repository contains a digital image processing program that uses TrackPy and OpenCv to determine strain measurements by employing particle tracking. 


## Test Parameters
PVC gels are hyper-elasatic materials with extremely low elastic modulii. When testing specimens cut to ASTM 638 Type-4 dog bone shape, noticable elongation in the non-gauge length region occurs. The displacement measurements by the tensile machine would incorperate this elongation, inducing errors in strain measurements. By using digital image processing, the PVC's strain can be measured by capturing images of the specimen with markers placed on it's surface. Setting the tensile machine's sample rate and camera frame rate to a commmon rate couples the input force and strain with marginal deviation. Tests are conducted at 1 mm/min. 

## Program
Strain Processing jupitor notebook is the main script. The file name is called as a string from the image data folder. Smalls and bigs are variables setting acceptable marker areas. Often, image processing functions will need to be altered within the Functions file which houses all the program functions. The markers need to be blue to be identified. With each image window that's produced, pressing esc closes it.   

