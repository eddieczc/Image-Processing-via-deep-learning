# CS_T0828_HW3
code for the fifth accuracy in Instance Segmentation Challenge. <br>

## Hardware
● Ubuntu 20.04 <br>
● Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
● NVIDIA GeForce GTX 1080 Ti <br>

## Introduction and details
There is the outlines in this compitions <br>
1. [Installation](#Installation) <br>
2. [Architecture](#Architecture)<br>
3. [Implement](#Implement) <br>
4. [Testing](#Testing) <br>
5. [Results](#Results)<br>
6. [Make-Submission](#Make-Submission)<br>
7. [Reference](#Reference)<br>

## Installation
Using Anaconda and pytorch to implement this method.

    conda create -n Segmentation python=3.6
    git clone https://github.com/facebookresearch/detectron2.git
    git clone https://github.com/dbolya/yolact.git

## Architecture
#### Detectron2
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Detectron2_Architecture1.PNG) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Detectron2_Architecture2.PNG) <br> 
#### YOLACT
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/YOLACT_Architecture.PNG) <br> 

## Implement
The code is programing on 

    Ubuntu 20.04
    python 3.6
    Nvidia GeForce 1080Ti 
    CUDA 11.0
    cuDNN v8.0.5.
 
#### Detectron2 (The pretrained model must be made by ImageNet)
Put the pretrained model in the `train.py` download from https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md?fbclid=IwAR16iSEh72483X0q6DZaTqGYtwxjFZXuSOKLo5LS2Vzgha-umDwnSTG0Cns <br> 
Change the training data's path where you put the training data <br> 
*Hyperparameters* <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Detectron2_Hyperparameters.PNG) <br> 
#### YOLACT (The pretrained model must be made by ImageNet)
Put the pretrained model in the `train.py` download from https://github.com/dbolya/yolact <br> 
Change the training data's path where you put the training data <br> 
*Hyperparameters* <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/YOLACT_Hyperparameters.PNG) <br>    


## Testing
After the initial training in order to find the best pretrained model for each method.
Load the TA's program `utils_HW3.py` in the `test.py` to get the submission file.
#### Detectron2
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Detectron2_Performance.PNG) <br> 
#### YOLACT
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/YOLACT_Performance.PNG) <br> 

Choose the best performance pretrained model in each method and to do training for several epoch. <br>
Get the final mAP score for each method.
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Final_Performance.PNG) <br> 


## Results
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Result2.PNG) <br> 
### Prediction 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Result.PNG) <br>     


## Make-Submission
Use the `test.py` or `eval.py` to get the final file. <br>
Submit the file `StudentID.json`, to the google drive and  get the mPA scroe from TA. <br>


## Reference
#### Detectron2 (https://github.com/facebookresearch/detectron2)
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Detectron2.PNG) <br>     
#### YOLACT (https://github.com/dbolya/yolact)
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/YOLACT.PNG) <br>     
