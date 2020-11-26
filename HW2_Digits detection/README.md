# CS_T0828_HW2
code for the fifth accuracy in Digits detection Challenge. <br>

## Hardware
● Ubuntu 20.04 <br>
● Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
● NVIDIA GeForce GTX 1080 Ti <br>

## Introduction and details
There is the outlines in this compitions <br>
1. [Installation](#Installation) <br>
2. [Getting_labels](#Getting_labels) <br>
3. [Implement](#Implement) <br>
4. [Testing](#Testing) <br>
5. [Organising Materials](#testing) <br>
6. [Results](#Results)<br>
7. [Make-Submission](#Make-Submission)<br>

## Installation
Using Anaconda and pytorch to implement this method.

    conda create -n Classification python=3.6
    conda install pytorch -c pytorch
    conda install torchvision -c pytorch

## Getting_labels
Change the path which is in the `get_labels.py`.

    f = h5py.File('./train/digitStruct.mat','r')
    Image.open('./train/'+IMG_NAME)
In order to get the ground truth. <br>
For example： <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/labels.png)

## Implement
The code is programing on 

    Ubuntu 20.04
    python 3.6
    Nvidia GeForce 1080Ti 
    CUDA 11.0
    cuDNN v8.0.5.
 
The training flowchart：
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/training%20flowchart.png)

*Step 1.* Git clone the darknet form https://github.com/AlexeyAB/darknet. <br>
 
*Step 2.* Open the file Makefile, and modify the values of some parameters, the first few lines of the makefile: turn on GPU acceleration, turn on opencv, and turn on the cuDNN. Change the value from 0 to 1. <br>

    GPU=1
    CUDNN=1
    OPENCV=1
    
*Step 3.* Recompile the darknet in darknet-master directory:make. <br>
    
    make
    
*Step 4.* Preparing the data : training data, ground truth label, and some data that the yolov4’s method needs. For example:.data and .names <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/path.png)
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/data.png)

*Step 5.* After preparing all of the data, we need to create the .cfg file and start to train the model with the yolo [pretrained](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights "link") model. <br>

**Changed the filter value such as filters=(classes + 5)x3) and the class value.** <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/model.png)

## Testing
Using this code to do the testing. <br>


    Loading weights from /home/naip/IOC/yolo_tiny/darknet/backup/yolov4_IOC_tiny_last.weights...
    seen 64, trained: 33718 K-images (526 Kilo-batches_64)
    Done! Loaded 38 layers from weights-file
    Detection layer: 30 - type = 28
    Detection layer: 37 - type = 28
    /home/naip/IOC/yolo_tiny/darknet/data/IOC/test/1.png: Predicted in 240.240000 milli-seconds.
    5: 93%


    /home/naip/IOC/yolo_tiny/darknet/data/IOC/test/13067.png: Predicted in 4.770000 milli-seconds.
    2: 52%  (left_x:  115   top_y:   14   width:   13   height:   18)
    2: 56%  (left_x:  125   top_y:   14   width:   12   height:   18)
    7: 71%  (left_x:  138   top_y:   14   width:   12   height:   17)
    Enter Image Path:  Detection layer: 30 - type = 28
     Detection layer: 37 - type = 28
    /home/naip/IOC/yolo_tiny/darknet/data/IOC/test/13068.png: Predicted in 4.959000 milli-seconds.
    6: 99%  (left_x:   35   top_y:    8   width:   11   height:   25)
    7: 95%  (left_x:   47   top_y:    9   width:   10   height:   24)


## Organising Materials
Input command to get the result form the test data by yolov4 model, the value of the bounding box is x_center, y_center, width and height. <br>
So, I load the test image size to calculate the (y1, x1, y2, x2) as (top , left, right ,bottom) in order to get the mAP value. <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/result.png)   

## Results
### Prediction 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/pred_img.png) <br> 
### speed benchmark from google Colab
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/speed.png) <br> 

## Make-Submission
Submit the file `StudentID.csv`, to the google drive and  get the mPA scroe from TA. <br>
