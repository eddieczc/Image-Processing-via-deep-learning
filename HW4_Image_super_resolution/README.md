# CS_T0828_HW4
code for the first psnr in  Image super resolution Challenge. <br>

## Hardware
● Windows 10 <br>
● Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
● NVIDIA GeForce GTX 1080 Ti <br>
and <br>
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
#### SRGAN
SRGAN uses perceptual loss (perceptual loss) and confrontation loss (competition loss) to improve the realism of the restored picture. Perceptual loss is the feature extracted by the convolutional neural network. By comparing the difference between the features of the generated image and the target image after the convolutional neural network, the generated image and the target image are semantically and styled. More similar.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/SRGAN1.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/SRGAN2.png) <br> 
#### SRResNet
#### ESRGAN
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/ESRGAN.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/ESRGAN2.png) <br> 
#### EDSR
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/EDSR.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/EDSR2.png) <br> 
#### VDSR
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/VDSR.png) <br> 
#### RFSR
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/RFSR.png) <br> 
#### MZSR
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/MZSR.png) <br> 


## Implement
The code for `VDSR`and `RFSR` are programing on <br>
    Windows 10 <br>
    Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
    NVIDIA GeForce GTX 1080 Ti <br>
The code for `SRGAN`,`SRResNet`,`ESRGAN` and `MZSR` are programing on <br> 
    Ubuntu 20.04
    python 3.6
    Nvidia GeForce 1080Ti 
    CUDA 11.0
    cuDNN v8.0.5.
 


## Testing
After the initial training in order to find the best pretrained model for each method.
Load the TA's program `utils_HW3.py` in the `test.py` to get the submission file.



## Results
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Result2.PNG) <br> 
### Prediction 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW3_Instance_Segmentation/Images/Result.PNG) <br>     


## Make-Submission
Use the `test.py` or `eval.py` to get the final file. <br>
Submit the file `StudentID.json`, to the google drive and  get the mPA scroe from TA. <br>


## Reference
[1] C. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 105-114, doi: 10.1109/CVPR.2017.19.<br> 

[2] WANG, Xintao, et al. Esrgan: Enhanced super-resolution generative adversarial networks. In: Proceedings of the European Conference on Computer Vision (ECCV). 2018. p. 0-0.<br> 

[3] B. Lim, S. Son, H. Kim, S. Nah and K. M. Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Honolulu, HI, 2017, pp. 1132-1140, doi: 10.1109/CVPRW.2017.151.<br> 

[4] J. Kim, J. K. Lee and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 1646-1654, doi: 10.1109/CVPR.2016.182.<br> 

[5] M. Haris, G. Shakhnarovich and N. Ukita, "Deep Back-Projection Networks for Super-Resolution," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 1664-1673, doi: 10.1109/CVPR.2018.00179.<br> 

[6] J. W. Soh, S. Cho and N. I. Cho, "Meta-Transfer Learning for Zero-Shot SuperResolution," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 3513-3522, doi: 10.1109/CVPR42600.2020.00357.<br> 

[7] SRGAN ESRGAN EDSR： https://github.com/xinntao/BasicSR <br> 

[8] VDSR： https://github.com/twtygqyy/pytorch-vdsr <br> 

[9] RFSR：https://github.com/jshermeyer/RFSR?fbclid=IwAR3lEu7Vy0aPzrrNBpTMcSYzpdBLGiEBSq2M5T8QHzFVwQeiilFxU3isO2k <br> 

[10] MZSR：https://github.com/JWSoh/MZSR?fbclid=IwAR2p1SAJMIWSxb7cUSLK75wa9K3qVZZrMi4WCRYGvfBASPVlZoESv2VOGc4 <br> 
