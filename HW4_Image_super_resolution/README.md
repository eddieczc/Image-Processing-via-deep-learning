# CS_T0828_HW4
code for the first psnr in  Image super resolution Challenge. <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/Examples_LR.png) <br> 

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
### SRGAN
SRGAN uses perceptual loss (perceptual loss) and confrontation loss (competition loss) to improve the realism of the restored picture. Perceptual loss is the feature extracted by the convolutional neural network. By comparing the difference between the features of the generated image and the target image after the convolutional neural network, the generated image and the target image are semantically and styled. More similar.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/SRGAN1.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/SRGAN2.png) <br> 
### SRResNet
The generation network part (SRResNet) part contains multiple residual blocks. Each residual block contains two 3×3 convolutional layers. The convolutional layer is followed by batch normalization (BN) and PReLU as the activation function. Two 2×sub-pixel convolution layers are used to increase the feature size. The discriminant network part contains 8 convolutional layers. As the number of network layers deepens, the number of features continues to increase, and the feature size continues to decrease. The activation function is selected as LeakyReLU, and finally the prediction is obtained through two fully connected layers and the final sigmoid activation function Probability of being a natural image.<br> 
### ESRGAN
The article makes improvements to these three points: 1. The basic unit of the network is changed from the basic residual unit to Residual-in-Residual Dense Block (RRDB); 2. The GAN network is improved to Relativistic average GAN (RaGAN); 3. Improvements Perceptual domain loss function, using the VGG feature before activation, this improvement will provide sharper edges and more visually consistent results.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/ESRGAN.png) <br> 
The author also used some techniques to train the deep network: 1. Scaling the residual information, that is, multiplying the residual information by a number between 0 and 1, to prevent instability; 2. Smaller initialization, the author found When the variance of the initialization parameters becomes smaller, the residual structure is easier to train.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/ESRGAN2.png) <br> 
### EDSR
Recent research on super-resolution has entered the era of deep convolutional neural networks (DCNN), and residual networks have performed particularly well. In this article, we propose an enhanced super-resolution network (EDSR), which has reached the state-of- The effect of the-art. Our model performs so well because we have removed some unnecessary modules in the convolutional residual network, and we have expanded the size of the model for stable training.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/EDSR.png) <br> 
The structure used by the author is very similar to SRResnet, but bn and most relu are removed (only in the residual block). The final training version has B=32 residual blocks and F=256 channels.And when training *3, *4 models, use *2 pre-training parameters.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/EDSR2.png) <br> 
### VDSR
The VDSR model mainly has the following contributions: 1. It increases the receptive field and has advantages in processing large images, from 13*13 of SRCNN to 41*41. 2. Using residual images for training, the convergence speed becomes faster. Because the residual image is more sparse and easier to converge (another understanding is the low-frequency information of the lr carrier, this information is still trained to the hr image, but the low-frequency information of the hr image and the lr image are similar, which takes a lot of time to train) . 3. Considering multiple scales, a convolutional network can handle multi-scale problems.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/VDSR.png) <br> 
### RFSR
RFSR is an adaptation and major twist on other random-forest super-resolution techniques such as SRF by Schulter et al. Our method uses a random forest regressor with a few simple standard parameters. All parameters were finely tweaked using empirical testing to maximize PSNR scores while maintaining minimal training time (4 hours or less per level of enhancement using 200 million pixels on a 64GB RAM CPU). The technique is trained only using the luminance component from a YCbCr converted image.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/RFSR.png) <br> 
### MZSR
During meta-transfer learning, the external dataset is used, where internal learning is done during meta-test time. From random initial \theta_0, large-scale dataset DIV2K with “bicubic” degradation is exploited to obtain \theta_T. Then, meta-transfer learning learns a good representation \theta_M for super-resolution tasks with diverse blur kernel scenarios. In the meta-test phase, self-supervision within a test image is exploited to train the model with corresponding blur kernel.<br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/MZSR.png) <br> 


## Implement
This homework prohibits all pre-trained models and other training data.<br>
The code for `VDSR`and `RFSR` are programing on <br>
    Windows 10 <br>
    Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
    NVIDIA GeForce GTX 1080 Ti <br>
The code for `SRGAN`,`SRResNet`,`ESRGAN`,`EDSR` and `MZSR` are programing on <br> 
    Ubuntu 20.04
    python 3.6
    Nvidia GeForce 1080Ti 
    CUDA 11.0
    cuDNN v8.0.5.
    
### VDSR：<br>
Tn order to get the annotation for training and testing.<br>
    generate_train.m
    generate_test_mat.m
Using this instruction to train the model.<br>
    python main_vdsr.py --cuda --gpus 0

### RFSR：<br>
    pip install -r Requirements.txt
Simply launch a jupyter notebook instance and open the notebook. <br>

### SRGAN,SRResNet,ESRGAN,EDSR and MZSR：<br>
First, use `create_lr_images.py` to get the LR images.<br>
    img_or.resize((int(w/3), int(h/3)),Image.BICUBIC)
The code for `SRGAN`,`SRResNet`,`ESRGAN` and `EDSR` using the folder `BasicSR`  <br> 
    cd BasicSR
    pip install -r requirements.txt
Then, choose the model you want to train.<br>

    python basicsr/train.py -opt options/train/SRGAN/train_SRGAN_x3.yml
    python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x3.yml
    python basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_x3.yml
    python basicsr/train.py -opt options/train/EDSR/train_EDSR_Lx3.yml
    
The code for `MZSR` using the folder `MZSR` <br> 
Make sure all configurations in config.py are set.<br>
    python main.py --train --gpu [GPU_number] --trial [Trial of your training] --step [Global step]
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/process.png) <br>     
## Testing

### VDSR：<br>
Tn order to get the annotation for training and testing.<br>
    generate_train.m
    generate_test_mat.m
Using this instruction to train the model.<br>
    python eval.py --cuda --dataset HW4

### RFSR：<br>
    pip install -r Requirements.txt
Simply launch a jupyter notebook instance and open the notebook. <br>

### SRGAN,SRResNet,ESRGAN,EDSR and MZSR：<br>
First, use `create_lr_images.py` to get the LR images.<br>
    img_or.resize((int(w/3), int(h/3)),Image.BICUBIC)
The code for `SRGAN`,`SRResNet`,`ESRGAN` and `EDSR` using the folder `BasicSR`  <br> 
    cd BasicSR
Then, choose the model you want to test.<br>

    python basicsr/test.py -opt options/test/SRGAN/test_SRGAN_x3.yml
    python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x3.yml
    python basicsr/test.py -opt options/test/ESRGAN/test_ESRGAN_x3.yml
    python basicsr/test.py -opt options/test/EDSR/test_EDSR_Lx3.yml
    
Load the model you have already trained in folder experiments and you can get the results in folder results.<br>    

The code for `MZSR` using the folder `MZSR` <br> 
Make sure all configurations in config.py are set.<br>
    python main.py --gpu [GPU_number] --inputpath [LR path] --gtpath [HR path] --savepath [SR path]  --kernelpath [kernel.mat path] --model [0/1/2/3] --num [1/10]

## Results of each model
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/table1.PNG) <br> 
### PSNR and the tesults 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/table2.PNG) <br>     


## Make-Submission
Put the 3x upscaling image into the folder which is created by TAs in google drive.  <br>
Getting the PSNR scroe from TA. <br>
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW4_Image_super_resolution/Images/final.PNG) <br>     

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
