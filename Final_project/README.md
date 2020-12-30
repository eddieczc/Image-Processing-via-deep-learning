# Final_project
link：https://www.kaggle.com/c/severstal-steel-defect-detection <br>
code for the 2% accuracy in Kaggle Severstal: Steel Defect Detection Challenge.
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/example.PNG) <br> 
## Hardware
● Windows 10 <br>
● Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
● NVIDIA GeForce GTX 1080 Ti <br>

## Introduction and details
There is the outlines in this competition <br>
1. [Installation](#Installation) <br>
2. [Dataloader](#Dataloader) <br>
3. [Method](#Model) <br>
4. [experiment](#experiment) <br>
5. [result](#result) <br>
6. [Make-Submission](#Make-Submission) <br>
7. [Reference](#Reference) <br>

## Installation
Using Anaconda and pytorch to implement this method.

    conda create -n steeldefect python=3.6
    conda install pytorch -c pytorch
    conda install torchvision -c pytorch
    pip install -r requirements.txt
    

## Datatrans
Change the path which is in the `datatrans.py`. <br>   
    pd.read_csv('./train.csv')
Use the `datatrans.py` to get the `original_train.csv`<br>
This file is more convenient for training. <br>

## Method
    python train.py
If you want to read the model for training, please remove the annotation of the main program and change the path <br> 
    # Initialize mode and load trained weights
    #ckpt_path = "./model/model34.pth"
    #device = torch.device("cuda")

    #state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #model.load_state_dict(state["state_dict"])    
    
### Architecture
#### UNET
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/UNET.png) <br> 
##### The details of U-net
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/layer.PNG) <br> 
#### Efficientnet 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/eff_1.png) <br> 
Refer to the github：https://github.com/lukemelas/EfficientNet-PyTorch <br>
#### D-LinkNet plus the conv1
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/over.PNG) <br> 
##### The details of D-LinkNet
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/details.png) <br> 
Refer to the github：https://github.com/khornlund/severstal-steel-defect-detection <br>

### Data_augmentation
     transforms.Normalize(
     mean=(0.5, 0.5, 0.5),std=(0.5,0.5,0.5)),
     transforms.RandomAffine(30, translate=None,scale=None,shear=None,resample=False,fillcolor=0),
     transforms.TenCrop(480,vertical_flip=False),
     transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip()

### Loss function
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/loss.PNG) <br> 

                           
## experiment
### Number of training materials
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/dataset.PNG) <br> 
### The Training data analysis
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/dataset_3.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/dataset_4.png) <br> 
### Comparison with different method
#### Comparison of using different architectures
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_1.png) <br> 
#### With/without Conv1 and Data augmentation
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_2.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_3.png) <br> 
#### Comparison of using different loss function
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_4.png) <br> 
##### The Loss plot
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/bceloss.png) <br> 
##### The IoU Score plot
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/IOU.png) <br> 
#### Comparison of using different optimizer
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_5.png) <br> 
##### Optimizer’s comparison ( SGD and Adam )
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/table_6.png) <br> 
### Ablation studies
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/ablation.png) <br> 

## result
Using this code to do the testing. <br>
    python test.py
#### Our group’s results
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/res.PNG) <br> 

#### Score ranking in the Leaderboard
A total of 2431 teams participated in this competition. After comparing the values on the leaderboard, our results are about 2% of the total.<br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/res_2.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/res_1.png) <br> 
![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/Final_project/images/res_3.png) <br> 

## Make-Submission
Submit the file `submission.csv`, to the kaggle  get the testing accuracy <br>

## Reference
[1] H. Noh, S. Hong and B. Han, "Learning Deconvolution Network for Semantic Segmentation," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 1520-1528, doi: 10.1109/ICCV.2015.178.<br> 

[2] H. Zhang et al., "Context Encoding for Semantic Segmentation," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7151-7160, doi: 10.1109/CVPR.2018.00747.<br> 

[3] RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. p. 234-241.<br> 

[4] TAN, Mingxing; LE, Quoc V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.<br> 

[5] SIMONYAN, Karen; ZISSERMAN, Andrew. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.<br> 

[6] L. Zhou, C. Zhang and M. Wu, "D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, 2018, pp. 192-1924, doi: 10.1109/CVPRW.2018.00034.<br> 

[7] Reference codel：https://github.com/khornlund/severstal-steel-defect-detection <br> 

[8] Kaggle’s competition：https://www.kaggle.com/c/severstal-steel-defect-detection <br> 
