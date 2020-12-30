# Final_project
link：https://www.kaggle.com/c/severstal-steel-defect-detection <br>
code for the 2% accuracy in Kaggle Severstal: Steel Defect Detection Challenge.

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
    

## Dataloader
Change the path which is in the `dataloader.py` and `main.py`.
    
    pd.read_csv('./training_labels.csv')
    train_dataset = RetinopathyLoader('./training_data', 'train', augmentation=augmentation)
Validation and testing data are ues the same method to load <br>

## Method
Taking the pretrained model for desent121 <br>
densenet121_pretrained = torchvision.models.__dict__['densenet{}'.format(121)](pretrained=True)
If you want to change the deeper model, you can change the model name or the num of `.format(121).`
You can try the efficientnet with the pretrained b-7, though it costs many memory, the performance is better than densenet.


Refer to the github：https://github.com/khornlund/severstal-steel-defect-detection <br>

### Data_augmentation
     transforms.Normalize(
     mean=(0.5, 0.5, 0.5),std=(0.5,0.5,0.5)),
     transforms.RandomAffine(30, translate=None,scale=None,shear=None,resample=False,fillcolor=0),
     transforms.TenCrop(480,vertical_flip=False),
     transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip()


### Optimizer

         
                        
## experiment
After processing every epoch, you need to save the model, in oder to avoid the model breaking. <br>   
You can load model as the code： 

    densenet121_pretrained_model = densenet121_pretrained.to(device)
    densenet121_pretrained_model.load_state_dict(torch.load('model.pkl'))

## result
Using this code to do the testing. <br>

    validation_correct, y_truth, y_predict = evalModels(models,'test', validation_loader = DataLoader(test_dataset, batch_size=8), return_predict_y=True)
    
## Make-Submission
Submit the file `pred.csv`, to the kaggle  get the testing accuracy <br>

## Reference
[1] H. Noh, S. Hong and B. Han, "Learning Deconvolution Network for Semantic Segmentation," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 1520-1528, doi: 10.1109/ICCV.2015.178.<br> 

[2] H. Zhang et al., "Context Encoding for Semantic Segmentation," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7151-7160, doi: 10.1109/CVPR.2018.00747.<br> 

[3] RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. p. 234-241.<br> 

[4] TAN, Mingxing; LE, Quoc V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.<br> 

[5] SIMONYAN, Karen; ZISSERMAN, Andrew. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.<br> 

[6] L. Zhou, C. Zhang and M. Wu, "D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, 2018, pp. 192-1924, doi: 10.1109/CVPRW.2018.00034.<br> 
[7] Reference codel：https://github.com/khornlund/severstal-steel-defect-detection <br> 
[8] Kaggle’s competition：https://www.kaggle.com/c/severstal-steel-defect-detection <br> 
