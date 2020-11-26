# CS_T0828_HW2
code for the fifth accuracy in Digits detection Classification Challenge. <br>

## Hardware
● Ubuntu 20.04 <br>
● Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz <br>
● NVIDIA GeForce GTX 1080 Ti <br>

## Introduction and details
There is the outlines in this compitions <br>
1. [Installation](#Installation) <br>
2. [Getting labels](#dataloader) <br>
3. [Model](#Model) <br>
4. [Testing](#Load-Model) <br>
5. [Organising Materials](#testing) <br>
6. [Make-Submission](#Make-Submission)<br>

## Installation
Using Anaconda and pytorch to implement this method.

    conda create -n Classification python=3.6
    conda install pytorch -c pytorch
    conda install torchvision -c pytorch

## Dataloader
Change the path which is in the `get_labels.py`.
        f = h5py.File('./train/digitStruct.mat','r')
        Image.open('./train/'+IMG_NAME)
In order to get the ground truth. <br>

![image](https://github.com/eddieczc/Image-Processing-via-deep-learning/blob/master/HW2_Digits%20detection/Images/labels.png)

## Model
Taking the pretrained model for desent121 <br>
densenet121_pretrained = torchvision.models.__dict__['densenet{}'.format(121)](pretrained=True)
If you want to change the deeper model, you can change the model name or the num of `.format(121).`
You can try the efficientnet with the pretrained b-7, though it costs many memory, the performance is better than densenet.

    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b7')
Refer to the github：https://github.com/lukemelas/EfficientNet-PyTorch <br>

### Data_augmentation
     ImageNetPolicy(),
     transforms.Normalize(
     mean=(0.5, 0.5, 0.5),std=(0.5,0.5,0.5)),
     transforms.RandomAffine(30, translate=None,scale=None,shear=None,resample=False,fillcolor=0),
     transforms.TenCrop(480,vertical_flip=False),
     transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip()

I put the AutoAugment in the Data_augmentationu which is `ImageNetPolicy()` <br>
Refer to the github： https://github.com/DeepVoltaire/AutoAugment <br>

### Optimizer
    optimizer=optim.SGD,
    optimizer_option={'momentum': 0.9,'weight_decay': 0.0005},
    SAM(value.parameters(), optimizer, lr=learning_rate, **optimizer_option)
   
It let the accuracy increase a lot, when my model is stuck. <br>
Refer to the github： https://github.com/davda54/sam <br>            
                        
## Load-Model
After processing every epoch, you need to save the model, in oder to avoid the model breaking. <br>   
You can load model as the code： 

    densenet121_pretrained_model = densenet121_pretrained.to(device)
    densenet121_pretrained_model.load_state_dict(torch.load('model.pkl'))

## testing
Using this code to do the testing. <br>

    validation_correct, y_truth, y_predict = evalModels(models,'test', validation_loader = DataLoader(test_dataset, batch_size=8), return_predict_y=True)
    
## Make-Submission
Submit the file `pred.csv`, to the kaggle  get the testing accuracy <br>
