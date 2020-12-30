# CS_T0828_HW1
link：https://www.kaggle.com/c/cs-t0828-2020-hw1 <br>
code for the fifth accuracy in Kaggle Severstal: Steel Defect Detection Challenge.

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

    conda create -n Classification python=3.6
    conda install pytorch -c pytorch
    conda install torchvision -c pytorch

## Dataloader
Change the path which is in the `dataloader.py` and `main.py`.
    
    pd.read_csv('./training_labels.csv')
    train_dataset = RetinopathyLoader('./training_data', 'train', augmentation=augmentation)
Validation and testing data are ues the same method to load <br>

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
