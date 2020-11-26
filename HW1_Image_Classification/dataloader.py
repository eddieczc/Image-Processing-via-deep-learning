#dataloader
""" load the dataset and do some processing"""
import os
import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms
import PIL

label_dict = {}
label_list = []
label_id = []
label_label = []
validation_list = []

#get the dictionary
def get_key (select, value):
    """find the same value to process"""
    return [k for k, v in select.items() if v == value]
#transform the label to label
def predict_data(path):
    """Create the csv file which can be submited in kaggle"""
    imagelist = os.listdir(path)
    img_name = []
    for i in imagelist:
        img_name.append(i.split('.')[0])
    return img_name
#save the file
def stor_file(filename,id_value,labe_value):
    """Save the Csv file"""
    dataframe = pd.DataFrame({'id':id_value,'label':labe_value})
    dataframe.to_csv(filename,index=False,sep=',')
#select the mode
def get_data(mode):
    """ Totally we have tgree modes can be choosen"""
    if mode == 'train':
        ground_truth = pd.read_csv('./training_labels.csv')
        img = ground_truth['id']
        label = ground_truth['label']
        label_idx = 0
        label_len = len(label)
        for idx in range(0,label_len):
            if label[idx] not in label_dict.values():
                label_dict['{}'.format(label_idx)]='{}'.format(label[idx])
                label_list.append(label_idx)
                label_id.append(label_idx)
                label_label.append(label[idx])
                label_idx = label_idx + 1
            else:
                label_classifer= int(get_key(label_dict,label[idx])[0])
                label_list.append(label_classifer)
        label_final= np.array(label_list)
        stor_file('classifer.csv',label_id,label_label)
    if mode == 'validation':
        ground_truth = pd.read_csv('./validation/validation_labels.csv')
        classifer = pd.read_csv('classifer.csv')
        img = ground_truth['id']
        label = ground_truth['label']
        for i in label.values:
            idx = 0
            for j in classifer['label'].values:
                if i == j:
                    validation_list.append(classifer['id'][idx])
                idx = idx +1
        label_final= np.array(validation_list)
    if mode == 'test':
        ground_truth = pd.read_csv('testing_labels.csv')
        img = ground_truth['id']
        label = ground_truth['label']
        label_final = label.values
    return np.squeeze(img.values), np.squeeze(label_final)

#transform the image
class RetinopathyLoader(data.Dataset):
    """data augmentation"""
    def __init__(self, root, mode, augmentation=None):

        self.root = root
        self.img_name, self.label = get_data(mode)
        self.mode = mode
        trans_augmentation = []
        if augmentation is True:
            trans_augmentation = trans_augmentation + augmentation

        trans_augmentation = trans_augmentation + [transforms.ToTensor()]
        self.trans_res = transforms.Compose(trans_augmentation)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        num = str(self.img_name[index])
        path = os.path.join(self.root, num.zfill(6) + '.jpg')
        img = PIL.Image.open(path).convert('RGB')
        img_resize = img.resize((448, 448))
        img_data = self.trans_res(img_resize)
        label= self.label[index]
        return img_data,label
