# main
"""densenet121"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torchvision import transforms
from IPython.display import clear_output
import pyprind
import matplotlib.pyplot as plt
from dataloader import RetinopathyLoader
from dataloader import predict_data, stor_file
from sam import SAM
from autoaugment import ImageNetPolicy

# Entropy method
class LabelSmoothingCrossEntropy(nn.Module):
    """replace CrossEntropy"""

    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def computational(
        name,
        train_dataset,
        validation_dataset,
        models,
        epoch_size,
        batch_size,
        learning_rate,
        optimizer=optim.SGD,
        optimizer_option={
            'momentum': 0.9,
            'weight_decay': 0.0005},
    criterion=LabelSmoothingCrossEntropy(),
        show=True):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False)

    accs = {
        **{key + "_train": [] for key in models},
        **{key + "_validation": [] for key in models}
    }
    # optimizers's setting
    optimizers = {
        key: SAM(value.parameters(), optimizer, lr=learning_rate, **optimizer_option)
        for key, value in models.items()
    }

    # training
    for epoch in range(epoch_size):
        bar = pyprind.ProgPercent(
            len(train_dataset),
            title="Training epoch {} : ".format(epoch))

        train_correct = {key: 0.0 for key in models}
        validation_correct = {key: 0.0 for key in models}
        # training multiple model
        for model in models.values():
            model.train()

        for idx, data in enumerate(train_loader):
            x, y = data

            y = list(y)
            y = torch.Tensor(y)
            y.requires_grad_()
            inputs = x.to(device)
            labels = y.to(device).long().view(-1)

            for key, model in models.items():
                outputs = model(inputs)
                # print('out',outputs)
                loss = criterion(outputs, labels)
                loss.backward()

            for optimizer in optimizers.values():
                optimizer.first_step(zero_grad=True)

            for key, model in models.items():
                outputs = model(inputs)
                # print('out',outputs)
                loss = criterion(outputs, labels)
                loss.backward()

                cur_correct = (
                    torch.max(outputs, 1)[1] == labels
                ).sum().item()

                train_correct[key] += cur_correct

            for optimizer in optimizers.values():
                optimizer.second_step(zero_grad=True)

            bar.update(batch_size)

        # save model
        for key, model in models.items():
            model_name = str(key) + str("_") + str(epoch) + str(".pkl")
            model_path = "figure_norelu/" + model_name
            torch.save(model.state_dict(), model_path)

        # validation multiple model
        validation_correct = eval_models(
            models, 'validation', validation_loader,
        )

        for key, value in train_correct.items():
            accs[key + "_train"] += [(value * 100.0) / len(train_dataset)]

        for key, value in validation_correct.items():
            accs[key + "_validation"] += [(value * 1.0) / 1]

        if show:
            clear_output(wait=True)
            makefigure(
                title='Epoch [{:4d}]'.format(epoch),
                **accs
            )
        print("train_Accuracy:{}".format(train_correct))
        print("validation_Accuracy:{}".format(validation_correct))
        # epoch end
        torch.cuda.empty_cache()

    return accs


results = {}

# count the accuracy
def eval_models(models, mode, validation_loader, return_predict_y=False):
    """testing"""
    validation_correct = {key: 0.0 for key in models}
    if return_predict_y:
        y_pred = {key: torch.Tensor([]).long() for key in models}
        y_true = torch.Tensor([]).long()

    bar = pyprind.ProgPercent(
        len(validation_loader.dataset), title="Testing epoch: ")
    # 繪製測試進度
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for idx, data in enumerate(validation_loader):
            x_in, y_in = data
            inputs = x_in.to(device)
            labels = y_in.to(device)

            if return_predict_y:
                y_true = torch.cat((y_true, y_in.long().view(-1)))

            for key, model in models.items():
                outputs = model(inputs)

                validation_correct[key] += (
                    torch.max(outputs, 1)[1] == labels.long().view(-1)
                ).sum().item()

                if return_predict_y:
                    y_pred[key] = torch.cat(
                        (y_pred[key], torch.max(
                            outputs, 1)[1].to(
                            torch.device('cpu')).long()))

            bar.update(validation_loader.batch_size)

    if mode == 'train':
        for key in validation_correct:
            validation_correct[key] = (
                validation_correct[key] * 1.0) / len(train_dataset)
        print("training_Accuracy:{}".format(validation_correct[key]))
    elif mode == 'validation':
        for key in validation_correct:
            validation_correct[key] = (
                validation_correct[key] * 100.0) / len(validation_dataset)
        print("validation_Accuracy:{}".format(validation_correct[key]))
    else:
        for key in validation_correct:
            validation_correct[key] = (
                validation_correct[key] * 100.0) / len(test_dataset)
        print("testing_Accuracy:{}".format(validation_correct[key]))

    if return_predict_y:
        return validation_correct, y_true, y_pred
    else:
        return validation_correct

# draw a picture


def makefigure(title='', accline=[87], **kwargs):
    """drawing"""
    fig = plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    for label, data in kwargs.items():
        plt.plot(
            range(1, len(data) + 1), data,
            label=label)

    # let the label on upper left
    plt.legend(
        loc='upper left',
        fancybox=True, shadow=True)

    # make a line which is baseline 87
    if accline:
        plt.hlines(
            accline,
            1,
            len(data) + 1,
            linestyles='--',
            colors=(
                0,
                0,
                0,
                0.5))
    plt.savefig('./figure_norelu/epoch_{}.png'.format(title))

    plt.show()

    return fig


augmentation = [
    ImageNetPolicy(),
    transforms.Normalize(
        mean=(
            0.5,
            0.5,
            0.5),
        std=(
            0.5,
            0.5,
            0.5)),
    transforms.RandomAffine(
        30,
        translate=None,
        scale=None,
        shear=None,
        resample=False,
        fillcolor=0),
    transforms.TenCrop(
        480,
        vertical_flip=False),
    transforms.ColorJitter(
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
]

# 前處理設定
train_dataset = RetinopathyLoader(
    './validation/training_data',
    'train',
    augmentation=augmentation)
validation_dataset = RetinopathyLoader(
    './validation/training_data', 'validation')
stor_file('testing_labels.csv', predict_data('./testing_data'), -1)
test_dataset = RetinopathyLoader('./testing_data', 'test')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet121_pretrained = torchvision.models.__dict__[
    'densenet{}'.format(121)](pretrained=True)

densenet121_pretrained.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=448, out_features=356),
    torch.nn.Dropout(0.5, inplace=True),
    torch.nn.Linear(in_features=356, out_features=262),
    torch.nn.Dropout(0.5, inplace=True),
    torch.nn.Linear(in_features=262, out_features=196),
    torch.nn.Dropout(0.5, inplace=True)
)
densenet121_pretrained.drop_rate = 0.5

models_name = {"densenet121_pretrained": densenet121_pretrained.to(device), }

names = 'densenet121'
# Training & Testing
results[names] = computational(
    names,
    train_dataset,
    validation_dataset,
    models_name,
    epoch_size=40,
    batch_size=12,
    learning_rate=15e-4,
    show=True)

makefigure = makefigure(title=names, **results[names])  # read the best accuracy
print('densenet121：pretrain_validation Accuracy:' +
      str(max(results[names]['densenet121_pretrained_validation'])))


validation_correct, y_truth, y_predict = eval_models(
    models, 'test', validation_loader=DataLoader(
        test_dataset, batch_size=8), return_predict_y=True)
csv_pretrained_18 = y_predict['densenet121_pretrained'].numpy()
csv_pretrained_18 = csv_pretrained_18.tolist()
pred_classifer = pd.read_csv('classifer.csv')
pred_test = pd.read_csv('testing_labels.csv')
pred_label = []
pred_id = []
for i in csv_pretrained_18:
    for j in pred_classifer['id']:
        if i == j:
            pred_label.append(pred_classifer['label'][j])
pred_id = []
for i in pred_test['id']:
    pred_id.append('\t' + str(i).zfill(6))


pred_test = pd.read_csv('testing_labels.csv')
pred_test['label'] = pred_label
stor_file('pred.csv', pred_id, pred_test['label'])
