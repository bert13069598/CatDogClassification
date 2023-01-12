from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

import tqdm as tqdm
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)
    plt.show()


# data check
raw_data_path = './data/archive/train/'


# files = glob(os.path.join(path, '*.jpg'))

def filecheck(path, name):
    files = []
    tmp = os.listdir(path)
    for file in tmp:
        if '.jpg' in file:
            files.append(file)
    print(f'{name} num of images {len(files)}')
    return files


files = filecheck(raw_data_path, 'Total')

no_of_images = len(files)
shuffle = np.random.permutation(no_of_images)
val_num = int(no_of_images * 0.8)
print(f'Train num of images : {val_num}')
print(f'Valid num of images : {no_of_images - val_num}')

# create train & validation folder
path = './data/dogsandcats/'
os.makedirs('./data/dogsandcats', exist_ok=True)
for mode in ['train', 'valid']:
    for folder in ['dog/', 'cat/']:
        os.makedirs(os.path.join(path, mode, folder), exist_ok=True)

# put data in train & validation folder
for i in shuffle[:val_num]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    shutil.copyfile(os.path.join(raw_data_path, files[i]), os.path.join(path, 'train', image))
    os.rename(os.path.join(raw_data_path, files[i]), os.path.join(path, 'train', folder, image))

for i in shuffle[val_num:]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    shutil.copyfile(os.path.join(raw_data_path, files[i]), os.path.join(path, 'valid', image))
    os.rename(os.path.join(raw_data_path, files[i]), os.path.join(path, 'valid', folder, image))

filecheck('./data/dogsandcats/train/', 'Total train')
filecheck('./data/dogsandcats/train/cat', 'Train cat')
filecheck('./data/dogsandcats/train/dog', 'Train dog')
filecheck('./data/dogsandcats/valid/', 'Total valid')
filecheck('./data/dogsandcats/valid/cat', 'Valid cat')
filecheck('./data/dogsandcats/valid/dog', 'Valid dog')

# Check if GPU is present
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
    print('cuda available')

# load data into PyTorch tensors
simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('data/dogsandcats/train/', simple_transform)
valid = ImageFolder('data/dogsandcats/valid/', simple_transform)
print(train.class_to_idx)
print(train.classes)

# print sample img
# imshow(train[0][0])

# Create data generators
train_data_gen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=3)
valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=3)

train_features, train_labels = next(iter(train_data_gen))
print(f"Data batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
print(f'dataset_sizes: {dataset_sizes}')

dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}

# Create a network
# model_ft = models.resnet18(pretrained=True)
model_ft = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

# print(model_ft)

# Loss and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
