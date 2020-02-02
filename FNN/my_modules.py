import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch.nn as nn
import cv2

class FeaturesDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.features = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        output = self.features.iloc[idx,0]
        sample = self.features.iloc[idx,1:]
        sample = np.array([sample])
        output = np.array([output])
        output = output.astype('float')
        sample = sample.astype('float').reshape(-1,21)
        if(self.transform):
            sample = self.transform(sample)
        return {'output':output, 'features':sample}

class ImagesDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.meta_data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self,idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        image_name = self.meta_data.iloc[idx,0]
        
        target = self.meta_data.iloc[idx,1]
        img = cv2.imread("data/"+str(image_name),cv2.IMREAD_GRAYSCALE)
        if(self.transform):
            img = self.transform(img)
        return {'img':img, 'target':target}

class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        features = self.base_dataset[index]['features']
        target = self.base_dataset[index]['output']
        return self.transform(features), target

    def __len__(self):
        return len(self.base_dataset)

class CNNDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img = self.base_dataset[index]['img']
        target = self.base_dataset[index]['target']
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)

class FullyConnectedRegularized(nn.Module):

    def __init__(self, input_size, num_classes, l2_reg):
        super(FullyConnectedRegularized, self).__init__()
        self.l2_reg = l2_reg
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, 256)
        self.lin5 = nn.Linear(256, num_classes)


    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) + self.lin2.weight.norm(2) + self.lin3.weight.norm(2))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))
        x = nn.functional.relu(self.lin4(x))
        y = self.lin5(x)
        return y

class VanillaCNN(nn.Module):
    def __init__(self, input_size, num_classes, l2_reg):
        super(VanillaCNN, self).__init__()
        self.l2_reg = l2_reg
        self.conv_classifier1 = nn.Sequential(
            nn.Conv2d(1, 32,
                kernel_size=3,stride=1,padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64,
                kernel_size=3,stride=1,padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        spot_sizes = torch.zeros(input_size).unsqueeze(0)
        out = self.conv_classifier1(spot_sizes)

        out = out.view(out.size()[0], -1)
        nfeatures = out.size()[1]

        self.lin1 = nn.Linear(nfeatures, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, num_classes)

    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) + self.lin2.weight.norm(2) + self.lin3.weight.norm(2))

    def forward(self, x):
        x = self.conv_classifier1(x)
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))
        y = self.lin4(x)
        return y
def compute_mean_std(loader):
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)
    std_img[std_img == 0] = 1

    return mean_img, std_img


def test(model,loader,f_loss,device):
    with torch.no_grad():
        model.eval()
        N = 0
        tot_loss, correct = 0.0,0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            N += inputs.shape[0]

            tot_loss += inputs.shape[0] * f_loss(outputs,targets).item()

            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N