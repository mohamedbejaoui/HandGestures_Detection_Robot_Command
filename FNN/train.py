from my_modules import FeaturesDataset,DatasetTransformer,FullyConnectedRegularized
from torch.utils.data import DataLoader,dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--classes",help="number of classes for the classification",type=int)
args = parser.parse_args()

hand_features = FeaturesDataset('./data/all_data.csv')

num_threads = 4
batch_size = 128
valid_ratio = 0

nb_train = int((1.0 - valid_ratio) * len(hand_features))
nb_valid =  int(valid_ratio * len(hand_features))

train_dataset, valid_dataset = dataset.random_split(hand_features, [nb_train, len(hand_features) - nb_train])

train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_threads)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_threads)

print("The train set contains {} features, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} features, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

model = FullyConnectedRegularized(21,args.classes,10**-5)
f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
device = torch.device('cpu')

def train(model, loader, f_loss, optimizer, device):
    model.train()
    correct = 0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        targets = targets.long()
        outputs = model(inputs)
        targets = targets.squeeze(1)
        loss = f_loss(outputs, targets)
        # Backward and optimize
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('correct : {} \r'.format(correct))

for t in range(15):
    print("Epoch {}".format(t))
    train(model, train_loader, f_loss, optimizer, device)


torch.save(model.state_dict(),"data/model.pt")