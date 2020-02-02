from my_modules import CNNDatasetTransformer,VanillaCNN,ImagesDataset,compute_mean_std,test
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


images_dataset = ImagesDataset('./data/meta_cnn.csv')

num_threads = 4
batch_size = 128
valid_ratio = 0.2

nb_train = int((1.0 - valid_ratio) * len(images_dataset))
nb_valid =  int(valid_ratio * len(images_dataset))

train_dataset, valid_dataset = dataset.random_split(images_dataset, [nb_train, len(images_dataset) - nb_train])

normalizing_dataset = CNNDatasetTransformer(train_dataset, transforms.ToTensor())
normalizing_loader = torch.utils.data.DataLoader(dataset=normalizing_dataset,batch_size=batch_size,num_workers=num_threads)
mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

#data_transforms = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: (x - mean_train_tensor)/std_train_tensor)])
data_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = CNNDatasetTransformer(train_dataset, data_transforms)
valid_dataset = CNNDatasetTransformer(valid_dataset, data_transforms)

#train_dataset = CNNDatasetTransformer(train_dataset, transforms.ToTensor())
#valid_dataset = CNNDatasetTransformer(valid_dataset, transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_threads)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_threads)

print("The train set contains {} features, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} features, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

model = VanillaCNN((1,80,80),3,10**-5)
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
        loss = f_loss(outputs, targets)
        # Backward and optimize
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()
        optimizer.zero_grad()
        loss.backward()
        model.penalty().backward()
        optimizer.step()
    print('correct : {} \r'.format(correct))

for t in range(5):
    print("Epoch {}".format(t))
    train(model, train_loader, f_loss, optimizer, device)
    _,correct_exp = test(model,valid_loader,f_loss,device)
    print("Validation correct : {}".format(correct_exp))


torch.save(model.state_dict(),"data/model_cnn2.pt")
torch.save(mean_train_tensor,"data/mean_tensor.pt")
torch.save(std_train_tensor,"data/std_tensor.pt")
print("The mean is:{} and the variance is :{}".format(mean_train_tensor.size(),std_train_tensor.size()))