import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, RandomCrop, ColorJitter, RandomAffine, RandomApply, RandomChoice, RandomErasing, GaussianBlur
from torch.utils.data.dataset import ConcatDataset

#####   Dataset class For Train, Val #####

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True, val = None):
        super(AgeDataset, self).__init__()
        self.val = val
        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.val:
            return Compose([
                Resize(n_px),
                RandomApply([RandomHorizontalFlip()], p=0.5),  # Random horizontal flip with 50% probability
                RandomRotation(degrees=15),  # Random rotation up to 15 degrees
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),  # Random color jitter
                RandomApply([GaussianBlur(kernel_size=3)], p=0.1),  # Random Gaussian blur with 10% probability
                ToTensor(),  # Convert PIL image to PyTorch tensor
                Normalize(mean=mean, std=std)  # Normalize with mean and standard deviation
            ])
        
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])
        

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img
        

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img
    def __len__(self):
        return len(self.files)
    

##### Reading CSVs and making Loaders #####
    
train_path = 'faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)

val_dataset = AgeDataset(train_path, train_ann, train=True, val = True)

train_dataset = ConcatDataset([train_dataset, val_dataset])

test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

import matplotlib.pyplot as plt

##### Function for display images #####

def show_images(images, titles):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            axs[i, j].imshow(images[idx].permute(1, 2, 0))  # Permute dimensions for matplotlib
            axs[i, j].set_title(titles[idx])
            axs[i, j].axis('off')
    plt.show()

images = []
titles = []
for i in range(10):
    image, age = val_dataset[i]
    images.append(image)
    titles.append(f"Age: {age}")

# Display images
print(images[0].shape)
show_images(images, titles)


@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

##### Model Defination #####
model = torchvision.models.resnet18(pretrained=True)

num_features = model.fc.in_features
intermediate_features = 256  
model.fc = nn.Sequential(
    nn.Linear(num_features, intermediate_features), # first layer
    nn.ReLU(inplace=True),  # ReLU
    nn.Linear(intermediate_features, 64),  # second layer
    nn.ReLU(inplace=True),  # ReLU
    nn.Linear(64, 1)  # third layer
)

##### Freezed half parameters #####

total_layers = len(list(model.parameters()))
layers_to_freeze = total_layers // 2
print(total_layers, layers_to_freeze)
frozen_layers = 0

for param in model.parameters():
    param.requires_grad = False
    frozen_layers += 1
    if frozen_layers >= layers_to_freeze:
        break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


##### Model Training #####

criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
num_epochs = 10

for epoch in range(5):
    model.train()
    running_loss = 0.0
    
    for images, ages in tqdm(train_loader, total = len(train_loader),desc = "training"):
        images, ages = images.to(device), ages.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages.unsqueeze(1).float()) 
        
        abs_diff = torch.abs(outputs.squeeze(1) - ages)
        mask = abs_diff > 0
        masked_loss = torch.where(mask, loss, torch.zeros_like(loss))
        masked_loss.mean().backward()
        optimizer.step()
        
        running_loss += masked_loss.mean().item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
    # evaluate model using val dataset
    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for images, ages in tqdm(val_loader, total=len(val_loader), desc="Validation"):
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, ages.unsqueeze(1).float())
            val_running_loss += val_loss.item() * images.size(0)
    
    val_epoch_loss = val_running_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")


##### Save model #####
torch.save(model.state_dict(), 'trained_model.pth')


###### SUBMISSION CSV FILE #####

@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

preds = predict(test_loader, model)

submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('baseline.csv',index=False)
