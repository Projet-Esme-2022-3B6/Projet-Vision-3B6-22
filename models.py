## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
       

        
        
        self.conv1 = nn.Conv2d(1, 68, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(68, 136, 5)
        
        self.conv3 = nn.Conv2d(136, 272, 5)
        
        self.conv4 = nn.Conv2d(272, 544, 2)
       
        self.conv5 = nn.Conv2d(544, 1088, 1)

        self.conv6 = nn.Conv2d(1088, 2176, 1)

        
        self.fc1 = nn.Linear(8704, 2176)
        self.fc2 = nn.Linear(2176, 544)
        self.fc3 = nn.Linear(544, 136)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
    
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x= self.conv1(x)
        x=F.relu(x)
        x = self.pool(x)
        
        x= self.conv2(x)
        x=F.relu(x)
        x = self.pool(x)
        
        x= self.conv3(x)
        x=F.relu(x)
        x = self.pool(x)
        
        x= self.conv4(x)
        x=F.relu(x)
        x = self.pool(x)
        
        x= self.conv5(x)
        x=F.relu(x)
        x = self.pool(x)
        
        x= self.conv6(x)
        x=F.relu(x)
        x = self.pool(x)
        
        
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x=self.dropout1(x)
        x = F.relu(self.fc2(x))
        x=self.dropout2(x)        
        x = self.fc3(x)
        

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
