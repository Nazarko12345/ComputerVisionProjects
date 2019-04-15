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
        self.conv = nn.Sequential(nn.Conv2d(1,32,3),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                              nn.Conv2d(32,64,3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
                              nn.Conv2d(64,128,3),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
                              nn.Conv2d(128,256,3),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2))
        
        self.lin_output = 12*12*256
        self.linear = nn.Sequential(nn.Linear(self.lin_output,4000),nn.BatchNorm1d(4000),nn.Dropout(0.3),nn.ReLU()
                                            ,nn.Linear(4000,1600),nn.BatchNorm1d(1600),nn.Dropout(0.2),nn.ReLU(),
                                               nn.Linear(1600,136))
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x =self.conv(x)
        res = self.linear(x.view(x.size(0),-1))        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return res

