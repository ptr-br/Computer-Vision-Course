
import torch
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
              
        
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5)
        #self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=8)
        
        # fully conected linear layers
        self.lin1 = nn.Linear(in_features=4608, out_features=2500)
        self.lin2 = nn.Linear(in_features=2500,out_features=1000)
        self.lin3 = nn.Linear(in_features=1000,out_features=136)
                
        
        # maxPool to reduce the size of the processe data 
        self.pool = nn.MaxPool2d(3,3)
        
        
        
        # droput layer 
        self.lin_drop = nn.Dropout(p=0.4)
        #self.conv_drop= nn.Dropout(p=0.3)

       
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        #x = self.conv_drop(x)
        #x = self.pool(F.relu(self.conv4(x)))
       
        # Reduce dimensions
        x = x.view(x.size(0),-1)  
        # linear layers
        x = self.lin_drop(x)
        x = F.relu(self.lin1(x))   
        x = self.lin_drop(x)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        

        # a modified x, having gone through all the layers of your model, should be returned
        return x