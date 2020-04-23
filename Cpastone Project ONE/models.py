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
        

        
        # The network will be build with 4 convolutional layers (Each surround with a maxpool) and three fully conected layers
        # between the last fully conected layers a droput layer is implemented to provide from overfitting 
        
        
        ##NotImplemented## calculation for changing incoming sizes is done in the calc_input_features funciton!
        ##NotImplemented## Calculation of all the needed Input and Output sizes for the given network and an input of a squared image
        ##NotImplemented## first_con_output_size = calc_input_features()
        
        
        
        
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)
        ## More layers can be added after the network runs properly
        #self.conv3 = nn.Conv2d(in_channels=x,out_channels=x,kernel_size=x)
        #self.conv4 = nn.Conv2d(in_channels=x,out_channels=x,kernel_size=x)
        
        # fully conected linear layers
        self.lin1 = nn.Linear(in_features=18496,out_features=300)
        self.lin2 = nn.Linear(in_features=300,out_features=136)
        
        ## More layers can be added after the network runs properly
        #self.lin3 = nn.Linear(in_features=x,out_features=136)
        
        
        # maxPool to reduce the size of the processe data 
        self.pool1 = nn.MaxPool2d(4,4)
        self.pool2 = nn.MaxPool2d(3,3)
        
        
        # droput layer 
        self.lin2_drop = nn.Dropout(p=0.45)
        
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        ###notation -> (Channels,Height,Width)
        
        ###input_conv1-> (1,224,224)
        ###output_conv1-> (32,220,220) == input_maxPool
        ###output_maxPool -> (32,55,110)       
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        
        ###input_conv2-> (32,55,55)
        ###output_conv2-> (64,51,51) == input_maxPool
        ###output_maxPool -> (64,17,17)  
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        
        # Reduce dimensions
        x = x.view(x.size(0),-1)
        #print(x.shape)
        
        
        x = F.relu(self.lin1(x))
        #print(x.shape)
        x = self.lin2_drop(x)
        x = self.lin2(x)
        #print(x.shape)
       
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    