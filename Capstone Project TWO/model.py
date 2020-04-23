import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # add BatchNormalization layer
        self.bn1 = nn.BatchNorm1d(num_features=embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        #print(features)
        #tensor_max_value = torch.max(features)
        #print(f'Max Values is {tensor_max_value}')
        
        features = self.bn1(features)
        #print(features)
        #tensor_max_value = torch.max(features)
        #print(f'Max Values is {tensor_max_value}')
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
          
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers
        
        
        #define layers
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=1,batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size,out_features=vocab_size)
        self.embedding_words = nn.Embedding(vocab_size,embed_size)
        
      
    
    def forward(self, features, captions):
        
        
        
        # embed captions to desired length and usequeeze features tensor
        captions = self.embedding_words(captions)
        features = features.unsqueeze(1)
        
        
        # concatenate feature and caption tensors
        inputs = torch.cat((features,captions[:,:-1]),dim=1)
        output, _ =self.lstm(inputs)
      
        output = self.lin(output)
        #output = F.softmax(self.lin(output),dim=1)
        
                
        return output
                                 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence = []
        
        for count in range(max_len):
            output, states = self.lstm(inputs,states)
            tokens   = self.lin(output)
            value, index    = torch.max(tokens,dim=2)
            sentence.append(index.item())
            
            # Update inputs
            inputs =self.embedding_words(index)
            
            
            
                        
                    
            
            
        return sentence
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        