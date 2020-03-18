import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, input_size, output_size):
        
        super(Network,self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16,8)
        self.out = nn.Linear(8, output_size)
        
        
    def forward(self,x):
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.out(x)
        return x


