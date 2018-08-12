from torch import nn
import torch
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, input_size, output_size , layers=1):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.layers = layers
        self.lstm = nn.LSTM(input_size, output_size , num_layers=self.layers)
        self.out = nn.Linear(output_size,input_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, output):
        input = input.view(1,1,-1)
        output, hidden = self.lstm(input, output)
        linear = self.out(output)
        output = self.softmax(linear)
        return output, hidden

    def initHidden(self , mode):
        if(mode.lower() == "lstm" ):
            return (torch.zeros(self.layers,1, self.hidden_size) ,torch.zeros(self.layers,1, self.hidden_size))
        else:
            return torch.zeros(1, 1, self.hidden_size)