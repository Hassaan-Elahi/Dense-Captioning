import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size , layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.LSTM = nn.LSTM(input_size, hidden_size , num_layers=self.layers , bidirectional=True)
        self.linearOutput = nn.Linear(hidden_size *2 , hidden_size)
        
    def forward(self, input, hidden):
        input = input.view(1 , 1 , -1)
        output, hiddenr = self.LSTM(input, hidden)
        output = self.linearOutput(output)
        return output, hiddenr

    def initHidden(self , mode):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        
        if(mode.lower() == "lstm"):
            return ((torch.zeros(2 * self.layers, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size)))
        else:
            return torch.zeros(2 * self.layers, 1, self.hidden_size)