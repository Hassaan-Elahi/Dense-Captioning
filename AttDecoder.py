from torch import nn
import torch
import torch.nn.functional as F
MAX_LENGTH = 50

class AttnDecoderRNN(nn.Module):
    def __init__(self, VocabSizeForEmbedding , hidden_size, featureSize, dropout_p=0.1 ,max_length = MAX_LENGTH ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size # hidden units
        self.VocabSizeForEmbedding = VocabSizeForEmbedding # size of vocabulary
        self.dropout_p = dropout_p
        self.featureSize = featureSize
        self.max_length = max_length # 50 sequence ki lenght

        self.embedding = nn.Embedding(self.VocabSizeForEmbedding, self.featureSize)
        self.attn = nn.Linear(self.hidden_size + self.featureSize, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.featureSize, self.featureSize)
        self.dropout = nn.Dropout(self.dropout_p)
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.VocabSizeForEmbedding)

    def forward(self, input, hidden, encoder_outputs):
        input = torch.tensor(torch.argmax(input.view(-1)), dtype=torch.long)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hiddenr = self.LSTM(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hiddenr, attn_weights


    def initHidden(self, mode):
        if (mode.lower() == "lstm"):
            return (torch.zeros(self.layers, 1, self.hidden_size), torch.zeros(self.layers, 1, self.hidden_size))
        else:
            return torch.zeros(1, 1, self.hidden_size)