import torch
import torch.nn as nn
#import torchvision.models as models
#from vgg import *
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn 
import torch.nn.functional as F

   

class MyModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, vocab):
        """Set the hyper-parameters and build the layers."""
        super(MyModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab = vocab
        self.linear = nn.Linear(embed_size*22,4800)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        #print(embeddings.size())
        embeddings = embeddings.view(embeddings.size()[0],-1)
        #embeddings = embeddings.view(1,-1)
        #print(embeddings.size())
        outputs = F.relu(self.linear(embeddings))
        outputs = outputs.view(outputs.size(0),3,40,40)
        return outputs
    
