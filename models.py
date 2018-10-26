import torch.nn as nn
import torch


class Specialist(nn.Module):

    def __init__(self, input_size, hidden_size, num_tags, num_layers):
        """[summary]
        
        Arguments:
            input_size {int} -- Size of the input layer
            hidden_size {int} -- Size of the hidden layer
            num_tags {[type]} -- [description]
            num_layers {[type]} -- [description]
        """
        pass


class Generalist(nn.Module):
    """ The generalists composes music without learning the difference between 
        different composers using an LSTM.
    """    
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(Generalist, self).__init__()
        self.input_size = input_size
        self.output_size = input_size # Since input is the same as output (we want to predict next notes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = self.init_hidden()
        self._cellstate = torch.zeros(num_layers, 1, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=(0 if num_layers==1 else 0.05))
        self.hidden_to_output = nn.Linear(hidden_size, self.output_size)
    
    def init_hidden(self):
        return  (torch.zeros(self.num_layers, 1, self.hidden_size),
                 torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, inputs):
        # Since this is the generalist, do not pass any category info to the module
        output, self.hidden = self.lstm(inputs, self.hidden)
        output = self.hidden_to_output(output)
        return output
