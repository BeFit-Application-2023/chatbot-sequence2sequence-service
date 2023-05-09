# Importing all needed libraries.
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loung attention Layer.
class Attn(nn.Module):
    def __init__(self,
                 method : str,
                 hidden_size : int) -> None:
        '''
            The Long attention layer.
                :param method: str
                    The name of the method used for Loung attention layer.
                    Should be "dot", "general", "concat" else raises the ValueError
                :param hidden_size: int
                    The hidden size of the linear layers used for attention layer.
        '''
        # Setting up the method.
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropiate attention method!")
        self.hidden_size = hidden_size

        # Setting up the attention layer.
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden : "torch.Tensor", encoder_output : "torch.Tensor") -> "torch.Tensor":
        '''
            Defining the dot score.
                :param hidden: torch.Tensor
                    The hidden layer output.
                :param encoder_output: torch.Tensor
                    The output of the encoder.
        '''
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden : "torch.Tensor", encoder_output : "torch.Tensor") -> "torch.Tensor":
        '''
            Defining the general score.
                :param hidden: torch.Tensor
                    The hidden layer output.
                :param encoder_output: torch.Tensor
                    The output of the encoder.
        '''
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden : "torch.Tensor", encoder_output : "torch.Tensor") -> "torch.Tensor":
        '''
            Defining the concat score.
                :param hidden: torch.Tensor
                    The hidden layer output.
                :param encoder_output: torch.Tensor
                    The output of the encoder.
        '''
        energy = self.attn(torch.cat((
            hidden.expand(encoder_output.size(0), -1, -1), encoder_output
        ), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) base on the given method.
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max length and batch size dimensions.
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension).
        return F.softmax(attn_energies, dim=1).unsqueeze(1)