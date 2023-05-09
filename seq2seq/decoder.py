# Importing all needed modules.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the attention layer.
from .attention import Attn


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,
                 attn_model : str,
                 embedding : "nn.Embedding",
                 hidden_size : int,
                 output_size : int,
                 n_layers : int = 1,
                 dropout : float = 0.1) -> None:
        '''
            Defining the Luong Attention Decoder.
                :param attn_model: str
                    The method of the attention layer.
                :param embedding: nn.Embedding
                    The torch embedding layer used for mapping integer indexes to vectors.
                :param hidden_size: int
                    The hidden size of the GRU layer.
                :param output_size: int
                    The size of the prediction vector.
                :param n_layers: int, default = 1
                    The number of recurrent layers in the GRU layer.
                :param dropout: float, default = 0.1
                    The dropout of the GRU layer.
        '''
        # Setting up the decoder.
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference.
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define the of the decoder layers.
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout)
        )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Defining the attention module.
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time.
        # Get embedding of current input word.
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU.
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output.
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector.
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong.
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Loung eq 6.
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state.
        return output, hidden