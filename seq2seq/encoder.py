# Importing all needed libraries.
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self,
                 hidden_size : int,
                 embedding : "nn.Embedding",
                 n_layers : int = 1,
                 dropout : float = 0) -> None:
        '''
            The encoder of the sequence to sequence model.
                :param hidden_size: int
                    The hidden size of the GRU layer.
                :param embedding: nn.Embedding
                    The torch embedding layer used for mapping integer indexes to vectors.
                :param n_layers: int, default = 1
                    The number of recurrent layers in the GRU layer.
                :param dropout: float, default = 0
                    The dropout probability of the GRU layer.
        '''
        # Setting up the configuration of the encoder.
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Setting up the GRU layer.
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self,
                input_seq : "torch.LongTensor",
                input_lengths : "torch.Tensor",
                hidden : "torch.Tensor" = None) -> "torch.Tensor":
        '''
            The forward function of the encoder.
                :param input_seq: torch.LongTensor
                    The tensor containing the indexes of the words.
                :param input_lengths: torch.Tensor
                    The tensor containing the lengths of the sequences.
                :param hidden: torch.Tensor, default = None
                    The hidden states, used only during training.
        '''
        # Convert word indexes to embeddings.
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequence for RNN module.
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU.
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding.
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # um bidirectional GRU outputs.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state.
        return outputs, hidden