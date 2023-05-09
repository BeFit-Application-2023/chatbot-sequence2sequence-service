# Importing all needed module.
import re
import torch
import pickle
import codecs
import unicodedata
import torch.nn as nn

# Importing neural network modules.
from .encoder import EncoderRNN
from .decoder import LuongAttnDecoderRNN
from .searcher import GreedySearchDecoder

class SequenceGenerator:
    def __init__(
            self,
            encoder_path : str,
            decoder_path : str,
            vocabulary_path : str,
            encoder_config : dict,
            decoder_config : dict
    ) -> None:
        # Loading the vocabulary.
        self.vocabulary = pickle.load(open(vocabulary_path, 'rb'))

        # Creating the embedding layer.
        embedding = nn.Embedding(self.vocabulary.num_words, encoder_config["hidden_size"])

        # Creating the encoder and luong decoder.
        self.encoder = EncoderRNN(
            encoder_config["hidden_size"],
            embedding,
            encoder_config["encoder_n_layers"],
            encoder_config["dropout"]
        )
        self.decoder = LuongAttnDecoderRNN(
            decoder_config["attn_model"],
            embedding,
            decoder_config["hidden_size"],
            self.vocabulary.num_words,
            decoder_config["decoder_n_layers"],
            decoder_config["dropout"]
        )
        # Setting the encoder and decoder to the evaluation mode.
        self.encoder.eval()
        self.decoder.eval()

        # Loading the encoder and the decoder.
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

        # Loading the Greedy Searcher Decoder.
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)

        # Creating the parameters for sequence generation.
        self.device = "cpu"
        self.EOS_token = 1
        self.max_length = 10

        # Creating the grammar corrector.
        self.grammar_corrector = {
            "i m" : "i'm",
            " s " : "is ",
            "don t" : "don't",
            " ll" : "'ll",
            " re" : "'re"
        }

    def unicode_to_ascii(self, string : str) -> str:
        '''
            This function converts the unicode strings to ascii.
                :param string: str
                    The string to be converted from unicode to ascii.
                :return: str
                    The converted string.
        '''
        return "".join(
            character for character in unicodedata.normalize("NFD", string)
            if unicodedata.category(character) != "Mn"
        )

    def normalize_string(self, string : str) -> str:
        '''
            This function normalizes the string.
                :param string: str
                    The string to be normalized.
                :return: str
                    THe normalized string.
        '''
        # Converting string from unicode to ascii.
        string = self.unicode_to_ascii(string.lower().strip())

        # Replacing a useless characters from string.
        string = re.sub(r"([.!?])", r" \1", string)
        string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
        string = re.sub(r"\s+", r" ", string).strip()
        return string

    def indexes_from_sentence(self, sentence : str) -> list:
        '''
            This function converts a sentence into list of indexes.
                :param sentence: str
                    The sentence to be converted into indexes.
                :return: list
                    The list of indexes converted using the vocabulary.
        '''
        return [self.vocabulary.word2index[word] for word in sentence.split(' ')] + [self.EOS_token]

    def normalize_output(self, text : str) -> str:
        '''
            This function normalizes the output of the Sequence-to-sequence model.
                :param text: str
                    The predicted sequence of words by sequence-to-sequence model.
                :return: str
                    The normalized output text.
        '''
        # If the sequence contains dots, then we return the sequence only until the first dot.
        if "." in text:
            text = text[:text.index(".")].strip()

        # Repairing the grammar mistakes in the generated sequence.
        for error in self.grammar_corrector:
            if error in text:
                text = text.replace(error, self.grammar_corrector[error])

        # Capitalize the sequence and return it.
        text = text.capitalize()
        return text

    def predict(self, text : str) -> str:
        '''
            This function returns the corrected generated sequence by sequence-to-sequence.
                :param text: str
                    The input text sent by user.
                :return: str
                    The generated sequence.
        '''
        # Normalizing the input sequence.
        text = self.normalize_string(text)
        # Filter words that are not in the vocabulary.
        text = ' '.join([word for word in text.split(" ")
                         if word in self.vocabulary.word2index])
        # Converting the text into a batch of indexes.
        indexes_batch = [self.indexes_from_sentence(text)]
        # Getting the lengths of the indexes list from the bathc.
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Converting the index batch into Long Torch Tensor.
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Converting the input batch and length to device.
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)
        # Getting the tokens and the scores from searcher.
        tokens, scores = self.searcher(input_batch, lengths, self.max_length)
        # Decoding the words.
        decoded_words = [self.vocabulary.index2word[token.item()] for token in tokens]
        # Keeping only the sequence till the first End-of-Sequence of PAD token.
        decoded_words[:] = [x for x in decoded_words if not (x == "EOS" or x == "PAD")]
        # Normalizing the output.
        output_string = self.normalize_output(" ".join(decoded_words))

        return output_string