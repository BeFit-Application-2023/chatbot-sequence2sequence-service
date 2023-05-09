# Defining the tokens.
PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:
    def __init__(self, name : str) -> None:
        '''
            The constructor of the Vocabulary.
                :param name: str
                    The name of the vocabulary.
        '''
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token : "SOS", EOS_token : "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence : str) -> None:
        '''
            This function adds a sentence and adds every word from it to the vocabulary.
                :param sentence: str
                    The sentence to be added to the vocabulary.
        '''
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word : str) -> None:
        '''
            This function adds words it to the vocabulary.
                :param word: str
                    The word to be added to the vocabulary.
        '''
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove word below a certain count threashold.
    def trim(self, min_count : int) -> None:
        '''
            This function removes the words below the count threshold.
                :param min_count: int
                    The threshold bellow which words are eliminated.
        '''
        # Checking if the vocabulary is already trimmed.
        if self.trimmed:
            return
        self.trimmed = True

        # Keeping the words that are below the threshold.
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("kepp word {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries.
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token : "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        # Re-adding words that had a frequency above the threshold.
        for word in keep_words:
            self.add_word(word)