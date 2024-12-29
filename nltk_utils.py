import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure the required NLTK data is available
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenize a sentence into an array of words/tokens.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stem a word to its root form.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return a bag of words array: 1 for each known word that exists in the sentence, 0 otherwise.
    """
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word in the vocabulary
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
