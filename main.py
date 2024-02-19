from data import train_data, test_data
import numpy as np

# Create vocab
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)

# print(vocab_size)
# for w in vocab: 
#     print(w)

# Assign indices to each word
word_to_idx = {w:i for i,w in enumerate(vocab)}
idx_to_word = {i:w for i,w in enumerate(vocab)}

def createInputs(text): 
    '''
    Returns an array of one-hot vectors representing the words
    in the input text string.
        - text is a string
        - Each one-hot vector has dimensions (vocab_size, 1)
    '''
    for w in text.split(' '): 
        inputs = []
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
        
    return inputs





