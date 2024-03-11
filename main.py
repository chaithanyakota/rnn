from data import train_data, test_data
import numpy as np
from rnn import RNN

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



def softmax(xs): 
      # Applies the Softmax Function to the input array xs.
    return np.exp(xs) / sum(np.exp(xs))


# Initialize rnn
rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
prob = softmax(out)
print(prob)

# Loop over each training example
for x, y in train_data.items():
  inputs = createInputs(x)
  target = int(y)

  # Forward
  out, _ = rnn.forward(inputs)
  probs = softmax(out)

  # Build dL/dy
  d_L_d_y = probs
  d_L_d_y[target] -= 1

  # Backward
  rnn.backprop(d_L_d_y)
