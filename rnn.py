import numpy as np
from numpy.random import randn

class RNN: 
    # Vanilla RNN
    
    def __init__(self, input_size, output_size, hidden_size=64): 
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def forward(self, inputs): 
        '''
        Perform a forward pass of the RNN using the given inputs.
        
        Returns the final output and hidden state.
            - inputs is an array of one-hot vectors with shape (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))
        
        # Perform each step of the RNN
        for i,x in enumerate(inputs): 
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            
        y = self.Why @ h + self.by
        
        return y,h
    
    def backprop(self, d_y, learn_rate=2e-2):
        '''
        Perform a backward pass of the RNN.
        - d_y (dL/dy) has shape (output_size, 1).
        - learn_rate is a float.
        '''
        pass

        
        
