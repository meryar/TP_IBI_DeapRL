import random

class Buffer:
    
    def __init__(self, size):
        self.size = size
        self.data = []
        self.oldest = 0
        self.nb_elements = 0
        
    def insert(self, event):
        if len(self.data) < self.size:
            self.data.append(event)
            self.nb_elements += 1
        
        else:
            self.data[self.oldest] = event
            self.oldest += 1
            self.oldest %= self.size
        
    
    def sample(self, size):
        return random.sample(self.data, size)
    
    def get_nb_elements(self):
        return self.nb_elements
            
    
    
    