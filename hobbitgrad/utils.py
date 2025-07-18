from random import shuffle

def split(data, ratio):
    shuffle(data)    
    return data[:ratio], data[ratio:]
   