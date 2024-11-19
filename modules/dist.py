import torch 
import torch.nn as nn


def get_layer_data(layer):
    return layer.weight.shape, layer.weight.dtype 

def xavier_uniform(layer):
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using Xavier uniform initialization
    tensor = nn.init.xavier_uniform_(tensor)
    return tensor 
  

def xavier_normal(layer):
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using Xavier normal initialization
    torch = nn.init.xavier_normal_(tensor)
    return tensor

def standard_uniform(layer):
    size, dtype = get_layer_data(layer)
    tensor = torch.rand(size=size, dtype=dtype) 
    return tensor 

def standard_normal(layer):
    size, dtype = get_layer_data(layer)
    tensor = torch.randn(size=size, dtype=dtype)
    return tensor   

def none(layer):
    return layer.weight