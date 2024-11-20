import torch 
import torch.nn as nn


def get_layer_data(layer, part='weight'):
    if part == 'weight':
        return layer.weight.shape, layer.weight.dtype
    elif part == 'bias':
        return layer.bias.shape, layer.bias.dtype
   

def xavier_uniform(layer, part='weight'):
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using Xavier uniform initialization
    tensor = nn.init.xavier_uniform_(tensor)
    return tensor 
  

def xavier_normal(layer, part='weight'):
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using Xavier normal initialization
    torch = nn.init.xavier_normal_(tensor)
    return tensor

def standard_uniform(layer, part='weight'):
    size, dtype = get_layer_data(layer)
    tensor = torch.rand(size=size, dtype=dtype) 
    return tensor 

def standard_normal(layer, part='weight'):
    size, dtype = get_layer_data(layer)
    tensor = torch.randn(size=size, dtype=dtype)
    return tensor   

def none(layer, part='weight'):
    return getattr(layer, part)

def he_uniform(layer, part='weight'):
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using he uniform initialization
    tensor = nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
    return tensor 
  

def he_normal(layer, part='weight'):    
    size, dtype = get_layer_data(layer)
    # Create a tensor filled with zeros
    tensor = torch.zeros(size, dtype=dtype)
    # Initialize the tensor using he normal initialization
    torch = nn.init.kaiming_normal_(tensor, nonlinearity='relu')
    return tensor

def default(layer, part='weight'):
    k = torch.tensor(1./layer.in_features)
    a = torch.sqrt(k)
    return torch.distributions.uniform.Uniform(-a, a).sample(getattr(layer, part).shape)