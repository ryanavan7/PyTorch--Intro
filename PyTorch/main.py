import torch
import numpy as np

#data = [[1,2,6],[3,4,4]]
#x_data = torch.tensor(data)

#np_array = np.array(data)
#x_np = torch.from_numpy(np_array)

#data1 = [[1,2],[0,0],[5,6],[7,0]]
#x1_data = torch.tensor(data1)

#print(x_data)
#print(x_np)

#x_ones = torch.ones_like(x_data) # retains the properties of x_data
#print(f"Ones Tensor: \n {x_ones} \n")

#x_rand = torch.rand_like(x_np, dtype=torch.float) # overrides the datatype of x_data
#print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,2,5)

rand_tensor = torch.rand(shape)
#ones_tensor = torch.ones(shape)
#zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Random Tensor: \n {ones_tensor} \n")
#print(f"Random Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(shape)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
