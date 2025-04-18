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

shape = (3,4,)

#rand_tensor = torch.rand(shape)
#ones_tensor = torch.ones(shape)
#zeros_tensor = torch.zeros(shape)

#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Random Tensor: \n {ones_tensor} \n")
#print(f"Random Tensor: \n {zeros_tensor} \n")

############

#attributes of a tensor
#tensor = torch.rand(shape)

#print(f"Shape of tensor: {tensor.shape}")
#print(f"Datatype of tensor: {tensor.dtype}")
#print(f"Device tensor is stored on: {tensor.device}")

#if torch.accelerator.is_available():
#    tensor = tensor.to(torch.accelerator.current_accelerator())

###########

#indexing and slicing
tensor = torch.rand(shape)

#print(f"First Row: {tensor[0]}")
#print(f"First Col: {tensor[:, 0]}")
#print(f"Last Colz; {tensor[..., -1]}")

#tensor[:,2] = 0

print(tensor)

#joining tensors, what is dim?

#t1 = torch.stack([tensor, tensor, tensor], dim =1)
#print(t1)

###########

#arithmetic operations

#y1 = tensor @ tensor.T #transpose of 
#y2 = tensor.matmul(tensor.T)

#y3 = torch.rand_like(y1)
#torch.matmul(tensor, tensor.T, out=y3)

#print(y3)

#z1 = tensor * tensor
#z2 = tensor.mul(tensor)

#z3 = torch.rand_like(tensor)
#torch.mul(tensor, tensor, out=z3)

#agg = tensor.sum()
#agg_item = agg.item()
#print(agg_item, type(agg_item))

#print(f"{tensor} \n")
#tensor.add_(5)
#print(tensor)

#########

#bridging between numpy and tensor
#t = torch.ones(2,3)
#t.add_(2.1)
#print(f"t: {t}")
#n = t.numpy()
#print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")