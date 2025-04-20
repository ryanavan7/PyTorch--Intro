import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) #requires_grad is used to compute the gradients of the loss function
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b #computational graph
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#computing gradients
loss.backward() #optimization with loss fucntion
print(w.grad)
print(b.grad)

#disbaling gradient tracking
#To mark some parameters in your neural network as frozen parameters.
#To speed up computations when you are only doing forward pass, 
#because computations on tensors that do not track gradients would be more efficient.
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

#DAG (directed acylic graph), they are dynamic, new one created after each .backward() call
#leaves are the input tensors, roots are the output tensors
