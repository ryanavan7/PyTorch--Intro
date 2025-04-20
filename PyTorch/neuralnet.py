import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#if accelorator is available, if not use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

#defining the class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #defining layers of neural network
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): #defining forwad pass
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits #raw values in [-infinity, infinity]

#creating and instance of NeuralNetwork and move it to the device 
model = NeuralNetwork().to(device)
print(model)


#passing the input data to the model
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
 
 #taking a minibatch of 3 images
input_image = torch.rand(3,28,28)
print(input_image.size())

#convert the 2D 28x28 image to a contiguous array of 784 pixels
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#linear layer that applies a linear transformation on the input using stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#seeing ReLU, a non-linear activation: creates complex mapping between model's inputs and outputs
#applied after linear transformation
#ReLu: negative values = 0
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After Relu: {hidden1}")

#makes sure that data is passed through all modules in the same order as defined

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#logits are scaled to [0,1] represeting the model's predicted probabiltis for each class. dim paremeter indicates
#values must sum to 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

#model parameters
print(f"Model Structure: {model}\n\n")

print(f"Model structure: {model}\n\n")

#parameterized, having associated weights and biases that are optimized during traing
#Subclassing nn.Module automatically tracks all fields defined inside your model object
# and makes all parameters accessible using your model’s parameters() or named_parameters() methods.
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
