import torch
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


class DigitDataset(Dataset): # Dataset class describes how to read our data
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx, 1:].as_matrix().astype("float32") # Neural Networks generally always want
                                                                       # float32 data
        label = self.data.iloc[idx, 0] # First entry is always the label

        sample = [pixels, label] # Pack pixels and label into a list
        return sample # And return


class Net(nn.Module): # Our neural network
    def __init__(self):
        super(Net, self).__init__() # Every pytorch module needs to first be initialized like this
        self.hidden_layer = nn.Linear(784, 500) # Our hidden layer, going from 784 pixels to 500 hidden neurons
        self.output_layer = nn.Linear(500, 10) # Our output layer, going from 500 hidden neurons to 10 output neurons

    def forward(self, x): # The forward method defines how the module computes it's output
        x = self.hidden_layer(x) # Apply the hidden layer
        x = F.relu(x) # Apply a RELU function
        x = self.output_layer(x) # Apply our output layer
        x = F.log_softmax(x, dim=1) # Apply a softmax to get our results
        return x # CORRECTION: The critical bug, forgot to return x


data = DataLoader(DigitDataset("train.csv"), batch_size=16, shuffle=True) # A Dataloader takes in a Dataset
                                                                          # And then takes care of loading it for us
net = Net() # Create our neural net

lr = 0.001 # CORRECTION: This was originally set to 0.01, which is too high, to demonstrate the affect of learning rate
momentum = 0.5
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # Create our optimizer

log_interval = 100 # This is every how many steps we want to print our status

def train(epoch): # The method that does the training
    net.train() # Sets the model into training mode. Does nothing in this case, but good to remember to do
    for idx, sample in enumerate(data): # For every batch of data...
        pixels = sample[0] # Seperate pixels and label for readability
        label = sample[1]
        optimizer.zero_grad() # Always zero the gradients on the optimizer before each training setp
        output = net(pixels) # Generate our predictions
        loss = F.nll_loss(output, label) # Score our predictions
        loss.backward() # Perform the backward pass
        optimizer.step() # Update our model

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx* len(pixels), len(data.dataset), # CORRECTION: Changed len(data) to len(pixels) to properly count up
100. * idx / len(data), loss.item()))

for i in range(1, 5+1): # 1 to x+1 so that it starts counting from 1 rather than 0
    train(i)

class TestDigitDataset(Dataset): # Slightly modified dataset since the test data doesn't include labels
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].as_matrix().astype("float32")
        return sample

def generate_output(): # Function to write our final output file
    with open("output.csv", "w") as f:
        f.write("ImageId,Label\n") # Write the header line
        net.eval() # Set our model into evaluation mode
        test_data = DataLoader(TestDigitDataset("test.csv"))
        for idx, sample in enumerate(test_data):
            output = net(sample)
            pred = output.max(1, keepdim=True)[1].item() # Our model outputs ten numbers between 0 and 1 showing
                                                         # How confident it is that the input is a given number
                                                         # This line selects the largest of those numbers and gets
                                                         # Its dimension, which is our predicted number
            f.write(str(idx+1)+","+str(pred)+"\n") # Write our result to the output file, making sure to remember
                                                   # To write idx+1 so that we start by 1 instead of 0

generate_output() # Write our output file