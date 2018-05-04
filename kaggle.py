import torch
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


class DigitDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx, 1:].as_matrix().astype("float32")
        label = self.data.iloc[idx, 0]

        sample = [pixels, label]
        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(784, 500)
        self.output_layer = nn.Linear(500, 10)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x # The critical bug, forgot to return x


data = DataLoader(DigitDataset("train.csv"), batch_size=16, shuffle=True)
net = Net()

lr = 0.001 # This was originally set to 0.01, which is too high, to demonstrate the affect of learning rate
momentum = 0.5
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

log_interval = 100

def train(epoch):
    net.train()
    for idx, sample in enumerate(data):
        pixels = sample[0]
        label = sample[1]
        optimizer.zero_grad()
        output = net(pixels)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx* len(pixels), len(data.dataset), # Changed len(data) to len(pixels) to properly count up
100. * idx / len(data), loss.item()))

for i in range(1, 1+1): # 1 to x+1 so that it starts counting from 1 rather than 0
    train(i)

class TestDigitDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].as_matrix().astype("float32")
        return sample

def generate_output():
    with open("output.csv", "w") as f:
        f.write("ImageId,Label\n")
        net.eval()
        test_data = DataLoader(TestDigitDataset("test.csv"))
        for idx, sample in enumerate(test_data):
            output = net(sample)
            pred = output.max(1, keepdim=True)[1].item()
            f.write(str(idx+1)+","+str(pred)+"\n")

generate_output()