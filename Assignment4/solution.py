# Done
# from your solution module import ...
# your outputs should include Training MSE and Dev MSE.
# useful imports
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data_utils

# Load the data 
housing = fetch_california_housing()
data = pd.DataFrame(data=housing.data)
target = housing["target"]

# Define the training hyperprameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(housing.data, housing.target)

# Split data (1000 train sets and 100 dev sets)
x_train, x_dev, y_train, y_dev = train_test_split(data, target, train_size=1000, test_size=100, random_state=40)

# Convert data to tensors (get error with pandas)
x_train_tensored = torch.tensor(x_train).type(torch.float32)
x_dev_tensored = torch.tensor(x_dev).type(torch.float32)

y_train_tensored = torch.tensor(y_train).type(torch.float32) 
y_dev_tensored = torch.tensor(y_dev).type(torch.float32) 

# unsqueeze target data
y_train = torch.unsqueeze(y_train_tensored, dim = 1)
y_dev = torch.unsqueeze(y_dev_tensored, dim = 1) 

# Create data loaders
train = data_utils.TensorDataset(x_train_tensored, y_train)
train_loader = DataLoader(train, batch_size=10)

dev = data_utils.TensorDataset(x_dev_tensored, y_dev)
dev_loader = DataLoader(dev, batch_size=10)


# Create the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_layer1 = nn.Linear(8, 32)
        self.linear_layer2 = nn.Linear(32, 1)
        
    
    def forward(self, x):
        x = F.relu(self.linear_layer1(x))      # activation function for hidden layer
        x = self.linear_layer2(x)              # linear output
        return x


def start():
    net = NeuralNetwork()

    # Initialize the loss function
    loss_fn = nn.MSELoss()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)    

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, net, loss_fn, optimizer)
        test_loop(dev_loader, net, loss_fn)
    print("Done!")


# Define training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch+1) % 64 == 0:
            loss = loss.item()
            print(f"training loss: {loss:>7f}")

# Define test loop
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
          #  print(f"test loss: {test_loss:>7f}")

    test_loss /= num_batches
    print(f"Avg Test loss: {test_loss:>8f} \n")
