import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Define the Feedforward Neural Network with Dropout and BatchNorm
class FNN(nn.Module):
    def _init_(self):
        super(FNN, self)._init_()
        # Input to first hidden layer
        self.fc1 = nn.Linear(28*28, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        
        # Second hidden layer
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input (batch_size, 784)
        x = F.relu(self.batchnorm1(self.fc1(x)))  # First layer with BatchNorm and ReLU
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Second layer with ReLU
        x = self.fc3(x)  # Output layer
        return F.log_softmax(x, dim=1)  # Log softmax for classification

# 2. Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3. Initialize the model, loss function, and optimizer
model = FNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Train the model and collect losses for plotting
def train_model(epochs):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Test the model after each epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

# 5. Train the model for a number of epochs and plot the results
epochs = 10
train_losses, test_losses = train_model(epochs)

# 6. Plot the training and test losses
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()