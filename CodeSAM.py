import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
import numpy as np
from sam import SAM
print('Adam samdler')
data = pd.read_csv(r'C:\Users\Mathias\Documents\Programmieren\GenBoost\ZeroFinalTrain.csv')

#hier werden die ergebnisse gespeichert
matrix = np.zeros((6, 4))



X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)


#train test split
train_size = int(0.8 * len(dataset))
indices = list(range(len(dataset)))
train_dataset = Subset(dataset, indices[:train_size])
test_dataset = Subset(dataset, indices[train_size:])




train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(16, 18)
        self.hidden2 = nn.Linear(18, 18)
        self.hidden3 = nn.Linear(18, 18)
        self.hidden4 = nn.Linear(18, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.hidden4(x)
        x = self.sigmoid(x) * 7
        return x

def create_NeuralNetwork():
    return NeuralNetwork()



criterion = nn.MSELoss()
epochs = 100

#these hyperparames will be tested
learning_rate =  [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 2e-2]
wd = [0, 1e-4, 1e-3, 1e-2]


for i in range(3):
    m = 0
    n = 0
    for j in wd:
        
        for i in learning_rate:
            model = create_NeuralNetwork()
            base_optimizer = optim.Adam
            optimizer = SAM(model.parameters(), base_optimizer, lr=i, weight_decay=j)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs / 2)

            for epoch in range(epochs):
                for inputs, targets in train_loader:
                    #first step
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    #second step
                    criterion(model(inputs), targets).backward()
                    optimizer.second_step(zero_grad=True)
                
       
            scheduler.step()



            model.eval()  
            total_loss = 0.0

            
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            with torch.no_grad():
                for features, labels in test_loader:
                    
                    loss = criterion(model(features), labels)
                    
                    
                    total_loss += loss.item()

            
            


            matrix [n][m] += total_loss
            
            n += 1
        n = 0
        m += 1

    print(matrix)
print('Ergebnis SGD SAM 0.1')
matrix = matrix / 3
print(matrix)