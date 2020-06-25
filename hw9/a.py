import torch
import torch.nn as nn
a = [(1,2),2]
b = [[a[0][0],a[0][1], a[1]]]
print(b)
b = torch.FloatTensor(b)
a = 1
label = torch.FloatTensor([[a]])
print(label)
#d =  torch.ones(3, dtype=torch.long)
#d = torch.LongTensor([d])
criterion = nn.MSELoss()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.ReLU()
        self.fc2 = nn.Linear(3, 16, bias=True)
        self.fc3 = nn.Linear(16, 32, bias=True)
        self.fc4 = nn.Linear(32, 1, bias=True)
        self.fc5 = nn.Tanh()

    def forward(self, x):

        x = self.fc2(x)
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc1(x)
        x = self.fc4(x)
        return x

net = Net()
loss = criterion(net(b), label)
print(loss)