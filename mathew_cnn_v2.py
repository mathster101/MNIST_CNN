import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from random import randint
from matplotlib import pyplot as plt
import sys
import numpy as np
train = datasets.MNIST("",train=True,download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",train=False,download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

#kappa = np.asarray(train[55][0].tolist())
#kappa = np.resize(kappa,(28,28))



bs = 64
trainset = torch.utils.data.DataLoader(train, batch_size = bs,
                                       shuffle = True)
testset = torch.utils.data.DataLoader(train, batch_size = 1,
                                       shuffle = True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10) # Change output_size to the number of classes

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x
     
net = Net().cuda()
optimizer = optim.Adam(net.parameters(), lr=1.0e-3)
criterion = nn.CrossEntropyLoss()

print('net created')
losses=[]
for epoch in range(1):
    for iters, (x, y) in enumerate(trainset):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        
        print(float(loss))
        
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        
#testing time
net.eval()
total = 0
correct = 0
for data in testset:
   x,y = data
   x = x.cuda()
   output = net(x)
   y = list(y)
   values, indices = output[0].max(0)
   y = int(y[0])
   indices = int(indices)
#   print('True Value = ',y)
#   print('Prediction = ',indices)
#   print('\n______________________________________')
   if y == indices:
      correct += 1

      
   total += 1
   print((correct/total)*100,'validation accuracy')


