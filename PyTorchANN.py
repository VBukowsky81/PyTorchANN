import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

xy = np.loadtxt(r"C:/PythonProjects/PyTorchBuild2/diabetes.csv.gz", delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:,0:-1]))
y_data = Variable(torch.from_numpy(xy[:,[-1]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
      
  #Building ANN
  
  #Linear layers. 8 is the input size, from the dataset size.   

        self.l1 = torch.nn.Linear(8,6)
        self.l2 = torch.nn.Linear(6,4) #6 is from previous layer
        self.l3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
            
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
            
model = Model()

#BCE - Binary Cross Entropy loss function
criterion = torch.nn.BCELoss(size_average=True)

#SGD - Stochastic Gradient Descent, lr - learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#Feeding data here, 100 epochs
for epoch in range(100):
     y_pred = model(x_data)
     
     loss = criterion(y_pred, y_data)
     print(epoch, loss.data)
 
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
    
    