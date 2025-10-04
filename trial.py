import torch
import numpy as np
import torch.nn as nn         ######loss
import torch.optim as optim         #####optimizer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Corrected dataset creation
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_informative=1,   # fix: 1 informative instead of 2
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# standardize features
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test   = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


#tensors
x=torch.tensor([1,2,3],dtype=torch.float32)
p=x.shape


#convert numpy to pytorch
arr=np.array([1,2,3])
torch_arr=torch.from_numpy(arr)         #numpy to pytorch
k=torch_arr.numpy()                     #pytorch to numpy

#grad
w=torch.tensor([1.0],requires_grad=True)

#simple operation
y=w*2
y.backward()


#Model definition
class LogisticReg(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.linear=nn.Linear(n_features,1)
    def forward(self,x):
        return torch.sigmoid(self.linear(x))

# instantiate model
model=LogisticReg(n_features=2)

#loss
criterion=nn.BCELoss()

#Optimizer
optimizer=optim.Adam(model.parameters(),lr=0.01)

#training loop
for epoch in range(100):
    y_pred=model(X_train).squeeze()
    #input
    loss=criterion(y_pred,y_train)
    #BCE too find loss
    optimizer.zero_grad()
    #sets new=0
    loss.backward()
    #back prop
    optimizer.step()

    if (epoch+1)%20==0:
        print(f"Epoch{epoch+1},Loss={loss.item():4f}")

with torch.no_grad():
    y_pred_test=model(X_test)
    acc=((y_pred_test.squeeze()>0.5)==y_test).float().mean()
    print("Accuracy: ",acc.item())
