import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a model class that inherits nn.module
class Model(nn.Module):
  #input layer (4 feutres of flower) -->
  #hidden layer1 (number of neurons) -->
  #H2(n) --> 
  # output (3 classes of iris flower)
  def __init__(self, in_features=4,h1=8,h2=9,out_features=3):
    super().__init__() #instantiate our nn.Module
    self.fc1=nn.Linear(in_features,h1)
    self.fc2=nn.Linear(h1,h2)
    self.out=nn.Linear(h2,out_features)

  def forward(self,x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=self.out(x)
    return x
  
  #Pick a manual seed for randomization
torch.manual_seed(32)
#Create instance of model
model=Model()

import pandas as pd
import matplotlib.pyplot as plt


url='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df=pd.read_csv(url)
my_df

#change last col from string to innteger
my_df["species"]=my_df["species"].replace("setosa",0.0)
my_df["species"]=my_df["species"].replace("versicolor",1.0)
my_df["species"]=my_df["species"].replace("virginica",2.0)
my_df

#train test split! Set X (feature) Y (outcome)
X=my_df.drop("species",axis=1)
y=my_df["species"]

#convert these to numpy array
X=X.values
y=y.values
from sklearn.model_selection import train_test_split
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=32)
#convert X features to float tensors
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
#convert y labels to tensors long   ##long tensors are 64 bit
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
#set the citerion of model to measure the error 
criterion=nn.CrossEntropyLoss()
#choose adam optimizer. lr=learning rate (if error doesnt go down after a bunch)
optimizer=torch.optim.Adam(model.parameters(),lr=0.02)
#model.parameters() gives layers only fc1.fc2.out
#train our model!
#epochs? (one run thru all the traininig data in our network)
epochs=100
losses=[]
for i in range(epochs):
  #fwd
  y_pred=model.forward(X_train) #get predicted results

  #measure loss
  loss=criterion(y_pred,y_train)

  #keep track of losses
  losses.append(loss.detach().numpy())

  #print every 10 epochs
  if i%10==0:
    print(f"Epochs: {i} and loss: {loss}")

  #backpropogation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

#grapph
plt.plot(range(epochs),losses)
plt.ylabel("loss/error")
plt.xlabel("Epochs")

#Evaluate Model on Test Data set
with torch.no_grad(): #turn off backpropogation
  y_eval=model.forward(X_test) #X_test are features from test set, y_eval
  loss=criterion(y_eval,y_test) #find the losses or error

loss
correct=0
with torch.no_grad():
  for i,data in enumerate(X_test):
    y_val=model.forward(data)

   
    
    #will tell us what type of flower class our network thinks it is
    print(f"{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}")

    #correct or not
    if y_val.argmax().item()==y_test[i]:
      correct+=1
   
print(f"we got {correct} correct")