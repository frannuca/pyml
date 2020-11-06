import torch
import numpy as np
import matplotlib.pyplot as plt

class UnivariateDensityEstimator(torch.nn.Module):
    def __init__(self):
        super(UnivariateDensityEstimator,self).__init__()
        self.layers = []
        self.linear1 = torch.nn.Linear(1,10)
        self.linear2 = torch.nn.Linear(10,10)
        self.linear3 = torch.nn.Linear(10,1)
        self.activation = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear3.weight)
        

    
    def forward(self,x):
        s = self.linear1(x)
        s = self.activation(s)

        s = self.linear2(s)
        s = torch.sigmoid(s)

        s = self.linear3(s)
        s = torch.sigmoid(s)

                 
        return s

#create training data:
Ns=20000
ss = np.concatenate((np.random.gamma(7,1,int(Ns/4)),
                    5.0*np.random.normal(-2,1,int(Ns/4)),
                    1.0*np.random.normal(-5,1,int(Ns/4)),
                    15*np.random.normal(12,1,int(Ns/4))),axis=0)
X, _ = torch.sort(torch.from_numpy(ss).float())
m = torch.mean(X)
s = torch.max(X)-torch.min(X)
X =  (X - m)/s
X=X.unsqueeze(-1)

sx = X
sx_p = sx + 0.1


model = UnivariateDensityEstimator()


def my_loss(output, target):
    d = model(sx)-model(sx_p)
    h = 0.0#torch.sum((1-torch.sign(d))*d**2)
    #print(f"h={h}")
    
    loss = torch.sum((output-target)**2)+1e3*h
    return loss

criterion = my_loss#torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


for t in range(10000):
    Y,_ = torch.sort(torch.rand(Ns))
    Y=Y.unsqueeze(-1)
    y_pred = model(X)
    loss = criterion(y_pred,Y)
    if t % 100 == 99 or t==0:
        print(t,loss.item())
        if loss.item()<0.1:
            break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


s = torch.from_numpy(np.linspace(-1,1,1000)).unsqueeze(-1)
e = 0.01
sp = s+e
p = (model(sp.float()) - model(s.float()))/e
x = s.squeeze()
plt.plot(s.squeeze(-1).detach().numpy(),p.squeeze(-1).detach().numpy())
xh = X.squeeze(-1).numpy()
plt.hist(xh, 100, density=True, facecolor='b', alpha=0.5)
plt.show()