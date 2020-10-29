import numpy as np
import pandas as pd
def covariancecalc(X):
    s = np.std(X,axis=0)
    mu = np.mean(X,axis=0)

    i = np.abs(s)>1e-8
    print(i)
    n = np.abs(s)<=1e-8
    Xp = X[:,i]
    print(pd.DataFrame(Xp))
    R = pd.DataFrame(Xp).corr()
    C = pd.DataFrame(X).cov()
    print(R)
    sm = np.zeros(X.shape[1])
    print(sm)
    sm[n]=1e-8
    sm[i]=s[i]
    l = np.sqrt(np.diag(sm))
    Rp = np.zeros((X.shape[1],X.shape[1]))   
    
    Rp[np.ix_(i,i)]=R
    Rp[np.ix_(n,n)]=0.0
    for k in range(X.shape[1]):
        Rp[k,k]=1

    l = np.matrix(l)
    Rp = np.matrix(Rp)

    print(pd.DataFrame(l))
    rho = np.matmul(np.matmul(l,Rp),l)

    print("---------rho-----------")
    print(pd.DataFrame(Rp))
    print("---------Cov-----------")
    print(pd.DataFrame(rho))
    print("---------eig-----------")
    print(np.linalg.eigvals(rho))
    print("---------eigwrong-----------")
    print(np.linalg.eigvals(C))

X=np.random.rand(50,5)
X[:,3]=0.0
X[:,1]=1e-9
covariancecalc(X)