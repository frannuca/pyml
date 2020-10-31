import numpy as np
import pandas as pd

def covariancecalc(X,min_sigma=1e-5):
    N=X.shape[1]

    sigma = np.std(X,axis=0)
    mu = np.mean(X,axis=0)

    valid_idx = np.abs(sigma)>min_sigma
    print(valid_idx)
    singular_idx = np.abs(sigma)<=min_sigma
    Xp = X[:,valid_idx]
    Np = Xp.shape[1]
    Rho = pd.DataFrame(Xp).corr()

    sm=np.zeros(N)
    sm[valid_idx]=sigma[valid_idx]
    sm[singular_idx]=min_sigma
    L = np.diag(sm)

    Rp = np.zeros((N,N))  
    L = np.matrix(L) 
    
    Rp[np.ix_(valid_idx,valid_idx)]=Rho
    Rp[np.ix_(singular_idx,singular_idx)]=0.0

    for k in range(N):
        Rp[k,k]=1

    
    Rp = np.matrix(Rp)
    C = np.matmul(np.matmul(L,Rp),L)
    return C

X=np.random.rand(50,5)
X[:,3]=0.0
X[:,1]=1e-9
X[:,0]=0.0
X[:,2] =0.0
X[:,4]=1e-9
C=covariancecalc(X)
Cinv = np.linalg.inv(C)

Co = pd.DataFrame(X).cov()
Coinv = np.linalg.inv(Co)

print (pd.DataFrame(Cinv))
print(pd.DataFrame(Coinv))