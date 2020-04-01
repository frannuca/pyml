import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lof
from itertools import combinations
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

class SubSpace:
    def __init__(self,indexes,contrast:float):
        self.__indexes = indexes
        self.__contrast = contrast
    @property
    def Indexes(self):
        return self.__indexes
    
    @property
    def Contrast(self):
        return self.__contrast


class Slice:
    def __init__(self,lowerval:float,upperval:float,index:int):
        self.__lowerlimit = lowerval
        self.__upperlimit = upperval
        self.__index = index
    
    @property
    def Lowerlimit(self):
        return self.__lowerlimit
    
    @property
    def Upperlimit(self):
        return self.__upperlimit

    @property
    def Index(self):
        return self.__index
   
class HiCS:
    def __init__(self,data:np.matrix,alpha:float):
        self.data=data
        self.alpha = alpha
        self.D = np.linspace(0,data.shape[1])
        self.lower_boundaries = np.min(self.data,axis=0)
        self.upper_boundaries = np.max(self.data,axis=0)
        self.Bands = self.upper_boundaries-self.lower_boundaries
        

    def getSlice(self,S):
        l = []
        u = []
        Bands = self.Bands[S]
        lower_boundaries = self.lower_boundaries[S]
        upper_boundaries = self.upper_boundaries[S]
        
        l = np.random.rand(len(S))*Bands+lower_boundaries
        u = l + Bands*self.alpha
        du = upper_boundaries-u
        du[du>0] = 0
        u = u + du
        l = l + du
        return [Slice(a,b,c) for (a,b,c) in  list(zip(l,u,S))]
    
    def suffleSubspace(self,S):
        Sx = list(S)
        np.random.shuffle(Sx)
        return Sx

   
    def deviation(self,S):
        """computes the deviation in between p(i) and p(i|C)"""
        i = S[0]
        S_1 = S[1:]
        slices = self.getSlice(S_1)

        xlow = self.lower_boundaries[i]
        xmax = self.upper_boundaries[i]
        
        N = self.data.shape[0]
        
        #reduced data set to C
        maskC_low = reduce(lambda a,b: a & b, [self.data[:,slices[j].Index]>=slices[j].Lowerlimit  for j in range(len(slices))])
        maskC_up =  reduce(lambda a,b: a & b, [self.data[:,slices[j].Index]<=slices[j].Upperlimit  for j in range(len(slices))])
        maskC = maskC_up & maskC_low
        dataC=self.data[maskC,i]
        datatotal = self.data[:,i]
        xsamples = np.sort(np.unique(datatotal))
        

        Pi =   np.array([ len(datatotal[datatotal < x])/N for x in xsamples])
        NC = dataC.shape[0]
        if NC>5:
            Pi_C = np.array([ len(dataC[dataC < x])/NC for x in xsamples])
            jjj = np.max(np.abs(Pi-Pi_C))
            return jjj
        else:
            return 0.0

    def contrast(self,S,M):            
        deviations = []
        for n in range(M):
            Ss=self.suffleSubspace(S)                      
            deviations.append(self.deviation(Ss))
        return np.sum(deviations)/M

    def processSubspace(self,subspaces,M):
        results = []
        for S in subspaces:
            results.append(SubSpace(S,self.contrast(S,M)))
        return np.array(sorted(results,key=lambda x: x.Contrast))[::-1]

    def run(self,M:int,minSdim:int,topN:int,maxDim:int):
        D = self.data.shape[1]
        indexes = range(self.data.shape[1])
        ndim = minSdim
        subspaces = list(combinations(indexes,ndim))
        results = []
        done=False
        T = None
        
        while len(subspaces)>0:
            if ndim<=maxDim:
                contrastres = self.processSubspace(subspaces,M)
                Tx = np.mean(list(map(lambda x: x.Contrast,contrastres[::topN])))
                if T is None:
                   T=Tx
                elif T<Tx:
                    T=Tx

                constrastres = list(filter(lambda x:x.Contrast>=T,contrastres))
                results = np.concatenate((results,constrastres),axis=0)
                subspaces = np.array([ [list(s.Indexes)+[n] for n in (set(range(D))-set(s.Indexes))]  for s in contrastres])
                subspaces = subspaces.reshape(subspaces.shape[0]*subspaces.shape[1],subspaces.shape[2])
                
                ndim += 1
            else:
                break        
        cons = list(map(lambda x: x.Contrast,results))
        idx = np.argsort(cons)[::-1]
        return np.array(results)[idx][0:topN]

if __name__ == "__main__":
    #generate random data set for clustering detection
    np.random.seed(42)
    D=3
    C=np.matrix(np.eye(D))
    C[0,2]=0.99
    C[2,0]=0.99
    data = np.random.multivariate_normal(mean=np.zeros(D),cov=C,size=(500))
    data = np.append(data,[[-3.5,-1.7,-3.37],[3.5,1.57,-3.5]],axis=0)
    hics = HiCS(data=data,alpha=0.10)
    S = hics.run(M=100,minSdim=2,topN=2,maxDim=D-1)
    total_scores = 0.0

    for s in S:
        datax = data[:,s.Indexes]
        print(f'{s.Indexes}, {s.Contrast}')
        Cx = np.eye(len(s.Indexes))        
        algo = lof.LOF_Mahalonobis(data=datax,K=50,C=np.linalg.inv(np.matrix(Cx)))
        scores = np.array(list([algo.lof(x) for x in datax]))        
        total_scores = total_scores + scores
        

    iscores = np.argsort(total_scores)
    scores_x = [data[n,0] for n in iscores[::-1]]
    scores_y = [data[n,1] for n in iscores[::-1]]
    scores_z = [data[n,2] for n in iscores[::-1]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(data[:,0],data[:,1] , data[:,2], marker='^')
    ax.scatter(scores_x[0:50],scores_y[0:50],scores_z[0:50],marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()