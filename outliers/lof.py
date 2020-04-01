import numpy as np
import pandas as pd
from numpy.linalg import *
import matplotlib.pyplot as plt

class LOF:
    def __init__(self, data:np.matrix,K:int):
        self.data = data
        self.K = K      
        
        
    def NkDk(self,p):        
        return self.cache_dist_k[hash(str(p))]

    def createCash(self):
        print("Starting Cache Creation")
        data=self.data
        self.cache_dist_k = {}
        for p in data:
            self.cache_dist_k[hash(str(p))] = self.NkDk_internal(p)
        print("Finished Cache Creation")

    def NkDk_internal(self, p:np.array):
        #distances to point p
        Dtotal = np.array([np.linalg.norm(self.distance(x,p)) for x in self.data])
        Dtotal = Dtotal[Dtotal>1e-12]

        #sorted indexes for distances to point p
        I= np.array(np.argsort(Dtotal))

        #Set of the k closest points to p (sorted)
        Nk = np.array(self.data[I[0:self.K],:])

        #distances of the closest points to p (sorted)
        Dk = np.array(Dtotal[I[0:self.K]])

        def K_1ItemCondition():
            d = Dk[-1]
            cond1 = len(Dk[Dk<=d]) >= self.K
            cond2 = len(Dk[Dk<d])  >= self.K-1
            return cond1 and cond2

        kaux = self.K
        while len(Nk)<len(I) and not(K_1ItemCondition()):
            kaux += 1
            Nk = np.array(self.data[I[0:kaux-1]]) 
            Dk = np.array(Dtotal[I[0:kaux-1]])
        
        return Nk,Dk
        
        
    def distance(self,a:np.array,b:np.array):
        d = (a-b)
        return np.sqrt(d*d)

    def k_distance(self,p:np.array):
        """
            Computes the Nk set and k-distance given a min points value
            params:
            k: number of points in the cluster
            data: matrix holding in row vectors of samples
            x: point on which to compute k-neighbour distance
        """        
        _, d = self.NkDk(p)
        return d[-1]

    def reach_distance(self, p:np.array,o:np.array):
        """Reachability distance of p with respect to o"""
        return np.max([self.k_distance(o),self.distance(p,o)])

    def lrd(self,p):
        """Local reachability density"""
        Nk,_ = self.NkDk(p)
        modNk = len(Nk)
        num = np.sum([self.reach_distance(p,o) for o in Nk])
        return modNk/num
        
    def lof(self,p):
        Nk,_ = self.NkDk_internal(p)
        return np.sum([self.lrd(o) for o in Nk]) * np.sum([self.reach_distance(p,o) for o in Nk])

class LOF_Mahalonobis(LOF):
    def __init__(self,data,K,C:np.matrix):
        super(LOF_Mahalonobis,self).__init__(data,K)
        self.C = C
        self.createCash()

    def distance(self, a, b):
        z = np.array(b)- np.array(a)
        x = np.matrix(z.reshape((len(z),1)))        
        return np.sqrt(np.sum(x.transpose()*self.C*x))


if __name__ == "__main__":
    np.random.seed(42)
    C=[[1.0,0.0],[0.0,1]]
    I = C.copy() # [[1.0,0.0],[0.0,1]]
    data = np.random.multivariate_normal(mean=[0,0],cov=C,size=(500))
    data = np.append(data,[[-1.3,1.6],[-1.9,-1.8],[3.55,3.26]],axis=0)
    algo = LOF_Mahalonobis(data=data,K=50,C=np.linalg.inv(np.matrix(I)))
    xx = algo.lof([-1.3,1.6])
    scores = [algo.lof(x) for x in data]
    iscores = np.argsort(scores)
    scores = [(data[n],scores[n]) for n in iscores[::-1]]
    handles = []
    
    plt.scatter(data[:,0],data[:,1])
    for (p,s) in scores[0:10]:
        plt.scatter(p[0],p[1],color=['r'])
    #plt.legend(handles=handles)
    plt.show()