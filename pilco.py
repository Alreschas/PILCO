# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def dynamics(x):
    N = x.shape[0]
    M = x.shape[1]
    return np.sin(x * np.pi) + np.random.randn(N,M) * 0.1

def gaussian(x,mu,s2):
    ret = np.exp(-(x-mu)**2/(2*s2))/np.sqrt(2*np.pi * s2) 
    return ret

class GaussianProcess:

    def gramMat(self,x):
        N = self.N
        kern = self.kern
        ret = np.zeros([N,N])
        
        for n1 in range(N):
            x1 = x[n1,:]
            for n2 in range(N):
                x2 = x[n2,:]
                ret[n1,n2] = kern(x1,x2)
        return ret
    
    def kMat(self,x,x1):
        kern = self.kern
        N = self.N
        ret = np.zeros([N,1])
        
        for n1 in range(N):
            ret[n1,0] = kern(x[n1,:],x1)
        return ret

    def init(self,tx,ty):
        self.sw2 = 0.0001 # prev dist variance
        self.tx = tx
        self.ty = ty
        self.N = tx.shape[0]
        self.M = tx.shape[1]
        
        #kernel setting
        self.sf2 = 10
        length = 0.1
        self.lam = length*np.eye(self.M)
        
    def learn(self,tx,ty):
        self.init(tx,ty)
        
        N = self.N
        sw2 = self.sw2
        CN = self.gramMat(tx) + np.eye(N)*sw2
        self.invCN = np.linalg.inv(CN)
        self.invCNdTy = self.invCN.dot(ty)


    #x is column vector
    def predict(self,x):
        tx = self.tx
        sw2 = self.sw2
        invCN = self.invCN
        invCNdTy = self.invCNdTy
        kern = self.kern

        k =  self.kMat(tx,x)
        c = kern(x,x) + sw2
                
        s2 = np.sqrt(c - k.T.dot(invCN).dot(k))
        y = k.T.dot(invCNdTy)
        return y,s2
        
    #mu_t : mean of prev dist (column vector)
    def momentMatching(self,mu_t,S_t):
        tx = self.tx
        sf2 = self.sf2        
        sw2 = self.sw2   
        lam = self.lam
        M = self.M
        N = self.N
        invCNdTy = self.invCNdTy
        invCN = self.invCN
        
        qa = np.zeros(N)
        
        nu = np.zeros([N,M])
        for i in range(N):
            nu[i] = (tx[i,:].T - mu_t)
            
            qa_i = sf2 / np.sqrt(np.linalg.det(S_t.dot(np.linalg.inv(lam))+np.eye(M)))
            qa_i = qa_i * np.exp(-0.5*nu[i].T.dot(np.linalg.inv(S_t+lam)).dot(nu[i]))
            qa[i] = qa_i
        mua = invCNdTy.T.dot(qa)
        
        invLam = np.linalg.inv(lam)
        R = S_t.dot(invLam + invLam) + np.eye(M)
        T = invLam + invLam + np.linalg.inv(S_t)
        invT = np.linalg.inv(T)
        detR = np.linalg.det(R)
        Q= np.zeros([N,N])

        for i in range(N):
            ka = self.kern(tx[i],mu_t)
            for j in range(N):
                kb = self.kern(tx[j],mu_t)
                zij = invLam.dot(nu[i])+invLam.dot(nu[j])
                Q[i,j] = ka * kb * np.exp(0.5*zij.T.dot(invT).dot(zij))/np.sqrt(detR)        

        varf = sf2 - np.trace(invCN.dot(Q)) + sw2
        ea2 = invCNdTy.T.dot(Q).dot(invCNdTy)
        Sa = varf + ea2 - mua.dot(mua) 

        return mua, Sa

    def kern(self,x1,x2):
        sf2 = self.sf2
        lam = self.lam
        
        ret = sf2 * np.exp(-0.5 * (x1-x2).dot(np.linalg.inv(lam)).dot(x1-x2))
        return ret


N = 5

#target x
tx=np.atleast_2d((np.random.rand(N)-0.5)*2).T

#target y
ty = dynamics(tx)

N_test = 500
y = np.atleast_2d(np.zeros(N_test)).T
x = np.atleast_2d(np.linspace(-1,1,N_test)).T

s2 = np.atleast_2d(np.zeros(N_test)).T

gp = GaussianProcess()
gp.learn(tx,ty)

for i in range(N_test):
    y[i],s2[i] = gp.predict(x[i,:].T)

mu_t = np.atleast_2d(0.1)
S_t = np.atleast_2d(0.01)
mua,Sa = gp.momentMatching(mu_t,S_t)

plt.subplot(2,2,3)
plt.plot(x,gaussian(x,mu_t,S_t))
p = plt.vlines(mu_t, 0, 4, "blue", linestyles='dashed')


plt.subplot(2,2,1)
plt.plot(tx,ty,'b.')
plt.plot(x,y)
plt.plot(x,y+np.sqrt(s2))
plt.plot(x,y-np.sqrt(s2))
p = plt.vlines(mu_t, -2, 2, "blue", linestyles='dashed')
p = plt.hlines(mua, -1, 1, "blue", linestyles='dashed') 
plt.ylim([-2,2])

plt.subplot(2,2,2)
y2 = np.linspace(-2,2,100)
plt.plot(gaussian(y2,mua[0],Sa[0]),y2)
p = plt.hlines(mua, 0, 1, "blue", linestyles='dashed')     # hlines
plt.ylim([-2,2])

plt.show()