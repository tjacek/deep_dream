import numpy as np
from scipy.special import softmax

class AttenMock(object):
    def __init__(self,embed_dim=7,
                      max_len=10):#,
#                       token_dim:int=13):
#        self.key_dim=key_dim
#        self.value_dim=value_dim
        self.embed_dim=embed_dim
        self.max_len=max_len
#        self.token_dim=token_dim
        self.W_Q=None
        self.W_K=None
        self.W_V=None
        self.x=None

    def sample(self):
        self.W_Q=np.random.rand(self.embed_dim,
        	                    self.embed_dim)
        self.W_K=np.random.rand(self.embed_dim,
        	                    self.embed_dim)
        self.W_V=np.random.rand(self.embed_dim,
        	                    self.embed_dim)
        self.x=np.random.rand(self.max_len,
        	                  self.embed_dim)

    def standard(self):
        Q=np.dot(self.x,self.W_Q)
        K=np.dot(self.x,self.W_K)
        V=np.dot(self.x,self.W_V)
        p=softmax( np.dot(Q,K.T)/np.sqrt(self.embed_dim),axis=1)
        return np.dot(p,V)

    def cs(self):
        y=[]
        for x_o in self.x:
            q_o= np.dot(x_o,self.W_Q)
            def helper(x_i):
                k_i= np.dot(x_i,self.W_K)
                return np.dot(q_o,k_i.T)
            p=np.array([helper(x_i) for x_i in self.x])
            p/= np.sum(p)
            p=softmax( p /np.sqrt(self.embed_dim))
            y_o=np.zeros(x_o.shape)
            for i,x_i in enumerate(self.x):
                print(x_i.shape)
                y_o+=  p[i]*np.dot(x_i, self.W_V)
            y.append(y_o)
        return np.array(y)



atten=AttenMock()
atten.sample()
#print(atten.cs().shape)
diff=atten.standard()-atten.cs()
print(np.mean(np.abs(diff)))
print(atten.standard())