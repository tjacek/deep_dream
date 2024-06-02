import numpy as np
from scipy.special import softmax

class AttenMock(object):
    def __init__(self,embed_dim=11,
                      max_len=100):#,
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
        p=softmax( np.dot(Q,K.T)) 
        A=np.dot(p,V)
        print(A.shape)


atten=AttenMock()
atten.sample()
atten.standard()