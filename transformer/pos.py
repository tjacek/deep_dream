import numpy as np 


class Positional(object):
    def __init__(self,n):
    	self.n=n

	def p(self,k,pos):
		if((pos % 2)==0):
			den= self.n**(pos/s )
            return np.sin(k/(den ) )
		else:
			den= self.n**( (pos-1)/s )
            return np.cos(k/(den ) )