import seq.io
import numpy as np
from scipy.interpolate import CubicSpline

class SplineUpsampling(object):
    def __init__(self,new_size=128):
        self.new_size=new_size

    def __call__(self,feat_i):
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
    	step=float(self.new_size)/float(old_size)
        old_x*=step  	
    	cs=CubicSpline(old_x,feat_i)
    	new_x=np.arange(self.new_size)
    	print(new_x.shape)
    	return cs(new_x)

def img_action(in_path,out_path):
    upsampling=SplineUpsampling()
    def action_helper(img_seq):
        img_seq=np.array(img_seq).T
        img_action=np.array([ upsampling(feat_i) for feat_i in img_seq])
        print(img_action.shape)
        return [img_action]
    seq.io.transform_actions(in_path,out_path,action_helper,
    	img_in=False,img_out=True,whole_seq=True)

img_action('all','imgs')