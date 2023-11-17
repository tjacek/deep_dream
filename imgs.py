import numpy as np
import cv2
from scipy.stats import skew,pearsonr
import utils

class ActionDict(dict):
    def __init__(self, arg=[]):
        super(ActionDict, self).__init__(arg)
    
    def compute_feats(self,extractors):
        def helper(frame_j):
            points=nonzero_points(frame_j)
            
        hc_dict={}
        for name_i,action_i in self.items():
            seq_i=[ helper(frame_j)
                        for frame_j in seq_i]
            hc_dict[name_i]=seq_i
        return hc_dict

    def len_dict(self)->dict:
        return { name_i:len(seq_i)
                  for name_i,seq_i in self.items()}

def read_action(in_path:str) -> ActionDict:
    action_dict=ActionDict()
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        action_i=[cv2.imread(path_j,0)
                for path_j in utils.top_files(path_i)]
        action_dict[name_i]=action_i
    return action_dict

def nonzero_points(frame_i):
    xy_nonzero=np.nonzero(frame_i)
    z_nozero=frame_i[xy_nonzero]
    xy_nonzero,z_nozero=np.array(xy_nonzero),np.expand_dims(z_nozero,axis=0)
    return np.concatenate([xy_nonzero,z_nozero],axis=0)

def moments(points):
    std_i=list(np.std(points,axis=1))
    skew_i=list(skew(points,axis=1))
    return std_i+skew_i

def corl(points):
    x,y,z=points[0],points[1],points[2]
    return [pearsonr(x,y)[0],pearsonr(z,y)[0],pearsonr(x,z)[0]]

in_path='../MSR/frames'
action_dict= read_action(in_path)
print(action_dict.len_dict())