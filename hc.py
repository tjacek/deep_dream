import numpy as np
from scipy.stats import skew,pearsonr

def moments(points):
    std_i=list(np.std(points,axis=1))
    skew_i=list(skew(points,axis=1))
    return std_i+skew_i

def corl(points):
    x,y,z=points[0],points[1],points[2]
    return [pearsonr(x,y)[0],pearsonr(z,y)[0],pearsonr(x,z)[0]]