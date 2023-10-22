#import dtaidistance #import dtw, dtw_ndim
from dtaidistance import dtw_ndim
import json
import seq

class DTWpairs(object):
    def __init__(self,names):
        self.pairs={name_i:{name_i:0.0}
                        for name_i in names}

    def set(self,key1,key2,data_i):
        self.pairs[key1][key2]=data_i

    def save(self,out_path:str):
    	with open(out_path, 'w') as f:
            json.dump(self.pairs,f)

def make_pairs(in_path,out_path):
    seq_dict=seq.read_seq(in_path)
    dtw_pairs=make_pairwise_distance(seq_dict)
    dtw_pairs.save(out_path)

def make_pairwise_distance(seq_dict):
    names=list(seq_dict.keys())[:30]
    dtw_pairs=DTWpairs(names)
    n_ts=len(names)
    for i in range(1,n_ts):
        print(i)
        for j in range(0,i):
            name_i,name_j=names[i],names[j]
            seq_i,seq_j=seq_dict[name_i],seq_dict[name_j]
            distance_ij=dtw_ndim.distance(seq_i,seq_j)
            dtw_pairs.set(name_i,name_j,distance_ij)
            dtw_pairs.set(name_j,name_i,distance_ij)
    return dtw_pairs  

in_path='../DTW/3DHOI/seqs/corl'
make_pairs(in_path,'test')