#import dtaidistance #import dtw, dtw_ndim
import numpy as np
from dtaidistance import dtw_ndim
import json
import seq,utils

class DTWpairs(object):
    def __init__(self,pairs):
        if(type(pairs)==list):    
            pairs={name_i:{name_i:0.0}
                        for name_i in pairs}
        if(pairs is None):
            pairs={}
        self.pairs=pairs	
    
    def __len__(self):
        return len(self.pairs)

    def set(self,key1,key2,data_i):
        self.pairs[key1][key2]=data_i

    def get(self,key1,key2):
        return self.pairs[key1][key2]

    def get_features(self):
        names=list(self.pairs.keys())
        train,test=utils.split(names)
        feat_dict={}
        for name_i in names:
            feat_i=np.array([ self.pairs[name_i][train_j]
                        for train_j in train])
            feat_dict[name_i]=feat_i
        return seq.FeatDict( feat_dict)
    
    def knn(self,k=1):
        names=list(self.pairs.keys())
        train,test=utils.split(names)
        y_true,y_pred=[],[]
        for name_i in test:
            feat_i=np.array([ self.get(name_i,train_j)
                        for train_j in train])
            best=feat_i.argsort()[0]
            cat_i=utils.get_cat(train[best])
            y_pred.append(cat_i)
            y_true.append( utils.get_cat(name_i))
        return y_true,y_pred

    def save(self,out_path:str):
    	with open(out_path, 'w') as f:
            json.dump(self.pairs,f)

    def __str__(self):
        return str(self.pairs)	

def make_pairs(in_path,out_path=None):
    seq_dict=seq.read_seq(in_path)
    dtw_pairs=make_pairwise_distance(seq_dict)
    if(not out_path is None):
        dtw_pairs.save(out_path)
    return dtw_pairs

def read_pairs(in_path):
    with open(in_path) as f:
        raw_pairs = json.load(f)
        return DTWpairs(raw_pairs)

def make_pairwise_distance(seq_dict):
    names=list(seq_dict.keys())#[:30]
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

def basic_exp(in_path):
    datasets=['MSR','MHAD','3DHOI']
    for data_i in datasets:
        data_dir=f'{in_path}/{data_i}'  	
        for path_i in utils.top_files(data_dir):
            seq_i=f'{path_i}/seqs'
            pairs_i=f'{path_i}/pairs'
            print(path_i)
            make_pairs(seq_i,pairs_i)

if __name__ == "__main__":
    in_path='../DTW'#3DHOI/seqs/corl'
    basic_exp(in_path)