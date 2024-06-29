import json,time
from sklearn.metrics import classification_report,accuracy_score
from dtaidistance import dtw_ndim
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import dtw,seq,train,utils

def dtw_exp(in_path:str,
	        out_path:str,
	        distance_type="base"):
    seq_dict=seq.read_seq(in_path)
#    seq_dict= seq_dict.subset(30)
    distance_fun=get_distance_fun(distance_type)
    start = time.time()
    dtw_pairs=dtw.make_pairwise_distance(seq_dict,
    	                                 distance_fun=distance_fun)
    end = time.time()
    utils.make_dir(out_path)
    dtw_pairs.save(f'{out_path}/pairs')
    with open(f'{out_path}/info', 'w') as f:
            json.dump({'dtw_alg':distance_type,
            	       'data_source':in_path,
            	        'time':end-start},
            	      f)
    return dtw_pairs

def get_distance_fun(distance_type):
    if(distance_type=="fast"):
        def dist(seq_i,seq_j):
            distance, path = fastdtw(seq_i, seq_j, dist=euclidean)
            return distance
        return dist
    return dtw_ndim.distance


def eval(paths,verbose=True):
    results=[]
    for path_i in paths:
        name_i=path_i.split('/')[-2]
        print(path_i)
        acc_i=eval_pairs(path_i,verbose=True)
        results.append((name_i,acc_i))
    print(results)

def eval_pairs(in_path,verbose=True):
    pairs=dtw.read_pairs(in_path)
    feat= pairs.get_features()
    y_true,y_pred=train.train_clf(feat)
    if(verbose):
        print(classification_report(y_true,
                                    y_pred,
                                    digits=4))
    return accuracy_score(y_true,y_pred)

def multiexp(in_path,out_path):
    feats_type=['max_z','corl','skew','std']
    dtw_types=['fast','base']
    utils.make_dir(out_path)
    for feat_i in feats_type:
        in_i=f'{in_path}/{feat_i}/seqs.npz'
        for dtw_j in dtw_types:
            out_ij=f'{in_path}/{feat_i}/{dtw_j}'
            dtw_exp(in_path=in_i,
	                out_path=out_ij,
	                distance_type=dtw_j)

#multiexp("data/MSR","data/MSR")
#in_path="data/MSR/corl/seqs.npz"
#dtw_exp(in_path,'test')
#eval_pairs('data/MSR/max_z/fast/pairs')
paths=['data/MSR/max_z/fast/pairs',
       'data/MSR/skew/fast/pairs']
eval(paths,verbose=True)