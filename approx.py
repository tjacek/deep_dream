import json,time
from dtaidistance import dtw_ndim
import dtw,seq,utils


def dtw_exp(in_path:str,
	        out_path:str,
	        distance_type="base"):
    seq_dict=seq.read_seq(in_path)
#    seq_dict= seq_dict.subset(100)
    distance_fun=get_distance_fun(distance_type)
    start = timeit.timeit()
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
    if(distance_type=="base"):
        return dtw_ndim.distance

def eval_pairs(in_path):
	dtw.read_pairs(in_path)

in_path="data/MSR/corl/seqs.npz"
dtw_exp(in_path,'test')