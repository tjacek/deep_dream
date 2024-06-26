import os
import dtw,seq


def dtw_exp(in_path:str,out_path:str):
    seq_dict=seq.read_seq(in_path)
    seq_dict= seq_dict.subset(10)
    print(type(seq_dict))
    dtw_pairs=dtw.make_pairwise_distance(seq_dict)
    os.mkdir(out_path)
    dtw_pairs.save(f'{out_path}/pairs')
    return dtw_pairs

in_path="data/MSR/corl/seqs.npz"
dtw_exp(in_path,'test')