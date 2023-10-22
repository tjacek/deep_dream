import numpy as np
import utils

class SeqDict(dict):
    def __init__(self, arg=[]):
        super(SeqDict, self).__init__(arg)
    
    def dim(self):
        seq_i= list(self.values())[0]
        return seq_i.shape[-1]

def read_seq(in_path):
    seq_dict=SeqDict()
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        ext_i=name_i.split('.')[-1]
        if(ext_i=='txt'):
            seq_i=np.loadtxt(fname=path_i, 
                             delimiter=',')
        else:
            seq_i= np.load(path_i)
        seq_dict[name_i]=seq_i
    return seq_dict

in_path='../DTW/3DHOI/seqs/corl'
seq_dict= read_seq(in_path)
print(seq_dict.dim())