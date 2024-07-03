import numpy as np
from scipy.stats import skew,pearsonr
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.svm import SVR
import utils

class SeqDict(dict):
    def __init__(self, arg=[]):
        super(SeqDict, self).__init__(arg)
    
    def dim(self):
        seq_i= list(self.values())[0]
        return seq_i.shape[-1]

    def as_features(self):
        def compute_feat(seq_i):
            mean_i=np.mean(seq_i,axis=0)
            std_i=np.std(seq_i,axis=0)
            skew_i=skew(seq_i,axis=0)

            all_feats=[mean_i,std_i,skew_i]
            return np.concatenate(all_feats,axis=0)
        feat_dict={name_i:compute_feat(seq_i)
                    for name_i,seq_i in self.items()
                    if(len(seq_i.shape)>1)}
        return FeatDict(feat_dict)

    def subset(self,keys):
        if(type(keys)==int):
            keys= list(self.keys())[:keys]
        seq_subset=SeqDict()
        for key_i in keys:
            seq_subset[key_i]=self[key_i]
        return seq_subset

class FeatDict(dict):
    def __init__(self, arg=[]):
        super(FeatDict, self).__init__(arg)

    def dim(self):
        feat_i= list(self.values())[0]
        return feat_i.shape[-1]

    def as_dataset(self):
        X,y=[],[]
        for name_i,data_i in self.items():
            if(type(data_i)!=np.ndarray):
                break
            data_i=np.nan_to_num(data_i)
            if(len(data_i)>0):
                X.append(np.nan_to_num(data_i))
                y.append(utils.get_cat(name_i))
        return np.array(X),y

    def selection(self,n_feats=100):
        estimator = LogisticRegression(solver='liblinear')#SVR(kernel="linear")
        selector = RFE(estimator=estimator, 
                       n_features_to_select=n_feats, 
                       step=1)
        train,test=self.split()
        X_train,y_test=train.as_dataset()
        X_train= preprocessing.RobustScaler().fit_transform(X_train)

        print(X_train.shape)
        selector = selector.fit(X_train, y_test)
        X,y=self.as_dataset() 
        X= preprocessing.RobustScaler().fit_transform(X)

        new_X=selector.transform(X)
#        new_X=[x_i for x_i in new_X]
        names=self.names()
        raw_dict={name_i:x_i 
             for name_i,x_i in zip(names,new_X)}
        return FeatDict(raw_dict)

    def names(self):
        names=list(self.keys())
        names.sort()
        return names

    def split(self):
        names=self.names()
        train,test=utils.split(names)
        train_dict={ train_i:self[train_i]
                     for train_i in train}
        test_dict={ test_i:self[test_i]
                     for test_i in test}
        return FeatDict(train_dict),FeatDict(test_dict)    

def concat_feat(all_dicts):
    full_dict=FeatDict()
    if(type(all_dicts)==dict):
        names=  list(full_dict.values())[0].keys()
        all_dicts=all_dicts.values()
    else:
        names=  full_dict[0].keys()
#    names=  list(all_dicts.values())[0].keys()
    for name_i in names:
        all_feats=[ dict_j[name_i] 
            for dict_j in all_dicts.values()]
        full_dict[name_i]=np.concatenate(all_feats,axis=0)
    return full_dict

def read_seq(in_path):
    post=in_path.split(".")[-1]
    if(post=="npz"):
        return read_npz(in_path)
    else:
        return read_text(in_path)

def read_npz(in_path:str):
    raw_seqs=np.load(in_path)
    seq_dict=SeqDict()
    for name_i,seq_i in raw_seqs.items():
        seq_dict[name_i]=seq_i
    return seq_dict

def read_text(in_path):
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
    
def concat_seq(all_dicts):
    full_dict=SeqDict()
    if(type(all_dicts)==dict):
        names=  list(all_dicts.values())[0].keys()
        all_dicts=all_dicts.values()
    else:
        names=  all_dicts[0].keys()
    for name_i in names:
        all_seqs=[ dict_j[name_i] 
            for dict_j in all_dicts.values()]
        full_dict[name_i]=np.concatenate(all_seqs,axis=1)
    return full_dict

if __name__ == "__main__":
    in_path='../DTW/3DHOI/seqs/corl'
    seq_dict= read_seq(in_path)
    print(seq_dict.dim())