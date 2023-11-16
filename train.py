from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report#
from sklearn import preprocessing
from sklearn.svm import SVC
import numpy as np
import os.path
import dtw,seq,utils

def dtw_knn(in_path:str,verbose=0):
    all_pairs=read_pairs(in_path)
    lines=[]
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=pairs_i.knn()
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'dtw_knn,-,{type_i},{metric_i}')
    if(verbose):
        print('\n'.join(lines))
    return lines

def dtw_feats(in_path,n_feats=None,verbose=0):
    all_pairs=read_pairs(in_path)
    lines=[]
    for type_i,pairs_i in all_pairs.items():
        feat_i= pairs_i.get_features()
        if(not (n_feats is None)):
            feat_i=feat_i.selection(n_feats=n_feats)
            dim_i=n_feats
        else:
            dim_i=feat_i.dim()
        y_pred,y_test=train_clf(feat_i)
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'dtw_feats,{dim_i},{type_i},{metric_i}')
    if(verbose):
        print('\n'.join(lines))
    return lines

def hc_feats(in_path,verbose=0,partial=False):
    if(partial):
        metric_fun=partial_metrics
    else:
        metric_fun=get_metrics
    all_feats=read_feats(in_path)
    lines=[]
    for type_i,feat_i in all_feats.items():
        y_test,y_pred=train_clf(feat_i)
        metric_i=metric_fun(y_test,y_pred)
        dim_i=feat_i.dim()
        lines.append(f'hc_feats,{dim_i},{type_i},{metric_i}')
    if(verbose):
        print('\n'.join(lines))    
    return lines

def read_feats(in_path):
    all_seqs={ name_i:seq.read_seq(f'{path_i}/seqs')
          for name_i,path_i in utils.iter_paths(in_path)
            if(os.path.exists(f'{path_i}/seqs'))}
    all_feats={name_i:seqs_i.as_features()
                 for name_i,seqs_i in all_seqs.items()}
    all_feats['all']=seq.concat_feat(all_feats)
    return all_feats

def read_pairs(in_path):
    return { path_i.split('/')[-1]:
                dtw.read_pairs(f'{path_i}/pairs')
              for path_i in utils.top_files(in_path)} 

def get_metrics(y_true,y_pred):
    acc_i=accuracy_score(y_true,y_pred)
    metrics_i=precision_recall_fscore_support(y_true=y_true,
                                              y_pred=y_pred,
                                              average='macro')
    metrics_i=[acc_i]+list(metrics_i)[:-1]
    metrics_i=','.join([ f'{m_j:.4f}' for m_j in metrics_i])
    return metrics_i

def partial_metrics(y_true,y_pred):    
    cf = confusion_matrix(y_true, y_pred)
    partial_acc=cf.diagonal()/cf.sum(axis=1)
    return partial_acc

def train_clf(feat_dict):
    train_dict,test_dict=feat_dict.split()
    X_train,y_train=train_dict.as_dataset()
    X_test,y_test=test_dict.as_dataset()
#    X_train=utils.rescale(X_train,type="standard")
    X_train= preprocessing.RobustScaler().fit_transform(X_train)
    X_test= preprocessing.RobustScaler().fit_transform(X_test)
    clf_i=SVC()
    clf_i.fit(X_train,y_train)
    y_pred=clf_i.predict(X_test)
    return y_test,y_pred

def rescale(X,type='robust'):
    if(type=='robust'):
        X=preprocessing.RobustScaler().fit_transform(X)
    else:
#        X=VarianceThreshold(threshold=0).fit_transform(X)
        X = preprocessing.StandardScaler().fit_transform(X)
    return X

def base_exp(in_path,datasets=None):
    if(datasets is None):
        datasets=['MSR','MHAD','3DHOI']
    algs=[hc_feats,dtw_knn,dtw_feats]
    txt=[]
    for data_i in datasets:
        path_i=f'{in_path}/{data_i}'
        for alg_j in algs:
            lines=alg_j(path_i)
            lines=[ f'{data_i},{line_k}' 
                    for line_k in lines]
            txt.append('\n'.join(lines))
            print(txt[-1])
    print("**************************")
    for txt_i in txt:
        print(txt_i)

def rfe_exp(in_path,n_feats=350,datasets=None):
    if(datasets is None):
        datasets=['3DHOI']#,'MSR','MHAD']
    def helper(path_j):
        pairs_j=dtw.read_pairs(f'{path_j}/pairs')
        feat_j=pairs_j.get_features()
        feat_j=feat_j.selection(n_feats=100)
        return feat_j
    for data_i in datasets:
        path_i=f'{in_path}/{data_i}'
        feat_dict={ name_j:helper(path_j)
                    for name_j,path_j in utils.iter_paths(path_i)
                      if(name_j!='all')}
        all_feats=seq.concat_feat(feat_dict)
        y_pred,y_test=train_clf(all_feats)
        metric_i=get_metrics(y_test,y_pred)
        print(f'dtw_feats,{all_feats.dim()},all,{metric_i}')

if __name__ == "__main__":
    in_path='../DTW/'
    rfe_exp(in_path)