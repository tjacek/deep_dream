from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report#
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dtw,seq,utils

def dtw_knn(in_path:str,verbose=0):
    all_pairs=read_pairs(in_path)
    lines=[]
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=pairs_i.knn()
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'dtw_knn,{type_i},{metric_i}')
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
        y_pred,y_test=train_clf(feat_i)
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'dtw_feats,{type_i},{metric_i}')
    if(verbose):
        print('\n'.join(lines))
    return lines

def hc_feats(in_path,verbose=0):
    all_feats=read_feats(in_path)
    lines=[]
    for type_i,feat_i in all_feats.items():
        y_test,y_pred=train_clf(feat_i)
#        metric_i=get_metrics(y_test,y_pred)
        metric_i=partial_metrics(y_test,y_pred)
        lines.append(f'hc_feats,{type_i},{metric_i}')
    if(verbose):
        print('\n'.join(lines))    
    return lines

def read_feats(in_path):
    all_seqs={ path_i.split('/')[-1]:
                seq.read_seq(f'{path_i}/seqs')
          for path_i in utils.top_files(in_path)}
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
    clf_i=SVC()
    clf_i.fit(X_train,y_train)
    y_pred=clf_i.predict(X_test)
    return y_test,y_pred

def base_exp(in_path,datasets=None):
    if(datasets is None):
        datasets=['MSR','MHAD','3DHOI']
    algs=[hc_feats]#hc_feats,dtw_knn,dtw_feats]
    for data_i in datasets:
        path_i=f'{in_path}/{data_i}'
        for alg_j in algs:
            lines=alg_j(path_i)
            lines=[ f'{data_i},{line_k}' 
                    for line_k in lines]
            print('\n'.join(lines))

def compare_knn(in_path):
    all_pairs=read_pairs(in_path)
    metric_dict={}
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=pairs_i.knn()
        metric_i=partial_metrics(y_test,y_pred)
        metric_dict[type_i]=metric_i
    df= pd.DataFrame.from_dict(metric_dict)
    show_bar(df,k=0)

def show_bar(df,step=10,k=0):
    fig, ax = plt.subplots()
    x=np.arange(step)+k*step +1
    diff=[0.2,0.0,-0.2,-0.4]
    labels={'corl':'I',
            'max_z':'II',
            'std':"III",
            'skew':"IV"}
    for i,col_i in enumerate(['corl','max_z','std','skew']):
        feat_i=df[col_i].tolist()
        feat_i=feat_i[step*k:step*(k+1)]
        plt.bar(x - diff[i],feat_i, 0.2, 
                label = labels[col_i]) 
    plt.title('DTW KNN on MSR-Action3D dataset')
    plt.xticks(x, [str(x_i) for x_i in x]) 
    plt.xlabel("Class") 
    plt.ylabel("Accuracy") 
    plt.legend() 
    plt.show()

def show_scatter(df):
    fig, ax = plt.subplots()
    x=np.arange(len(df))
    colors=['blue','red','green','yellow']
    for i,col_i in enumerate(df.columns):
        feat_i=df[col_i].tolist()
        ax.scatter(x=x, 
                   y=feat_i, 
                   c=colors[i],
                   vmin=0, 
                   vmax=100)
    plt.show()

in_path=f'../DTW'#{datasets[k]}'
print(in_path)
compare_knn(f'{in_path}/MSR')
#base_exp(in_path)