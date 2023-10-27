from sklearn.metrics import accuracy_score,precision_recall_fscore_support
#from sklearn.metrics import classification_report#
from sklearn.svm import SVC
import dtw,seq,utils

def dtw_knn(in_path:str):
    all_pairs=read_pairs(in_path)
    lines=[]
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=pairs_i.knn()
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'knn,{type_i},{metric_i}')
    print('\n'.join(lines))
#        acc_i=accuracy_score(y_pred,y_true)

def dtw_feats(all_exp):
    all_pairs=read_pairs(in_path)
    lines=[]
    for type_i,pairs_i in all_pairs.items():
        feat_i= pairs_i.get_features()
        y_pred,y_test=train_clf(feat_i)
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'dtw_feats,{type_i},{metric_i}')
    print('\n'.join(lines))

def hc_feats(all_exp):
    all_feats=read_feats(in_path)
    lines=[]
    for type_i,feat_i in all_feats.items():
        y_test,y_pred=train_clf(feat_i)
        metric_i=get_metrics(y_test,y_pred)
        lines.append(f'hc_feats,{type_i},{metric_i}')
    print('\n'.join(lines))    

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

def train_clf(feat_dict):
    train_dict,test_dict=feat_dict.split()
    X_train,y_train=train_dict.as_dataset()
    X_test,y_test=test_dict.as_dataset()
    clf_i=SVC()
    clf_i.fit(X_train,y_train)
    y_pred=clf_i.predict(X_test)
    return y_test,y_pred

datasets=['MSR','MHAD','3DHOI']
k=2
in_path=f'../DTW/{datasets[k]}'
print(in_path)
hc_feats(in_path)
dtw_knn(in_path)
dtw_feats(in_path)