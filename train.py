from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report#precision_recall_fscore_support
import dtw,seq,utils

def read_all_feats(in_path:str):
    all_feats={ path_i.split('/')[-1]:
                dtw.read_pairs(f'{path_i}/pairs')
        for path_i in utils.top_files(in_path)}    
    for type_i,pairs_i in all_exp.items():
        y_pred,y_true=pairs_i.knn()
#        acc_i=accuracy_score(y_pred,y_true)
        metrics_i=classification_report(y_true,y_pred)
        print(type_i)
        print(metrics_i)
#        print(f'{type_i}:{acc_i:.2f}')

def hc_exp(all_exp):
    all_seqs={ path_i.split('/')[-1]:
                seq.read_seq(f'{path_i}/seqs')
        for path_i in utils.top_files(in_path)}
    for type_i,seqs_i in all_seqs.items():
        feat_dict= seqs_i.as_features()
        print(feat_dict)

datasets=['MSR','MHAD','3DHOI']
k=2
in_path=f'../DTW/{datasets[k]}'
print(in_path)
hc_exp(in_path)