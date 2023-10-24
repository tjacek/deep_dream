from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report#precision_recall_fscore_support
from sklearn.svm import SVC
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
        train_dict,test_dict=feat_dict.split()
        X_train,y_train=train_dict.as_dataset()
        X_test,y_test=test_dict.as_dataset()
        clf_i=SVC()
        clf_i.fit(X_train,y_train)
        y_pred=clf_i.predict(X_test)
        metrics_i=classification_report(y_test,y_pred)
        print(type_i)
        print(metrics_i)

datasets=['MSR','MHAD','3DHOI']
k=1
in_path=f'../DTW/{datasets[k]}'
print(in_path)
hc_exp(in_path)