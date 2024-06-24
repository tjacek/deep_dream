import sklearn
import seaborn as sn
import matplotlib.pyplot as plt
import compare,train

def gen_cf(in_path,alg_type='DTW-FEATS'):
    all_pairs=train.read_pairs(in_path)
    alg=compare.get_alg(alg_type)
    for type_i,(y_pred,y_test) in alg(all_pairs):
        if(type_i=='all'):
            print(type_i)
            cf=sklearn.metrics.confusion_matrix(y_test, y_pred)
            print(cf)
#            plt.rcParams.update({'font.size': 22})
            sn.heatmap(cf,
                       cmap="YlGnBu",
                       annot=True,
                       annot_kws={"size": 12}, 
                       fmt='g')
            plt.show()

in_path=f'data'
gen_cf(f'{in_path}/MSR',alg_type='Cechy DTW')    