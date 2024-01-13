import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import train

def compare_knn(in_path,alg_type='1-NN-DTW'):
    all_pairs=train.read_pairs(in_path)
    metric_dict={}
    alg=get_alg(alg_type)
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=alg(pairs_i)
        metric_i=train.partial_metrics(y_test,y_pred)
        metric_dict[type_i]=metric_i
    df= pd.DataFrame.from_dict(metric_dict)
    show_bar(df,step=20,k=0,alg_type=alg_type)

def get_alg(alg_type):
    if(alg_type=='1-NN-DTW'):
        return lambda pairs_i:pairs_i.knn()
    def dtw_feats(pairs_i):
        feat_i= pairs_i.get_features()
        print(len(feat_i))
        feat_i=feat_i.selection(n_feats=350)
        print(len(feat_i))
        return train.train_clf(feat_i)
    return dtw_feats

def show_bar(df,step=10,k=0,alg_type='1-NN-DTW'):
    title=f'{alg_type} dla MSR-Action3D'# - klasy 1-10'
    xlabel='Klasa'
    ylabel='Dokładność'
    fig, ax = plt.subplots()
    x=np.arange(step)+k*step +1
    plt.rcParams.update({'font.size': 20})
    diff=[0.3,0.15,0.0,-0.15,-0.3]
    labels={'corl':'I',
            'max_z':'II',
            'std':"III",
            'skew':"IV"}
    labels['all']='I+II+III+IV'
    for i,col_i in enumerate(labels.keys()):
        feat_i=df[col_i].tolist()
        feat_i=feat_i[step*k:step*(k+1)]
        plt.bar(x - diff[i],feat_i, 0.15, 
                label = labels[col_i]) 
    plt.title(title)#'1-NN-DTW on MSR-Action3D dataset')
    plt.xticks(x, [str(x_i) for x_i in x],fontsize=20) 
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel(xlabel,fontsize=20) 
    plt.ylabel(ylabel,fontsize=20) 
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

if __name__ == "__main__":
    in_path=f'../DTW'#{datasets[k]}'
    print(in_path)
    compare_knn(f'{in_path}/MSR',alg_type='DTW FEATS')