import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import train

def compare_knn(in_path):
    all_pairs=train.read_pairs(in_path)
    metric_dict={}
    for type_i,pairs_i in all_pairs.items():
        y_pred,y_test=pairs_i.knn()
        metric_i=train.partial_metrics(y_test,y_pred)
        metric_dict[type_i]=metric_i
    df= pd.DataFrame.from_dict(metric_dict)
    show_bar(df,k=0)

def show_bar(df,step=10,k=0):
    title='1-NN-DTW dla MSR-Action3D - klasy 1-10'
    xlabel='Klasa'
    ylabel='Dokładność'
    fig, ax = plt.subplots()
    x=np.arange(step)+k*step +1
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
    plt.xticks(x, [str(x_i) for x_i in x]) 
    plt.xlabel(xlabel)#("Class") 
    plt.ylabel(ylabel) #"Accuracy") 
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
    compare_knn(f'{in_path}/MSR')