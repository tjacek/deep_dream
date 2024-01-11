import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

def box_plot(csv_path,name):
    df=pd.read_csv(csv_path)
    datasets,labels=clean_data(df)
    plt.rcParams.update({'font.size': 12})
#    fig = plt.figure(figsize =(10, 8))
#    ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.boxplot(datasets,
               notch=False, 
               patch_artist=True)
    ax.set_title(f'Niezbalansowanie klas w zbiorach {name}')
    ax.set_xlabel('Zbiór danych')
    ax.set_ylabel('Próbki w klasie [%]')
    ax.set_xticklabels(labels,
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor')
    plt.tight_layout()
    plt.show()  
    plt.close()

def clean_data(df):
    datasets,labels=[],[]
    for i,row_i in df.iterrows():
        row_i=row_i.to_list()
        datasets.append([ 0.01*value_j 
            for value_j in row_i[1:]
               if(value_j!=0)])
        labels.append(row_i[0])
    return datasets,labels

def box_plot2(csv_path,name):
    df=pd.read_csv(csv_path,index_col="dataset")
    df=df.transpose()
    plt.rcParams.update({'font.size': 15})
    boxplot=df.boxplot()
    boxplot.plot()
    boxplot.set_title(f'Niezbalansowanie klas w zbiorach {name}')
    plt.show()

def violin_plot(csv_path,name):
    df=pd.read_csv(csv_path)
    datasets,labels=clean_data(df)
    fig, ax = plt.subplots()
    sales=plt.violinplot(datasets, 
                         showextrema=True, 
                         showmeans=True, 
                         showmedians=True)
    ax.set_xticklabels(labels,
                       rotation=45,
                       ha='right',
                       rotation_mode='anchor')
    ax.set_title(f'Niezbalansowanie klas w zbiorach {name}')
    ax.set_ylabel('[%]')
    plt.show()

name='OpenML'
csv_path='csv/openml_imb.csv'
box_plot(csv_path,name)