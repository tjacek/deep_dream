import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm

def lda_coff(in_path):
    df=pd.read_csv(in_path)
   
#    df['label']=0
#    df['label'][df['ginie']>0.5]=1

    feats_names=['class','samples','features','ginie']
    X=df[feats_names].to_numpy()
    X=preprocessing.scale(X,axis=0)
    print(np.sum(X,axis=0))
    y=df[['label']].to_numpy()

    clf = LogisticRegression(random_state=0)
    clf.fit(X, y)
    coef= clf.coef_/np.sum(np.abs(clf.coef_))
    for i,coff_i in enumerate(coef.tolist()[0]):
        print(f'{feats_names[i]},{coff_i:.4f}')

lda_coff('csv/sig_uci.csv')
lda_coff('csv/best_uci.csv')
lda_coff('csv/best_openml.csv')