import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import train

pairs=train.read_pairs('../DTW/MSR')
corl_dict=pairs['max_z']
feat_dict=corl_dict.get_features()
X,y=feat_dict.as_dataset()
y=np.array(y)
X_emb = TSNE(n_components=2, 
	         learning_rate='auto',
             init='random', 
             perplexity=3).fit_transform(X)
fig, ax = plt.subplots()
n_cats=np.amax(y)
for cat_i in range(n_cats):
    x_i=X_emb[y==cat_i]
   
    ax.scatter(x=x_i[:,0], 
               y=x_i[:,1],
               marker=f'${cat_i}$')
plt.show()

#print(X_emb[:,0].shape)

