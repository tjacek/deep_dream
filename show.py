from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import train

pairs=train.read_pairs('../DTW/MSR')
corl_dict=pairs['max_z']
feat_dict=corl_dict.get_features()
X,y=feat_dict.as_dataset()

X_emb = TSNE(n_components=2, 
	         learning_rate='auto',
             init='random', 
             perplexity=3).fit_transform(X)
fig, ax = plt.subplots()
ax.scatter(x=X_emb[:,0], 
           y=X_emb[:,1])
plt.show()
#print(X_emb[:,0].shape)

