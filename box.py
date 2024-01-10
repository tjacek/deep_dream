import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

name='OpenML'
csv_path='csv/openml_imb.csv'
df=pd.read_csv(csv_path)
datasets,labels=[],[]
for i,row_i in df.iterrows():
    row_i=row_i.to_list()
    datasets.append([ 0.01*value_j 
    	for value_j in row_i[1:]
    	   if(value_j!=0)])
    labels.append(row_i[0])

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize =(10, 8))
ax = fig.add_subplot(111)
#ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])

ax.boxplot(datasets)#,labels=labels) 
ax.set_title(f'Niezbalansowanie klas w zbiorach {name}')
#fig.suptitle('Niezbalansowanie klas w zbiorach UCI')
#ax.set_xlabel('Nazwa zbioru')
ax.set_ylabel('[%]')
ax.set_xticklabels(labels,rotation=45,ha='right',rotation_mode='anchor')
#plt.xticks(rotation=90)


plt.show()  
plt.close()