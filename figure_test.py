# NLPCA architecture diagram
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

layer_labels = ['Input', 'Mapping', 'Bottleneck', 'Demapping', 'Output']
layer_sizes = [None, 'M1', 'M', 'M2', 'N']

fig, ax = plt.subplots(figsize=(6,4)) 

for i in range(5):
    ax.annotate(layer_labels[i], 
                xy=(0.5, i), 
                xycoords='axes fraction',
                va="center",
                ha="center")
    
    ax.annotate(layer_sizes[i],
                xy=(0.25, i),
                xycoords='axes fraction',
                va="center",
                ha="center")

ax.set_xlim(0,1)
ax.set_ylim(0,5)
ax.axis('off')
fig.savefig('nlpca_architecture.svg')


# PCA vs NLPCA projection figure

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve

X, y = make_s_curve(n_samples=100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('Linear PCA')
ax1.scatter(X[:,0], X[:,1])
ax1.axis('square') 

ax2.set_title('Nonlinear PCA')
ax2.scatter(X[:,0], X[:,1])
ax2.axis('square')

fig.savefig('pca_vs_nlpca.svg')