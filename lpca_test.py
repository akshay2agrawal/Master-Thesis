from PCAfold import LPCA
import numpy as np

# Generate dummy data set:
X = np.random.rand(100,10)

# Generate dummy vector of cluster classifications:
idx = np.zeros((100,))
idx[50:80] = 1
idx = idx.astype(int)

# Instantiate LPCA class object:
lpca_X = LPCA(X, idx, scaling='none', n_components=2)

# Access the local covariance matrix in the first cluster:
S_k1 = lpca_X.S[0]

# Access the local eigenvectors in the first cluster:
A_k1 = lpca_X.A[0]

# Access the local eigenvalues in the first cluster:
L_k1 = lpca_X.L[0]

# Access the local principal components in the first cluster:
Z_k1 = lpca_X.principal_components[0]

# Access the local loadings in the first cluster:
l_k1 = lpca_X.loadings[0]

# Access the local variance accounted for in each individual variable in the first cluster:
tq_k1 = lpca_X.tq[0]

X_rec = lpca_X.X_reconstructed
from sklearn.metrics import mean_squared_error
# print(mean_squared_error(decoded_data, input_data))
print(mean_squared_error(X_rec, X))
print(X.shape)