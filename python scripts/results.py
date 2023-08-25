import pandas as pd 
import numpy as np
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, os

my_path = os.path.dirname(__file__)
save_path = my_path+'\\experimental results\\digits\\'

# Getting back the MSE values:
# Vqpca MSE
with open(save_path+'vqpca_mse.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    mse_vqpca = pickle.load(f)
    f.close()
print(f'-------------------------------------\n VQPCA MSE: {mse_vqpca}')

# Ae MSE:
with open(save_path+'ae_mse.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    mse_ae = pickle.load(f)
    f.close()
print(f'-------------------------------------\n Autoencoder MSE: {mse_ae}')


# open the abs diff MSE

with open(save_path+'abs_diff_mse.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    abs_diff_mse = pickle.load(f)
    f.close()
print(f'-------------------------------------\n abs difference MSE: {abs_diff_mse}')



# open frobenius norm file
with open(save_path+'frobenius_norm.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    frobenius_norm = pickle.load(f)
    f.close()
print(f'-------------------------------------\n Norm: {frobenius_norm}')




