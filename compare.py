import pandas as pd 
import numpy as np
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, os

my_path = os.path.dirname(__file__)
save_path = my_path+'\\experimental results\\cubic\\'

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


# Calculate the absolute difference between the two MSE values
abs_diff_mse = abs(mse_vqpca - mse_ae)
# Saving the objects:
with open(f'{save_path}abs_diff_mse.pkl', 'wb') as f:  # In Python 3, use: open(..., 'wb')
    pickle.dump(abs_diff_mse, f)
    f.close()
print(f'-------------------------------------\n MSE difference: {abs_diff_mse}')


# Load the distance matrix of the autoencoder
distance_matrix_ae = pd.read_csv(save_path+'ae_pairwise_distance.csv')
distance_matrix_vqpca = pd.read_csv(save_path+'vqpca_pairwise_distance.csv')
count =0 
count_black = 0
abs_diff = abs(distance_matrix_ae - distance_matrix_vqpca)    
abs_diff_arr = abs_diff.to_numpy()
binary_abs_diff = np.zeros_like(abs_diff_arr)
for i in range(0, len(abs_diff_arr)):
    for j in range(0,len(abs_diff_arr)):
        if abs_diff_arr[i][j] > 0.1:
            binary_abs_diff[i][j] = 1 # distances between the reduced cluster
            count_black = count_black+1
        else:
            count = count+1
print(f'count of < 0.1 is: {count}')
print(f'count of > 0.1 is: {count_black}')
# print(binary_abs_diff)
#plotting the heatmap for the binary distance matrix
plt.figure(figsize=(10, 10))
sns.heatmap(binary_abs_diff, cmap='Greys')
plt.title('Heatmap of Binary Distance Matrix')
plt.xlabel('Points')
plt.ylabel('Points')
plt.savefig(save_path+'abs_diff_binary_heatmap_0.1.png')
plt.clf()


# Calucalte the frobenius norm of the difference matrix
frobenius_norm = LA.norm(abs_diff)/(500*500)
# Saving the objects:
with open(f'{save_path}frobenius_norm.pkl', 'wb') as f:  # In Python 3, use: open(..., 'wb')
    pickle.dump(frobenius_norm, f)
    f.close()

print(f'-------------------------------------\n Frobenius norm: {frobenius_norm}')
# print(abs_diff)
# plot the heatmap of the difference matrix
sns.heatmap(abs_diff, cmap='gray_r')
plt.savefig(f'{save_path}abs_diff_heatmap.png')
plt.show()

