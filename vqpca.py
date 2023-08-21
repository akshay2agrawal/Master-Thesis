"""
**VQPCA Attributes:**

    - **idx** - vector of cluster classifications.
    - **collected_idx** - vector of cluster classifications from all iterations.
    - **converged** - boolean specifying whether the algorithm has converged.
    - **A** - local eigenvectors from the last iteration.
    - **principal_components** - local Principal Components from the last iteration./ Reduced data/ compressed data from pca result
    - **data_reconstructed** - Reconstructed scaled data
    - **X_pre_processed** - Preprocessed/scaled/normalized data
    - **clusters** - Datapoints segregated into clusters

"""

from PCAfold import plot_2d_clustering, VQPCA, plot_heatmap
import numpy as np
import os
from numpy import linalg as LA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits


def frobenius_norm(matrix):
   
   #Done for the reconstructed data
   norm  = LA.norm(matrix)
   print(f'-------------------------------------\n Frobenius norm: {norm}')
   
   return norm
   pass


def mse(og_data, rec_data):

  # between Normalized and reconstructed data
  mse = mean_squared_error(rec_data, og_data)
  print(f'--------------------------------------\n MSE: {mse}')

  return mse
  pass


def check_cluster(individual_cluster_sizes, i, j):

  cluster_range = []
  count = 0 
  for size in individual_cluster_sizes:
    cluster_range.append((count, count+size))
    count = count+size
  # print(f'Cluster_range: {cluster_range}')
  
  pos_i, pos_j = None, None
  for index in range(0,len(cluster_range)):
    if i in range(cluster_range[index][0], cluster_range[index][1]):
      pos_i = index
    if j in range(cluster_range[index][0], cluster_range[index][1]):
      pos_j = index
       
  # print(f'i belongs to cluster: {pos_i} \n j belongs to cluster: {pos_j}')
  return pos_i, pos_j


def pairwise_dist(data):
  
  distance_matrix = pairwise_distances(np.asarray(data))
  # print(distance_matrix.shape, '\n ', distance_matrix)
  
  return distance_matrix
  

def cluster_projection(eigenvectors, projecting_cluster, projectee_cluster, clustered_data, reduced_data):
  
  #projecting cluster x onto cluster y by multiplying the dataset of cluster x with the eigenvector of cluster y  
  proj = clustered_data[projecting_cluster-1]@eigenvectors[projectee_cluster-1]
  # print('>done')
  


  #projecting all the cluster data onto cluster x by multiplying the dataset of the clusters with the eigenvector of cluster x  
  projection = [clustered_data[i]@eigenvectors[projectee_cluster] for i in range(0,cluster_size) if i!=(projectee_cluster)]
  projection.insert(projectee_cluster, reduced_data) #insert the reduced/approximated/projected data of cluster x from the vqpca results.
  #Resulting projection list contains the projection of all data on the axis of cluster x
  # check = np.cumsum(np.asarray(projection))
  resultList = [element for nestedlist in projection for element in nestedlist]
  return resultList
  

def pairwise_distance_reduced_data(vqpca, cluster_size):
  overall_projections = {}
  distance_matrices = []

  for i in range(cluster_size):
    overall_projections[i] = cluster_projection(vqpca.A,
                                                 projecting_cluster=2,
                                                   projectee_cluster=i,
                                                     clustered_data = vqpca.clusters,
                                                       reduced_data=vqpca.principal_components[i])
    distance_matrices.append(pairwise_dist(overall_projections[i]))
  

  #calculating the sizes of individual clusters summing up to the size of dataset
  individual_cluster_sizes = [len(cluster) for cluster in vqpca.clusters]
  # print(f'individual cluster sizes: {individual_cluster_sizes}')
  
  
  #taking all the projections, choosing max and making one single matrix for pairwise distance
  pos_i, pos_j = check_cluster(individual_cluster_sizes=individual_cluster_sizes, i=105, j=499)
  
  pairwise_distance_matrix = np.zeros_like(distance_matrices[0])
  
  
  for i in range(0, len(pairwise_distance_matrix)):
    
    for j in range(i+1,len(pairwise_distance_matrix)):
      
      cluster_i, cluster_j = check_cluster(individual_cluster_sizes=individual_cluster_sizes, i=i, j=j)
      # print(f'cluster in which i={i} occurs: {cluster_i}, cluster in which j={j} occurs: {cluster_j}')
      
      if cluster_i==cluster_j:
        pairwise_distance_matrix[i][j] = distance_matrices[cluster_i][i][j] # distances between the reduced cluster
        
      else:
        # print(distance_matrices[cluster_i].shape, distance_matrices[cluster_j].shape)
        pairwise_distance_matrix[i][j] = max(distance_matrices[cluster_i][i][j], distance_matrices[cluster_j][i][j])  # distance between one cluster projected onto other
      
      pairwise_distance_matrix[j][i] = pairwise_distance_matrix[i][j]
        
  return pairwise_distance_matrix


#main function 
if __name__=='__main__':
  
  my_path = os.path.dirname(__file__)

  # Loading 2D data
  data = []
  i=34

  np.random.seed(i) 

  # Generate parabolic data set:
  # x = np.linspace(-10, 10, 500)
  # # y = x**3 + test
  # y = x**2 + np.random.normal(0, 10, 500)

  # X_load =np.array(list(zip(x,y)))

  # Loading Mnist fashion dataset uusing sklearn
  # from sklearn.datasets import fetch_openml

  # mnist = fetch_openml('mnist_784', version=1, as_frame=False)
  # mnist.target = mnist.target.astype(int)
  # data_mnist = mnist.data
  
  """
  digits = load_digits()
  # X_load = data_mnist
  
  X_load = load_wine().data
  """
  

  # Loading a dataset from a csv file
  seeds = pd.read_csv('seeds_dataset.txt', sep="\s+", index_col=False)
  seeds = seeds.reset_index()
  seeds_columns = seeds.columns

  categorical_features = ['wheat_class']
  numerical_features = ['area_A', 'perimeter_P', 'compactness_C', 'length_of_kernel',
        'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove']
  cat_data = seeds[categorical_features]
  num_data = seeds[numerical_features]

  X_load = num_data

  scaler = MinMaxScaler()
  scaler.fit(X_load)
  X = scaler.transform(X_load)
  print(X.shape)
  
  cluster_size = 4 #2 to 5
  dimensions = 2 #1-5
  save_path = f'{my_path}\\experimental results\\seeds\\'

  try:
    # Instantiate VQPCA class object:
      vqpca = VQPCA(X,
              n_clusters=cluster_size,
              n_components=dimensions,
              scaling='none',
              idx_init='random',
              max_iter=100,  
              tolerance=None,
              verbose=True)
  except:
    print('Error in the vqpca class')
  # Access the VQPCA clustering solution:
  idx = vqpca.idx
      

  # print('---------------------------------\nPosition of element inside cluster in preprocessed data', np.where(vqpca.X_pre_processed == vqpca.clusters[0][50])[0], vqpca.clusters[0][3])


  #function testing
  pairwise_distance = pairwise_distance_reduced_data(vqpca=vqpca, cluster_size=cluster_size)
  
  print(pairwise_distance)

  binary_pairwise_distance = np.zeros_like(pairwise_distance)
  for i in range(0, len(pairwise_distance)):
    for j in range(0,len(pairwise_distance)):
      if pairwise_distance[i][j] > 0.2:
        binary_pairwise_distance[i][j] = 1 # distances between the reduced cluster

  # print(binary_pairwise_distance)
  #plotting the heatmap for the binary distance matrix
  plt.figure(figsize=(10, 10))
  sns.heatmap(binary_pairwise_distance, cmap='Greys')
  plt.title('Heatmap of Binary Distance Matrix')
  plt.xlabel('Points')
  plt.ylabel('Points')
  plt.savefig(save_path+'vqpca_binary_heatmap_0.2.png')
  plt.clf()

  plt.figure(figsize=(10, 10))
  sns.heatmap(pairwise_distance, cmap='Greys')
  plt.title('Heatmap of Distance Matrix')
  plt.xlabel('Points')
  plt.ylabel('Points')
  plt.savefig(save_path+'vqpca_heatmap.png')

  #convert to dataframe
  pairwise_distance = pd.DataFrame(pairwise_distance)
  pairwise_distance.to_csv(save_path+'vqpca_pairwise_distance.csv')

  #saving preprocessed data
  np.savetxt(save_path+'preprocessed_data.csv', vqpca.X_pre_processed, delimiter=',')
  
  # save mse to file
  mse = mse(vqpca.X_pre_processed, vqpca.data_reconstructed)

  import pickle

  # Saving the objects:
  with open(save_path+'vqpca_mse.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
      pickle.dump(mse, f)
      f.close()

  combined_arr = [cluster_size, dimensions, mse]
  # save using savetxt
  # np.savetxt(f'{save_path}{cluster_size}{dimensions}_combined_arr.out', combined_arr, delimiter=',')

  #load the saved file
  # loaded_arr = np.loadtxt(save_path+'combined_arr.out', delimiter=',')