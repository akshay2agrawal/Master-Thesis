import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


my_path = os.path.dirname(__file__)
save_path = my_path+'\\experimental results\\breast cancer\\'


def plot3DGridSurface(k_bc, dim_bc, mse_bc,dset):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  triang = mtri.Triangulation(k_bc, dim_bc)

  ax.plot_trisurf(triang, mse_bc, cmap='jet')
  ax.scatter(k_bc, dim_bc, mse_bc, marker='.', s=10, c="black", alpha=0.5)
  ax.view_init(elev=60, azim=-45)

  ax.set_xlabel('k_bc')
  ax.set_ylabel('dim_bc')
  ax.set_zlabel('mse_bc')
  plt.title(dset)
  plt.savefig(save_path+f'{dset}_3d.png')
  plt.show() 

if __name__ == "__main__":

    bc_arr, cluster, dimension, mse = [], [], [], []

    for cluster_size in range(2,6):
        for dim in range(1,6):
            loaded_arr = np.loadtxt(save_path+f'{cluster_size}{dim}_combined_arr.out', delimiter=',')
            bc_arr.append(loaded_arr)
            pass

    for i in range(len(bc_arr)):
        cluster.append(bc_arr[i][0])
        dimension.append(bc_arr[i][1])
        mse.append(bc_arr[i][2])

    print(cluster)
    print(dimension)
    print(mse)
    plot3DGridSurface(cluster, dimension, mse,'Breast cancer Dataset')
    print(">ok")
