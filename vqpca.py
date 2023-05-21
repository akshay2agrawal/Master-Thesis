from PCAfold import variable_bins, plot_2d_clustering, VQPCA
import numpy as np
import os


my_path = os.path.dirname(__file__)
# print(my_path)
fig_path = my_path+'\\parabolic data\\'

data = []

for i in range(0,100):

    np.random.seed(i)

    # Generate parabolic data set:
    x = np.linspace(-10, 10, 500)
    y = x**2 + np.random.normal(0, 10, 500)
    X =np.array(list(zip(x,y)))


    try:
    # Instantiate VQPCA class object:
        vqpca = VQPCA(X,
                n_clusters=7,
                n_components=1,
                scaling='std',
                idx_init='random',
                max_iter=100,  
                tolerance=1.0e-08,
                verbose=True)
        # ``verbose=True``, the code above will print detailed information on each iteration
    except:
        continue

    # Access the VQPCA clustering solution:
    idx = vqpca.idx
    print('before plot')
    # Plot the clustering result:
    plt = plot_2d_clustering(x,
                        y,
                        idx,
                        x_label='$x$',
                        y_label='$y$',
                        color_map='viridis',
                        first_cluster_index_zero=False,
                        grid_on=True,
                        figure_size=(10,6),
                        title='x-y data set',
                        save_filename='clustering.pdf')
    print('>after plot')
    # plt.close()
    plt.savefig(f'{fig_path}test{i}.png')

"""
**Attributes:**

    - **idx** - vector of cluster classifications.
    - **collected_idx** - vector of cluster classifications from all iterations.
    - **converged** - boolean specifying whether the algorithm has converged.
    - **A** - local eigenvectors from the last iteration.
    - **principal_components** - local Principal Components from the last iteration.
    - **reconstruction_errors_in_clusters** - mean reconstruction errors in each cluster from the last iteration.

"""