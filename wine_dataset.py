#wine dataset


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

