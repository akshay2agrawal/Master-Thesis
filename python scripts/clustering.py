from PCAfold import variable_bins, plot_2d_clustering
import numpy as np

# Generate dummy data set:
x = np.linspace(-10, 10, 500)
y = x**3 + np.random.normal(0, 10, 500)
# X =np.array(list(zip(x,y)))
# Generate dummy clustering of the data set:
(idx, _) = variable_bins(x, 4, verbose=False)

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
plt.show()