from PCAfold import PCA, plot_eigenvectors
import numpy as np

# Generate dummy data set:
X = np.random.rand(100,3)

# Perform PCA and obtain eigenvectors:
pca_X = PCA(X, n_components=2)
eigenvectors = pca_X.A

# Plot second and third eigenvector:
plts = plot_eigenvectors(eigenvectors[:,[1,2]],
                         eigenvectors_indices=[1,2],
                         variable_names=['$a_1$', '$a_2$', '$a_3$'],
                         plot_absolute=False,
                         rotate_label=True,
                         bar_color=None,
                         title='PCA on X',
                         save_filename='PCA-X.pdf')
plts[0].show()
plts[1].show()