from PCAfold import PCA, plot_heatmap
import numpy as np

# Generate dummy data set:
X = np.random.rand(100,5)

# Perform PCA and obtain the covariance matrix:
pca_X = PCA(X)
covariance_matrix = pca_X.S

# Define ticks:
ticks = ['A', 'B', 'C', 'D', 'E']

# Plot a heatmap of the covariance matrix:
plt = plot_heatmap(covariance_matrix,
                   annotate=True,
                   text_color='w',
                   format_displayed='%.1f',
                   x_ticks=ticks,
                   y_ticks=ticks,
                   title='Covariance',
                   save_filename='covariance.pdf')
plt.show()