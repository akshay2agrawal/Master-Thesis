#imports for plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# load data
# from sklearn.datasets import load_iris, load_wine, load_breast_cancer
# from sklearn.preprocessing import MinMaxScaler


save_path = 'D:\\Masters\\Thesis\\vqpca\\experimental results\\digits\\'
# np.random.seed(34)
# x = np.linspace(-10, 10, 500)  # generate 100 evenly spaced values of x between -10 and 10
# y = x**2 + np.random.normal(0, 10, 500)  # compute the corresponding y values
# X = np.array(list(zip(x,y)))
# scaler = MinMaxScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#loading the preprocessed data form the VQPCA script (used center 0to1 scaling)
preprocessed_data = np.loadtxt(save_path+'preprocessed_data.csv', delimiter=',')
X_scaled = preprocessed_data

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# input_dim = X_scaled.shape[1]
input_dim = X_scaled.shape[1]
encoding_dim = 2  
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_scaled, X_scaled,
# history = autoencoder.fit(X, X,
                epochs=1000,
                batch_size=16,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)

# use our encoded layer to encode the training input
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# lower-dimensional projection of original (full dimensional) data to the learned bottleneck aka latent space/layer of the AE
# encoded_data = encoder.predict(X)
encoded_data = encoder.predict(X_scaled)
# back-projection of lower-dimensional latent layer representation to full dimensional data
decoded_data = decoder.predict(encoded_data)


# plot3clusters(encoded_data[:,:2], 'Linear AE', 'AE')

X_scaled_zip = list(zip(*X_scaled))
X_sc1, X_sc2 = list(X_scaled_zip[0]), list(X_scaled_zip[1])
# print(X_sc1, X_sc2)

X_scaled_reconstructed = list(zip(*decoded_data))
X_sc_r1, X_sc_r2 = list(X_scaled_reconstructed[0]), list(X_scaled_reconstructed[1])
# print(X_sc_r1,X_sc_r2)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('X_scaled vs Reconstructed')
ax1.plot(X_sc1, X_sc2)
ax1.set_title("X_scaled [0-1]")
ax2.plot(X_sc_r1, X_sc_r2)
ax2.set_title("Reconstructed")
# ax3.plot(x,y)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(decoded_data, X_scaled)
print(mse)

from sklearn.metrics import pairwise_distances

distance_matrix = pairwise_distances(encoded_data)
print(distance_matrix.shape, '\n ', distance_matrix)

pairwise_distance = pd.DataFrame(distance_matrix)
pairwise_distance.to_csv(save_path+'ae_pairwise_distance.csv')


import pickle

# Saving the objects:
with open(save_path+'ae_mse.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(mse, f)
    f.close()

plt.figure(figsize=(10, 10))
sns.heatmap(pairwise_distance, cmap='Greys')
plt.title('Heatmap of Distance Matrix')
plt.xlabel('Points')
plt.ylabel('Points')
plt.show()
plt.savefig(save_path+'ae_heatmap.png')