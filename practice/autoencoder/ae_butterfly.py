import sys
sys.path.insert(0, '../../tools/keras')
from DatasetGenerator import DatasetGenerator
from ImageTransform import ImageTransform
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

pathData = '../data/leedsbutterfly/images'
pathTrain = '../data/dataset_train.txt'
pathTest = '../data/dataset_test.txt'
trainSize = 583
testSize = 249
imgWidth = 300
imgHeight = 300


transformList = []
transformList.append(ImageTransform(0, [imgWidth, imgHeight]))
transformList.append(ImageTransform(10))

datasetTrain = DatasetGenerator(pathData, pathTrain, transformList, imgWidth, imgHeight, trainSize, 'true')
datasetTest = DatasetGenerator(pathData, pathTest, transformList, imgWidth, imgHeight, testSize, 'true')

(x_train, _) = datasetTrain.generateBatch(datasetTrain.generateIndexList())
(x_test, _) = datasetTest.generateBatch(datasetTest.generateIndexList())
print(x_train.shape)

encoding_dim = 100
input_img = Input(shape=(270000,))

encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(270000, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print(len(x_train))
print(np.prod(x_train.shape[1:3]))

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train, epochs=10, batch_size=50, shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(300, 300, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(300, 300, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()