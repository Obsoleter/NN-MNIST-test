import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow.keras as keras
from sklearn.preprocessing import scale


# MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# Scale
X_train = scale( X_train.reshape(len(X_train), -1) )
X_test = scale( X_test.reshape(len(X_test), -1) )


# NN
network = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid'),
])

network.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

network.fit(X_train, y_train, epochs=5)


# Evaluate
network.evaluate(X_test, y_test)


# CM
predicted = network.predict(X_test)
labels = [np.argmax(i) for i in predicted]
cm = tf.math.confusion_matrix(labels=y_test, predictions=labels)

sn.heatmap(cm, linewidth=0.5)
plt.show()