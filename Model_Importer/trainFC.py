8#codigo que crea un modelo de prueba que usa capas fully connected y lo guarda en 
#la carpeta models

import tensorflow as tf
from tensorflow import keras
import os


# Se prepara el dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Nombre del modelo
model_name="tryned.h5"

# Se define el modelo
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(200, activation='linear'),
  tf.keras.layers.Dense(80, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
])

# Entrenamos y guardamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.save("Import_engine/Models/"+model_name)

