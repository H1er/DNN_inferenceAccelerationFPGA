#codigo usado para entrenar un modelo de prueba que usa capas convolucionales y fully connected
#entrenado con el dataset MINST y lo guarda en la carpeta Import_engine/Models


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Nombre del modelo
model_name="convmnist.h5"

# Se prepara el dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


print(train_labels)

# Se define el modelo
model = models.Sequential()
model.add(layers.Conv2D(2, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(3, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))

# Entrenamos y guardamos el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

model.save("Import_engine/Models/"+model_name)
