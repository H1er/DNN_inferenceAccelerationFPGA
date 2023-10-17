import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

from tensorflow.keras import datasets, layers, models

# Archivo con el que se pueden generar los archivos de prueba
def createmnistData(nimages):
    f = open("testMnistData.txt", "w")
    for i in range(0,nimages):
        flattened = test_images[i].flatten()
        for j in range(0,flattened.size):
            f.write(str(str(flattened[j])+", "))
        f.write("\n")

    f.close()
    print(str(nimages)+" inputs added to file testMnistData.txt")


def saveModelResults(output_array,nimages):
    fils=nimages
    cols = int(output_array[0].size)

    f = open("TensorflowModelResults.txt", "w")
    for i in range(0,fils):
        l=""
        for j in range(0,cols):
            #print(str(i)+" "+str(j))
            l+=str(output_array[i][j])+", "
        f.write(l+" \n")
    print(str(nimages)+" outputs saved to file TensorflowModelResults.txt")

# Nombre del modelo a cargar
model_name="tryned.h5"

# Se carga el modelo 
loaded_model = tf.keras.models.load_model('Import_engine/Models/'+model_name)

# El feature expractor se encarga de obtener las salidas de la capa indicada
feature_extractor = tf.keras.Model(
    inputs=loaded_model.inputs,
    outputs=[
             loaded_model.output,  
             loaded_model.layers[4].output # < output layer
    ]
)


# Cargamos el dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


# Extraemos los resultados del modelo (y input, conv_y resultado de pasar y por la capa convolucional)
y, conv_y = feature_extractor(test_images)


# Creamos el fichero con los datos de entrada que se ejecutaran en el inference
nimages=5000
createmnistData(nimages)


image_index=0

# Descomentar si se quiere plotear la imagen de entrada image_index 

#image = test_images[image_index]
#fig = plt.figure
#plt.imshow(image, cmap='gray')
#plt.show()


# Obtenemos el objeto ndarray de conv_y
output_array=conv_y.numpy()

print("\n")


#se muestra por pantalla el resultado de la capa convolucional para cada input

line=""

# Pinta resultados de las convolucionales
#for l in range(0,25):
#    print("output "+str(l))
#    for i in range(0,nimages):
#        for j in range(0,22):
#            for k in range(0,22):
#                line= line+str(output_array[i][j][k][l])+" "
#    print(line+"\n\n")
#    line=""



# Pinta resultados de la flatten
#for i in range(0,12100):
#    line= line+str(output_array[0][i])+" "

for i in range (0,10):
    line+=str(output_array[9999][i])+" "



 
saveModelResults(output_array,nimages)



print(line)
line=""



