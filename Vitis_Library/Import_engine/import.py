# Se encarga de leer el modelo indicado guardado en la carpeta Import_engine/Models, obtener los datos de los pesos, sesgo 
# y demas para cada capa y de escribirlo en el fichero model_data correspondiente que luego sera usado en import_mopdel

import tensorflow as tf
from numpy import array
from tensorflow import keras
import sys
import numpy
import os
import math

model_name = sys.argv[1]
lib_path= sys.argv[2]

models_path = lib_path+"/Import_engine/Models"
#hacer que cada vez que se use algo en alguna ruta se añada abs_path al principio


print(os.getcwd())
print('Models/'+model_name)


loaded_model = tf.keras.models.load_model(models_path+'/'+model_name)

print('Model loaded\n\n')

if(model_name.endswith('.pb')):
    model_name=model_name[0:(len(model_name)-3)]

print(loaded_model.layers[0].input_shape)

# Descomentar si se quiere ver el summary del modelo generado por tf antes de que se importe a la libreria
loaded_model.summary()

cont1=0
cont=0

#variables que servirán para generar las constantes que se usaran en el kernel
weights_f_dims=""
weights_c_dims=""
input_f_dims=""
input_c_dims=""

tiles_input_f=""
tiles_input_c=""
tiles_weights_f=""
tiles_weights_c=""

parent=models_path

data_path=os.path.join(parent, "Model_data")

#print("\nIs dir "+data_path+"\n")
if(not os.path.isdir(data_path)):
    os.mkdir(data_path,0o777)
                                                        #falla el pb porque no se crean los directorios
                                                        
model_path = os.path.join(data_path, model_name)

#print("\nIs dir "+model_path+"\n")

print('Model path: '+str(model_path))
if(not os.path.isdir(model_path)):
    #print("\n\nmake dir: "+model_path+"\n")
    os.mkdir(model_path,0o777)


#print(model_path)
with open(model_path+"/model_data.txt", 'w+') as f:
    f.write("Model_Size.%d\n" % len(loaded_model.layers))
    #print("Model_Size.%d\n" % len(loaded_model.layers))

    for layer in loaded_model.layers:
        f.write("Layer.%d\n" % cont1)

        info = layer.get_config()
        layer_type=info.get("name")
        h=layer.get_weights()

        f.write("Type.%s\n" % layer_type)
        f.write("Activation.%s" % info.get("activation"))


        f.write("\n")


        if ("dense" in layer_type):

            #print("dense layer detected import")
            weight_mat = h[0]
            f.write("weights_dim.%dx%d \n" % (len(weight_mat), len(weight_mat[0]) ))
            #print("weights_dim.%dx%d \n" % (len(weight_mat), len(weight_mat[0]) ))
    
            weights_f_dims+=str(len(weight_mat))+","
            weights_c_dims+=str(len(weight_mat[0]))+","
            input_f_dims+=str(1)+","
            input_c_dims+=str(layer.input_shape[1])+","

            for i in range (len(weight_mat)):
                for j in range(len(weight_mat[i])):
                    f.write(" %f, " % weight_mat[i][j])
                f.write("\n")
            cont=cont+1
            f.write("\n")
            cont=0
            f.write("Bias\n")

            bias=h[1]
            for k in range(len(bias)):
                f.write(" %f, " % bias[k])
            f.write("\n")
        elif ("flatten" in layer_type):
            f.write("weights_dim.0x0")
            #print("flatten layer detected import")

            weights_f_dims+=str(0)+","
            weights_c_dims+=str(0)+","
            input_f_dims+=str(0)+","
            input_c_dims+=str(0)+","            

        elif("conv2d" in layer_type):
            filters = h[0]
            #print("conv layer detected import")
            kernel_h=filters.shape[0]
            kernel_w=filters.shape[1]
            number_of_chanels=filters.shape[2]
            number_of_filters=filters.shape[3]


            input_f = layer.input_shape[1]*layer.input_shape[2]

            weights_f_dims+=str(input_f)+","
            weights_c_dims+=str(1)+","

            convs = ((layer.input_shape[2]-kernel_w)+1)*((layer.input_shape[1]-kernel_h)+1)
            #((input_cols-kernel_w)+1)*((input_fils-kernel_h)+1)
            input_f_dims += str(convs)+","
            input_c_dims += str(input_f)+","

            f.write("weights_dim.%dx%dx%dx%d \n" % (kernel_h, kernel_w, number_of_chanels, number_of_filters))
            f.write("ishape.%dx%d\n" %(layer.input_shape[1],layer.input_shape[2]))

            for k in range(number_of_chanels):
                #f.write("Channel.%d\n"%k)
                for l in range(number_of_filters):
                    #f.write("Filter.%d\n"%l)
                    for i in range(kernel_h):
                        for j in range(kernel_w):
                            f.write(" %f, "%filters[i][j][k][l])

                        f.write("\n")
            f.write("Bias\n")
            for m in range(len(h[1])):
                f.write(" %f, "%h[1][m])
            f.write("\n")
        else:
            print("Error, invalid layer type detected import")


        f.write("\nENDLAYER\n")


        cont1=cont1+1

f.close()
print("Model_data file created at "+str(model_path)+"/model_data.txt\n")

#with open(lib_path+"/kernel_constants.h", 'w+') as f:
#    f.write("#ifndef KERNEL_CONSTANTS_H\n")
#    f.write("#define KERNEL_CONSTANTS_H\n\n")
#    f.write("#include <iostream>\n\n")

#    f.write("const int WEIGHTS_FILS["+str(cont1)+"]={"+weights_f_dims[0:-1]+"};\n")
#    f.write("const int WEIGHTS_COLS["+str(cont1)+"]={"+weights_c_dims[0:-1]+"};\n")
#    f.write("const int INPUT_FILS["+str(cont1)+"]={"+input_f_dims[0:-1]+"};\n")
#    f.write("const int INPUT_COLS["+str(cont1)+"]={"+input_c_dims[0:-1]+"};\n\n")

#    f.write("const int TILES_INPUT_FILS["+str(cont1)+"]={"+tiles_input_f[0:-1]+"};\n")
#    f.write("const int TILES_INPUT_COLS["+str(cont1)+"]={"+tiles_input_c[0:-1]+"};\n")
#    #f.write("const int TILES_WEIGHTS_F["+str(cont1)+"]={"+tiles_weights_f[0:-1]+"};\n")
#    f.write("const int TILES_WEIGHTS_COLS["+str(cont1)+"]={"+tiles_weights_c[0:-1]+"};\n\n")

#    f.write("const int TILE_SIZE="+str(10)+";\n")
#    f.write("const int BUFFER_SIZE="+str(20)+";\n")
#    f.write("const int CACHE_SIZE="+str(5)+";\n")



#    f.write("\n#endif\n")
#    f.close()


#print("kernel_constants.h modified with the model data of "+str(model_name))

#print("tile size from python: "+str(tile_size))
 
