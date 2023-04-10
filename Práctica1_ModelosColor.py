 # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data,color,io,img_as_ubyte
from sklearn import metrics
import cv2 as cv
import time

def Metodo_KMeans(imagen,clases,modelo):
    #Imprimir imagen original
    plt.figure(0)
    plt.imshow(imagen)
    plt.title('Imagen Original')
    plt.axis('off')
    
    #Separar las capas de la imagen ingresada
    primera_capa=(imagen[:,:,0])
    segunda_capa=(imagen[:,:,1])
    tercera_capa=(imagen[:,:,2])
    
    PR = primera_capa.reshape((-1,1))
    SE = segunda_capa.reshape((-1,1))
    TE = tercera_capa.reshape((-1,1))
    
    datos_imagen = np.concatenate((PR,SE,TE), axis = 1)
    
    #Método de KMeans para imagen ingresada
    inicio = time.time()
    salida_imagen = KMeans(n_clusters = clases)
    salida_imagen.fit(datos_imagen)
    print("Se completo en " + str(time.time() - inicio) + " segundos.")

    centros_imagen= salida_imagen.cluster_centers_
    print("Se realizaron " + str(salida_imagen.n_iter_) + " iteraciones.")
    
    #Obtener el procentaje de Silhouette
    # score = metrics.silhouette_score(datos_imagen, salida_imagen.labels_)
    # print(score)

    if modelo == 2:
        centros_imagen=color.hsv2rgb(centros_imagen[np.newaxis,:])
    elif modelo == 3:
        centros_imagen2 = centros_imagen/np.max(centros_imagen)
        centros_imagen2 = img_as_ubyte(centros_imagen2)
        centros_imagen=cv.cvtColor((centros_imagen2[np.newaxis,:]),cv.COLOR_HLS2BGR)
    elif modelo == 4:
        centros_imagen=color.lab2rgb(centros_imagen[np.newaxis,:])
    else:
       centros_imagen = centros_imagen  
       
    etiquetas_imagen = salida_imagen.labels_

    #Asignar un color a cada posicion según la etiqueta
    if modelo == 2 or modelo == 3 or modelo == 4:
        for i in range(PR.shape[0]):
            PR[i] = centros_imagen[0][etiquetas_imagen[i]][0] 
            SE[i] = centros_imagen[0][etiquetas_imagen[i]][1] 
            TE[i] = centros_imagen[0][etiquetas_imagen[i]][2] 
    else:
        for i in range(PR.shape[0]):
            PR[i] = centros_imagen[etiquetas_imagen[i]][0] 
            SE[i] = centros_imagen[etiquetas_imagen[i]][1] 
            TE[i] = centros_imagen[etiquetas_imagen[i]][2] 

    PR.shape = primera_capa.shape
    SE.shape = segunda_capa.shape
    TE.shape = tercera_capa.shape

    PR = PR[:,:,np.newaxis]
    SE = SE[:,:,np.newaxis]
    TE = TE[:,:,np.newaxis]

    new_imagen = np.concatenate((PR,SE,TE), axis = 2) # axis 2 es por capas
    
    #Imprimir imagen clasificada en formato rgb
    plt.figure(1)
    plt.imshow(new_imagen)
    plt.title('Imagen Clasificada')
    plt.axis('off')
    
################### SELECCION DE IMAGEN ############################
plt.close('all')
image = 'imagen0.jpg'
# image = 'flores.jpg'
# image = 'figuras3.jpg'
opcion = int(input('Opciones de imagenes a clasificar: \n'+
               '1) RGB\n'+'2) HSV\n'+'3) HSL\n'+'4) CIE LAB\n'+
               'Eliga la imagen: '))

if opcion == 1: 
    ima_rgb= io.imread(image)
    num_clases = int(input('Ingrese en cuantas clases quiere separar la imagen: '))
    Metodo_KMeans(ima_rgb, num_clases,opcion)
elif opcion == 2:
    ima_hsv = color.rgb2hsv(io.imread(image))
    num_clases = int(input('Ingrese en cuantas clases quiere separar la imagen: '))
    Metodo_KMeans(ima_hsv, num_clases,opcion)
elif opcion == 3:
    ima_hsl = cv.cvtColor(cv.imread(image),cv.COLOR_BGR2HLS)
    num_clases = int(input('Ingrese en cuantas clases quiere separar la imagen: '))
    Metodo_KMeans(ima_hsl, num_clases,opcion)
elif opcion == 4:
    ima_cielab = color.rgb2lab(io.imread(image))   
    num_clases = int(input('Ingrese en cuantas clases quiere separar la imagen: '))
    Metodo_KMeans(ima_cielab, num_clases,opcion)