#Code for generating a gif from the set of images
# generated by a Genetic algorithm

import glob
import os
import imageio
import plotly.express as px
import numpy as np

#Extrayendo los valores de la funcion fitness del label de
#las imagenes
def plot_fitness(sorted_images):
    generations = range(len(sorted_images))
    fitness = np.zeros(len(sorted_images))
    for count, image_name in enumerate(sorted_images):
        num = float(image_name.split()[-1].split("_")[0])
        fitness[count] = num
    #PLOT ...
    fig_1 = px.line(x=generations, y=fitness,
                    title="Función Fitness a lo largo de las generaciones")
    fig_1.update_xaxes(title="Generación")
    fig_1.update_yaxes(title="Fitness")
    fig_1.update_traces(line=dict(width=3.3))
    fig_1.show()


plotFitness = True
generateGif = False
directory = "C:/Users/user/Documents/MATLAB/Fiscomp2/genetic_algorithms/feynman_experiment/"
file_list = glob.glob(directory + "*.jpg")  # Get all the pngs in the current directory
# Sort the images by generation (FORMAT : very_large_str_genration.jpg)
file_list.sort( key=lambda s: int(s.split("_")[-1].split(".")[0]))

if plotFitness == True:
    print("Creando la grafica ...")
    plot_fitness(file_list)

# #GIF
if generateGif == True:
    print("Generating GIF ... ", "\n")
    gif_path = "C:/Users/user/Documents/MATLAB/Fiscomp2/genetic_algorithms/gifs/"
    gif_name = "Feynman_final_2.gif"
    with imageio.get_writer(gif_path + gif_name, mode='I') as writer:
        for count,filename in enumerate(file_list):
            if count < 300:
                writer.append_data(imageio.imread(filename))
            elif count % 1464 == 0:
                writer.append_data(imageio.imread(filename))
                print(count)
    print("GIF Created :) ", "\n")
