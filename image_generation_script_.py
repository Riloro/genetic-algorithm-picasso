#GENERANDO IMAGENES CON CIRCULOS
#----------------------------------------
#El siguiente código realiza un algoritmo genético
#con el proposito de reproducir imagenes generadas
#con N circulos
# Ricardo López R.
###################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda, njit, vectorize
from timeit import default_timer as timer
import plotly.express as px
import plotly.io as pio
import os

#Template de las graficas
pio.templates.default = "plotly_white"

#Funcion que convierte un numero binario a decimal 
@njit
def binary_to_dec(binary):
    decimal_2 = 0
    k = len(binary) - 1
    for d in binary:
        decimal_2 += d * 2**(k)
        k -= 1
    return decimal_2

#Transformacion de Genotipo a Fenotipo para cada individuo ...
@njit
def genotype_phenotype_njit(agent, N_poli_):

    #Creamos una matriz donde estaran guadados los atributos
    #de cada poligono --> Dimensiones: N_poligonos X N_atributo
    poly_attributes = np.zeros((N_poli_, 5))

    chromosome = agent
    #Extrayendo los atributos del cromosoma ..
    #Acomodando los poligonos en una matriz de Numero de poligonos x 33
    polygons_of_ch = np.reshape(chromosome, (N_poli_, 33))  # .astype("str")

    #Extrayendo las caranteristicas para cada poligono

    #Tenemos 5 atributos que corresponden a cada columna
    # x_coor -> 0, y_coor -> 1, radio -> 2 , Intensidad -> 3, Transparencia ->4 (Atributo -> #Columna)
    count = 0
    for polygon in polygons_of_ch:
        #x_coor binary to decimal
        end_index_0 = 7
        initial_index = 0
        x_coor = binary_to_dec(polygon[initial_index:end_index_0])
        #y_coor
        initial_index += 7
        end_index = end_index_0*2
        y_coor = binary_to_dec(polygon[initial_index: end_index])
        #Radio
        initial_index += 7
        end_index = end_index_0*3
        r = binary_to_dec(polygon[initial_index:end_index])
        #Intensidad (Color)
        initial_index += 7
        end_index = end_index_0*4 + 1
        color = binary_to_dec(polygon[initial_index:end_index])
        #Transparencia(inidice de la "base de datos")
        initial_index += 8
        alpha_index = binary_to_dec(polygon[initial_index:])
        #Guardanto los atribuos
        poly_attributes[count, 0] = x_coor
        poly_attributes[count, 1] = y_coor
        poly_attributes[count, 2] = r
        poly_attributes[count, 3] = color
        poly_attributes[count, 4] = alpha_index
        #Nueva iteracion
        count += 1
    return poly_attributes

#Transformacion de genotipo a fenotipo para todos los individuos
def gene_phenotype_for_all(population, N_poli_, total_ind):
    k_person = 0
    #Creamos una matriz donde estaran guadados los atributos
    #de cada poligono --> Dimensiones: N_poligonos X N_atributos
    polygon_attributes = np.zeros((N_poli_, 5, total_ind))

    for person in population:
        polygon_attributes[:, :, k_person] = genotype_phenotype_njit(person, N_poli_)
        k_person += 1
    return polygon_attributes

#Construyendo los fenotipos ... (dibujando la imagen)
# x_coor -> 0, y_coor -> 1, radio -> 2 , Intensidad -> 3, Transparencia ->4
# (Atributo -> #Columna)
def painting(total_indivi_, phenotypes_):
    directory = 'C:/Users/user/Documents/MATLAB/Fiscomp2/genetic_algorithms/images'
    os.chdir(directory)
    alpha = np.linspace(0, 1, 16)
    generation = 0
    heightpx = 128
    widthpx = 128
    population_images = np.zeros((heightpx, widthpx, total_indivi_))

    for i in range(total_indivi_):
        image = 255 * np.ones(shape=[heightpx, widthpx, 3], dtype=np.uint8)
        for poligono in phenotypes_[:, :, i]:
            overlay = image.copy()
            intensity = int(poligono[3])
            cv2.circle(image, center=(poligono[0], poligono[1]), radius=poligono[2], color=(
                intensity, intensity, intensity), thickness=-1)
            #Agregando el coeficiente alpha de transparencia ....
            a = alpha[poligono[4]]  # Transparency factor.
            # Following line overlays transparent rectangle over the image
            image_new = cv2.addWeighted(overlay, a, image, 1 - a, 0)
            image = image_new

        #file_name = "indiviuo_" + str(i) + "_gene_" + str(generation) + ".jpg"
        #cv2.imwrite(file_name,image) #Save images
        population_images[:, :, i] = image[:, :, 0]
        # plt.figure(i+1)
        # plt.imshow(image)
        # plt.title("Individuo:" + str(i))
        # plt.show()
    return population_images

#Fitness function
@njit
def fitness_function(target_image, indiv_images):
    l, w = target_image.shape
    h, w_2, p = indiv_images.shape
    target_chromosome = np.ascontiguousarray(target_image).reshape((1, l*w))
    #Vector que guardara el fitness de cada imagen
    fitness = np.zeros(p)
    diff = np.zeros(p)
    #Iterando sobre cada individuo ...
    for i in range(p):
        indiv_chromosome = np.ascontiguousarray(
            indiv_images[:, :, i]).reshape((1, l*w))
        #Caculando el fitness para el individuo i ...
        diff[i] = np.sum(np.abs(target_chromosome - indiv_chromosome))
        fitness[i] = 1/(1 + diff[i])
    return fitness, diff

#Seleccion de los mejores individuos por 
#el metodo de Ruleta 
@njit
def roulette(fitness_vector, padres_N):
    #Vector que guardara los indices de los
    #individuos seleccionados
    slected_ind_index = np.zeros(padres_N)
    #Ordenando el fitness ...
    fitness_sorted = np.sort(fitness_vector)
    #Selección natural por el método de ruleta ...
    normalized_fitness = fitness_sorted/np.sum(fitness_sorted)
    #print(normalized_fitness)
    # Suma comulativa de los fitness normalizados
    cum_fitness = np.cumsum(normalized_fitness)
    #print('\n',cum_fitness)
    #Numero de veces que se jugara la ruleta ...(Depende del número de padres a seleccionar)
    for i in range(padres_N):
        #print("JUEGO", i)
        random_selector = np.random.rand()  # Numero aleatorio entre [0,1)
        index_selected = 0
        for cum_proba in cum_fitness:
            #print("CONTADOR -> ",contador)
            if random_selector < cum_proba:
                break
            else:
                index_selected += 1
        #Fitness del individuo seleccionado
        sorted_fitness_value = fitness_sorted[index_selected]
        # print("INDEX SELECTED", index_selected)
        # print(cum_fitness)
        #Buscando el indice del individuo seleccionado ...
        slected_ind_index[i] = np.nonzero(fitness_vector == sorted_fitness_value)[0][0]  
        # np.nonzero regresa una tupla
    return slected_ind_index

#Single-Point Crossover ...
@njit
def single_point_crossover(parents_chromosomes):
    length_parent, width_p = parents_chromosomes.shape
    offsprings = np.zeros((length_parent, width_p))
    padre_cont = 0
    mother_cont = 1
    #Realizando el cruce ... de las length_parent/2 parejas
    for i in range(int(length_parent/2)):
        #Determinando el punto aleatorio de cruce entre 0 y width_p - 1
        single_point = np.random.randint(width_p)
        #El padre i se reproduce con el padre i + 1 (100% de probabilidad de cruce Pc = 1)
        padre = parents_chromosomes[padre_cont, :]
        madre = parents_chromosomes[mother_cont, :]
        offsprings[padre_cont, :] = padre.copy()
        offsprings[mother_cont, :] = madre.copy()
        #Intercambiando genes ...
        offsprings[padre_cont, single_point:] = madre[single_point:]
        offsprings[mother_cont, single_point:] = padre[single_point:]
        #Actualizando contadores
        padre_cont += 2
        mother_cont += 2
    return offsprings

#Aplicando el procedo de mutacion a cada uno de los hijos ...
@njit
def mutation(arraynp, p_m):
    kids = arraynp.copy()
    l_m, w_m = kids.shape
    for i in range(l_m):
        for j in range(w_m):
            random_num_mutation = np.random.rand()
            #Bit flip si la siguiente condicion se cuple:
            if random_num_mutation <= p_m:
                if kids[i, j] == 1:
                    kids[i, j] = 0
                else:
                    kids[i, j] = 1
    return kids

#Guardando los genes de los padres seleccionados
@njit
def guardado_de_padres(population_,selected_parents_indices_, num_padres):
    p_count = 0
    l_2, w_2 = population_.shape
    parents_genes_ = np.zeros((int(num_padres),w_2))
    for index in selected_parents_indices_:
        parents_genes_[p_count,:] = population_[index,:]
        p_count += 1
    return parents_genes_

#COMIENZO DEL ALGORITMO GENETICO
#------------------------------------------------------------
#Cargando la imagen original...
rect_image = cv2.imread('C:/Users/user/Documents/MATLAB/Fiscomp2/genetic_algorithms/final_rect.png')
plt.figure(1)
plt.imshow(rect_image)
plt.title("Figura original")
plt.show()

#CONDICIONES INICIALES DEL ALGORITMO
#CREANDO LA POBLACION INICIAL...
#Definiendo el numero de poligonos N ...
N_poli = 60
#Eligir aleatoriamente la posicion, tamaño e intensidad para cada poligono, que
#caracterizan a los M individuos(imagenes)
total_indivi = 28
#Creando un un vector de 0s y 1s de manera aleatoria,
#con una longitud igual a total_individuos*33
total_bits = 33
generations = 1000
p_mutation = 0.5/100  # Probabilidad de mutacion
population = np.random.randint(2, size=(total_indivi, total_bits*N_poli))  # array de 0s y 1s
#Numero de veces que
#se jugara la ruleta de seleccion
num_padres = total_indivi/2
parents_images = np.zeros((128, 128, int(num_padres)))

#Transformacion de genotipo a fenotipo
phenotypes = gene_phenotype_for_all(population,N_poli,total_indivi).astype("int64")
#Construyendo los individuos a partir de los fenotipos
images_pobl = painting(total_indivi, phenotypes)
# plt.figure(2)
# plt.imshow(images_pobl[:, :, 27])
# plt.show()

#Calculando el fitness de cada individuo de la poblacion ...
fitness_vector, diff = fitness_function(rect_image[:, :, 0], images_pobl)
#sleccion de los padres por el metodo de ruleta ...
selected_parents_indices = roulette(fitness_vector,int(num_padres)).astype(int)

#Guardado los genes de los padres seleccionados ...
parents_genes = guardado_de_padres(population, selected_parents_indices, num_padres)
#CROSSOVER
children = single_point_crossover(parents_genes)
#MUTATION ...
mutated_children = mutation(children, p_mutation)
#New Population
new_population = np.concatenate((mutated_children, parents_genes))
