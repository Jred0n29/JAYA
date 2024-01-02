
"""
Algoritmo Jaya
@Author: Jesus D. Redondo
"""
import copy
import random
import numpy as np
from funciones import Optimization_func 
import time
from prettytable import PrettyTable

class JayaAlgorithm(object):
    # Paso N° 1 Inicializar párametros constantes
    
    def __init__(self):
        self.time_inicial =time.time()
        self.size = 100
        self.iter = 20000
        self.dim = 10 #
        self.liminferior = -5.12
        self.limisuperior = 5.12
        self.flags = True
    def boundary_check(self,value):

        for i in range(len(value)):
            if value[i] > self.limisuperior:
                value[i] = self.limisuperior
            elif value[i] < self.liminferior:
                value[i] = self.liminferior
        return value

    def evaluar(self, matriz):
        #Metodo para llamar a la funcion con la cual queremos trabajar
        funcion = Optimization_func(matriz,self.dim)
        valores = funcion.rastrigin_function()
        return valores
    
    def main(self):
        numberFocos = 5
        spotlights = np.empty((numberFocos, self.dim))    


        # Step 1. Initialization
        pos = np.empty((self.size, self.dim))
        for i in range(self.size):
            #¿Porque ramdom uniform?
            #Forma de crear una matriz con valores aleatorios entre los limites de dimensiones 20x2
            pos[i, :] = np.array([random.uniform(self.liminferior, self.limisuperior)
                                for _ in range(self.dim)])
        self.matriz_solve = self.evaluar(pos) #matriz completa con tres columnas
        self.score = self.matriz_solve[:,-1] #Vector solucion de la matriz

       
        iter_best = []
        gbest = min(self.score)
        #Conjunto de valores que dan la mejor solución        
        gbest_pos = pos[self.score.tolist().index(gbest)].copy()
        iter_con = 0
        # Step 2. The main loop
        for t in range(self.iter):
    
            # Step 2.1. Identify the best and worst solutions
            best_score = min(self.score)
            best_pos = pos[self.score.tolist().index(best_score)].copy()
            worst_score = max(self.score)
            worst_pos = pos[self.score.tolist().index(worst_score)].copy()
            #print(best_score,"----", gbest) #Pequeño problema y es que por lo menos en las primeras iteraciones no entrará en la condicion porque no hay numero menores que otros solo iguales
            if best_score < gbest:
                gbest = best_score
                gbest_pos = best_pos.copy()
                iter_con = t + 1
            iter_best.append(gbest) #No hace nada esta linea
 
            # Step 2.2. Modify solutions
            new_pos = copy.deepcopy(pos)
            new_score = []
            for i in range(self.size):
                for j in range(self.dim):
                    new_pos[i][j] = new_pos[i][j] + random.random() * (best_pos[j] - abs(pos[i][j])) - random.random() * (
                                worst_pos[j] - abs(pos[i][j]))
                new_pos[i] = self.boundary_check(new_pos[i])  # boundary check
                
                new_score.append(self.evaluar(np.array([new_pos[i]]))[:,-1])

            # Step 2.3. Evaluate new solutions
            for i in range(self.size):
                if new_score[i] < self.score[i]:
                    self.score[i] = new_score[i]
                    pos[i] = new_pos[i].copy()
            

            std = np.std(self.score)
            
            #Comencemos a explorar:
            contar = 0
            if std < 7:
                worst_score_n = max(self.score)
                #Entre mas se acerquen a la lampara atacar mas fuerte igual
                # que las hormigas cuando buscan comida
                if self.flags:
                    number_new_poblation = 100
                    new_poblation = np.empty((number_new_poblation, self.dim))
                    
                    for i in range(number_new_poblation):
                        new_poblation[i, :] = np.array([random.uniform(self.liminferior, self.limisuperior)
                                    for _ in range(self.dim)])
                    
                    self.flags = False
                    
                matriz_new_poblation = self.evaluar(new_poblation) #matriz completa con tres columnas
                score_new_poblation = matriz_new_poblation[:,-1]
                best_score_new_p = min(score_new_poblation)
                best_pos_new_p = new_poblation[score_new_poblation.tolist().index(best_score_new_p)].copy()
                if best_score_new_p < gbest:
                    gbest = best_score_new_p
                    gbest_pos = best_pos_new_p.copy()

                else:
                    vec_aleatorio = best_pos
                    vec_solve = np.array([best_pos_new_p,vec_aleatorio])
                    std = np.std(vec_solve, axis=0)
                    estrategia = np.array([vec_aleatorio - abs(std) * np.sign(vec_aleatorio-best_pos_new_p)])
                
                    estrategia_solve = self.evaluar(estrategia)
                    vec_viejo = self.evaluar(np.array([vec_aleatorio]))
                    
                    #print(estrategia_solve)
                                
                    if vec_viejo[:,-1] > estrategia_solve[:,-1]:
                        print("Mejor que la solucion global")
                        print(best_score,vec_viejo[:,-1] , estrategia_solve[:,-1])
                        contar = contar + 1
                        #self.score[self.score.tolist().index(worst_score)] = 
                        print()
                        pos[self.score.tolist().index(worst_score_n)] = estrategia
                        self.score[self.score.tolist().index(worst_score_n)] = worst_score_n

                        
                    elif estrategia_solve[:,-1] < self.evaluar(np.array([best_pos_new_p]))[:,-1]:
                        #print("Mejor que la solucion a la cual se le aplico std")
                        pos[self.score.tolist().index(worst_score_n)] = estrategia
                        self.score[self.score.tolist().index(worst_score_n)] = worst_score_n
                       
                

        #Haciendo tabla para mostrar resultados
        tabla = PrettyTable()
        tabla.field_names = ["Función", "Tiempo", "Dimensión", "Límites", "Solución", "Convergencia","iteraciones"]

        # Agregar filas con datos
        fila1 = ["rastrigin_function", time.time()-self.time_inicial, self.dim, 
                f"[{self.liminferior},{self.limisuperior} ]", gbest, iter_con, self.iter]

        tabla.add_row(fila1)

        # Personalizar el estilo de la tabla y los colores de los títulos
        tabla.align = "c"  # Centrar el contenido
        tabla.title = "Resultados del algoritmo Jaya"
        tabla.title_style = "bold magenta"  # Color y estilo del título
        tabla.field_names_style = "blue"  # Color de los títulos de las columnas
        tabla.horizontal_char = "-"  # Caracter horizontal de separación

        # Imprimir la tabla
        print(tabla)
        print("Vector Solucion")
        print(self.score)
        print("Desviacion estandar",np.std(self.score))
        #print(gbest_pos)

if __name__ == '__main__':
    
    jaya = JayaAlgorithm()
    jaya.main()
    

# Crear una instancia de PrettyTable con las columnas
