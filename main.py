"""
Algoritmo Jaya
@Author: Jesus D. Redondo
"""
import numpy as np
from funciones import Optimization_func 
import time
from prettytable import PrettyTable

class JayaAlgorithm(object):
    # Paso N° 1 Inicializar párametros constantes
    
    def __init__(self):
        self.time_inicial =time.time()
        self.size =100
        self.iter =2000
        self.dim =2 #
        self.lim_inf = -1
        self.lim_super = 1

    def boundary_check(self,value):

        for i in range(len(value)):
            if value[i] > self.lim_super:
                value[i] = self.lim_super
            elif value[i] < self.lim_inf:
                value[i] = self.lim_inf
        return value

    def evaluar(self, matriz):
        #Metodo para llamar a la funcion con la cual queremos trabajar
        funcion = Optimization_func(matriz,self.dim)
        valores = funcion.diferents_powe_function()
        return valores
    

    def matriz_generate(self):
        #Generar poblacion inicial
        poblation_initial = np.asarray([ self.lim_inf + ( self.lim_super -  self.lim_inf)
        *np.random.random(self.dim) for _ in range(self.size)])
        return poblation_initial
    

    def main(self):
        poblacion_inicial = self.matriz_generate()
        self.matriz_solve = self.evaluar(poblacion_inicial)
        print(self.matriz_solve)
        self.score = self.matriz_solve[:,-1]
        print(self.score)
        value_min = min(self.score)
        self.vector_min = poblacion_inicial[np.argmin(self.score)]
        iter = 0
        for t in range(self.iter):
            new_value_min = min(self.score)
            mew_vector_min = poblacion_inicial[np.argmin(self.score)]
            worst_pos = poblacion_inicial[np.argmax(self.score)]
            if new_value_min < value_min:
                value_min=new_value_min
                self.vector_min = mew_vector_min
                iter = t+1
            new_poblation = poblacion_inicial
            new_score = []
            for i in range(self.size):
                for j in range(self.dim):
                    new_poblation[i][j] += np.random.random() * (mew_vector_min[j] - abs(poblacion_inicial[i][j])) - np.random.random() * (
                            worst_pos[j] - abs(poblacion_inicial[i][j]))
                
              
                new_poblation[i] = self.boundary_check(new_poblation[i])
                new_score.append(self.evaluar(np.array([new_poblation[i]]))[:,-1])                
            for i in range(self.size):
                if new_score[i] < self.score[i]:
                    self.score[i] = new_score[i]
                    poblacion_inicial[i] = new_poblation[i]
        print(new_poblation)
        #Haciendo tabla para mostrar resultados
        tabla = PrettyTable()
        tabla.field_names = ["Función", "Tiempo", "Dimensión", "Límites", "Solución", "Convergencia","iteraciones"]

        # Agregar filas con datos
        fila1 = ["diferents_powe_function", time.time()-self.time_inicial, self.dim, 
                f"[{self.lim_inf},{self.lim_super} ]", min(self.score), iter, self.iter]

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
        print(poblacion_inicial)

if __name__ == '__main__': 
    
    jaya = JayaAlgorithm()
    jaya.main()
    

# Crear una instancia de PrettyTable con las columnas