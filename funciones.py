'''
Codigo donde tendremos todas las funciones que 
evaluaremos posteriormente en el algoritmo de busqueda
armonica.
'''

import numpy as np

class Optimization_func():
    def __init__(self,*args):
        self.matriz = args[0]
        self.dim =  args[1]
        self.fx = 0
    ''' ------------ Funciones de N dimensiones. ------------ '''

    def sphere_function(self):
       #La función suele evaluarse sobre el hipercubo xi ∈ [-5.12, 5.12], para todo i = 1, …, d.
        for i in range(self.dim):
            fx_sum= self.matriz[:,i]**2
            self.fx = self.fx + fx_sum
        
       
        return self.retornar_matrix()
        
    def styblinski_function(self):
        # La función suele evaluarse sobre el hipercubo xi ∈ [-5, 5], para todo i = 1, …, d.
        # ¡IMPORTANTE! El resultado sera igual a 39.165*self.dim 
        sum = 0
        for i in range(self.dim):
            new = self.matriz[:,i]**4 - 16*self.matriz[:,i]**2 + 5*self.matriz[:,i]
            sum = sum + new
        self.fx = sum/2
        return self.retornar_matrix()
    
    def michalewicz_function(self):
        #Esta funcion no da resultados tan precisos
        # La función suele evaluarse sobre el hipercubo xi ∈ [0, π], para todo i = 1, …, d.
        m = 10
        sum = 0
        for i in range(1,self.dim+1):
            new = np.sin(self.matriz[:,i-1]) * (np.sin(i*self.matriz[:,i-1]**2/np.pi))**(2*m)
            sum  = sum + new
        self.fx = -sum
        return self.retornar_matrix()

    def schwefel_function(self):
        #La función suele evaluarse sobre el hipercubo xi ∈ [-500, 500], para todo i = 1, …, d.
        for i in range(self.dim):
            fx_sum = self.matriz[:,i]*np.sin(np.sqrt(abs(self.matriz[:,i])))
            self.fx = fx_sum + self.fx
        self.fx = 418.9829*self.dim - self.fx
        
        return self.retornar_matrix()
    
    def rosenbrock_function(self):
        #La función suele evaluarse sobre el hipercubo xi ∈ [-5, 10], para todo i = 1, …, d
        for i in range(self.dim-1):
            fx_sum = 100.0*(self.matriz[:,i+1] -self.matriz[:,i]**2.0)**2.0 + (self.matriz[:,i]-1)**2.0
            self.fx = fx_sum + self.fx
            
        return self.retornar_matrix()
    
    def diferents_powe_function(self):
       # La función suele evaluarse sobre el hipercubo x i ∈ [-1, 1], para todo i = 1, …, d.
        sum = 0
        for i in range(1,self.dim+1):    
            new = (np.abs(self.matriz[:,i-1]))**(i+1)
            sum = sum + new
        self.fx = sum

        return self.retornar_matrix()

    def sum_squares_function(self):
        # La función suele evaluarse sobre el hipercubo x i ∈ [-10, 10], para todo i = 1, …, d
        sum = 0
        for i in range(self.dim+1):
            sum = sum + i*self.matriz[:,i-1]**2;
        self.fx = sum
        return self.retornar_matrix()

    def ackley_function(self):
        #La función suele evaluarse sobre el hipercubo xi ∈ [-32.768, 32.768], para todo i = 1, …, d, 
        #aunque también puede estar restringida a un dominio más pequeño.
        
        #Constantes:
        a = 20
        b = 0.2
        c = 2*np.pi
        sum1 = 0
        sum2 = 0
        
        for i in range(self.dim):
            sum1 = sum1 + self.matriz[:,i]**2.0
            sum2 = sum2 + np.cos(c*self.matriz[:,i])
            
        term1 = -a * np.exp(-b*np.sqrt(sum1/self.dim))
        term2 = -np.exp(sum2/self.dim)
        self.fx = term1 + term2 + a + np.exp(1)
        
        return self.retornar_matrix()
 
    def griewank_function(self):
        #La función suele evaluarse sobre el hipercubo xi ∈ [-600, 600], para todo i = 1, …, d.
        sum = 0
        prod = 1

        for i in range(self.dim):
            
            sum = sum + self.matriz[:,i]**2/4000
            prod = prod * np.cos(self.matriz[:,i]/np.sqrt(i+1));
        self.fx = sum - prod + 1
        return self.retornar_matrix()
 
    def levy_function(self):
        pass
    def rastrigin_function(self):
        # La función suele evaluarse sobre el hipercubo xi ∈ [-5.12, 5.12], para todo i = 1, …, d.

        sum = 0
        for i in range(self.dim):
            sum = sum + (self.matriz[:,i]**2.0 - 10*np.cos(2*np.pi*self.matriz[:,i]))
        self.fx = 10*self.dim + sum
        
        return self.retornar_matrix()
    
    
    '''' ------------ Comienzo de funciones 2D ------------'''
    
    def goldstein_function(self):
        # La función suele evaluarse en el cuadrado xi ∈ [-2, 2], para todo i = 1, 2.
        # Hay algun error corregir o prguntarle al profe
        if self.dim == 2: 
            x1bar = 4*self.matriz[:,0] - 2
            x2bar = 4*self.matriz[:,1] - 2

            fact1a = (x1bar + x2bar + 1)**2
            fact1b = 19 - 14*x1bar + 3*x1bar**2 - 14*x2bar + 6*x1bar*x2bar + 3*x2bar**2
            fact1 = 1 + fact1a*fact1b

            fact2a = (2*x1bar - 3*x2bar)**2
            fact2b = 18 - 32*x1bar + 12*x1bar**2 + 48*x2bar - 36*x1bar*x2bar + 27*x2bar**2
            fact2 = 30 + fact2a*fact2b

            prod = fact1*fact2

            self.fx = (np.log(prod) - 8.693) / 2.427
            return self.retornar_matrix()
        else:
            print("Esta funcion solo existe para 2D")
    
    def shubert_function(self):
        # La función suele evaluarse sobre el cuadrado x i ∈ [-10, 10], para todo i = 1, 2,
        if self.dim == 2: 
            sum1 = 0
            sum2 = 0
            for i in range(1,6): #Preguntar si en matlab los bucles for terminan en -1
                new1 = i * np.cos((i+1)*self.matriz[:,0]+i)
                new2 = i * np.cos((i+1)*self.matriz[:,1]+i)
                sum1 = sum1 + new1
                sum2 = sum2 + new2
            self.fx = sum1 * sum2
            return self.retornar_matrix()
        else:
            print("Esta funcion solo existe para 2D")
    
    def schaffer4_function(self):
        # La función suele evaluarse en el cuadrado x i ∈ [-100, 100], para todo i = 1, 2.

        if self.dim == 2: 
            fact1 = (np.cos(np.sin(np.abs(self.matriz[:,0]**2-self.matriz[:,1]**2))))**2 - 0.5
            fact2 = (1 + 0.001*(self.matriz[:,0]**2+self.matriz[:,1]**2))**2
            self.fx = 0.5 + fact1/fact2
            return self.retornar_matrix()
        else:
            print("Esta funcion solo existe para 2D")
    
    def cross_in_tray_function(self):
        # La función suele evaluarse sobre el cuadrado x i ∈ [-10, 10], para todo i = 1, 2.
        if self.dim == 2: 
            fact1 = np.sin(self.matriz[:,0])*np.sin(self.matriz[:,1])
            fact2 = np.exp(np.abs(100 - np.sqrt(self.matriz[:,0]**2+self.matriz[:,1]**2)/np.pi))
            self.fx = -0.0001 * (np.abs(fact1*fact2)+1)**0.1
            
            return self.retornar_matrix()
        else:
            print("Esta funcion solo existe para 2D")
    
    def drop_wave_function(self):
        # La función suele evaluarse sobre el cuadrado x i ∈ [-5,12, 5,12], para todo i = 1, 2.
        if self.dim == 2:
            frac1 = 1 + np.cos(12*np.sqrt(self.matriz[:,0]**2+self.matriz[:,1]**2))
            frac2 = 0.5*(self.matriz[:,0]**2+self.matriz[:,1]**2) + 2
            self.fx = -frac1/frac2
            
            return self.retornar_matrix()
        
        else:
            print("Esta funcion solo existe para 2D")   
            
    def easom_function(self):
        if self.dim == 2:
            fact1 = -np.cos(self.matriz[:,0])*np.cos(self.matriz[:,1])
            fact2 = np.exp(-(self.matriz[:,0]-np.pi)**2-(self.matriz[:,1]-np.pi)**2)
            self.fx = fact1*fact2
            
            return self.retornar_matrix()
        
        else:
            print("Esta funcion solo existe para 2D")   
            
    ''' ------------ Aqui terminan las funciones 2D. ------------ '''
    ''' ------------ Problemas de optimización especiales ------------ '''
    
    
    
    
    
    
    
    def retornar_matrix(self):
        #Con este metodo evitamos escribir las dos lineas contenidas aqui de manera repetida en cada función
        matriz_complete= np.concatenate((self.matriz, np.array([self.fx]).T), axis = 1)
        return matriz_complete