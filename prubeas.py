import simpy
import random
import matplotlib.pyplot as plt

# Parámetros de la simulación
tiempo_simulacion = 30  # días
tiempo_atencion_ninos = [2, 6]  # minutos
tiempo_atencion_adultos = [6, 14]  # minutos
tasa_llegada_ninos = [(8, 12, 0), (12, 14, 20), (14, 16, 20), (16, 18, 50), (18, 20, 10)]
tasa_llegada_adultos = [(8, 12, 10), (12, 14, 10), (14, 16, 5), (16, 18, 5), (18, 20, 50), (18, 20, 20)]

# Variables de desempeño
clientes_en_fila = []

def llegada_cliente(env, tipo_cliente):
    global clientes_en_fila
    if tipo_cliente == 'niño':
        inicio, fin, porcentaje = random.choice(tasa_llegada_ninos)
        intervalo_llegada = random.uniform(inicio, fin) * (porcentaje / 100)
        tiempo_atencion = random.uniform(tiempo_atencion_ninos[0], tiempo_atencion_ninos[1])
    else:
        inicio, fin, porcentaje = random.choice(tasa_llegada_adultos)
        intervalo_llegada = random.uniform(inicio, fin) * (porcentaje / 100)
        tiempo_atencion = random.uniform(tiempo_atencion_adultos[0], tiempo_atencion_adultos[1])

    yield env.timeout(intervalo_llegada)
    with empleado.request() as req:
        yield req
        llegada = env.now
        clientes_en_fila.append(len(empleado.queue))
        yield env.timeout(tiempo_atencion)
        clientes_en_fila.append(len(empleado.queue))
        salida = env.now
        print(f"{tipo_cliente.capitalize()} atendido: Tiempo de llegada {llegada}, Tiempo de salida {salida}")

# Configuración de la simulación
env = simpy.Environment()
empleado = simpy.Resource(env, capacity=1)

# Inicialización de la simulación
env.process(llegada_cliente(env, 'niño'))
env.process(llegada_cliente(env, 'adulto'))

# Ejecución de la simulación
env.run(until=tiempo_simulacion)

# Gráfico dinámico de la cantidad de clientes en la fila
plt.plot(clientes_en_fila)
plt.title('Clientes en la Fila a través del Tiempo')
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Clientes en la Fila')
plt.show()

# Cálculos de desempeño
utilizacion_empleado = empleado.time_in_service / env.now
tiempo_promedio_permanencia = sum(clientes_en_fila) / len(clientes_en_fila)

# Resultados
print(f"a) Utilización del empleado: {utilizacion_empleado:.2%}")
print(f"b) Tiempo promedio de permanencia en la fila: {tiempo_promedio_permanencia:.2f} minutos")
