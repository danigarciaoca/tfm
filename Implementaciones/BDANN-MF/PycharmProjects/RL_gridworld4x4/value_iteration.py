# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Nombre del fichero: value_iteration.py                  #
# Autor: Daniel M. García-Ocaña Hernández                 #
# Fecha de creación: 07/10/2016                           #
# Implementación in-place del algoritmo value iteration   #
# (p.88 de Reinforcement Learning: An Introduction)       #
# Ejemplo: gridworld 4x4, p.81 Example 4.1                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def get_dynamics(num_s, num_a, dynamics_sxa):
    # inicializa matriz s x a x s' que contendrá las probabilidades de transición a un estado s',
    # desde un estado s conocido y tomando una acción a conocida.
    P = np.zeros((num_s, num_a, num_s))
    for s in range(num_s):
        for a in range(num_a):
            # establecemos a 1 la probabilidad de pasar al estado dynamics_sxa[s,a] cuando estamos en el
            # estado s y tomamos la acción a
            P[s,a,dynamics_sxa[s,a]] = 1
    return P # retornamos la matriz de transición rellena


def policy_initialization(num_s, num_a):
    # inicializa un vector de policies vacío
    pi = np.zeros((num_s,1));
    return pi # retornamos el vector de policies vacío


def reward_initialization(num_s, num_a, dynamics_sxa):
    # inicializa matriz s x a x s' que contendrá la recompensa obtenida al pasar a un estado s',
    # desde un estado s conocido y tomando una acción a conocida.
    R = np.zeros((num_s, num_a, num_s))
    for s in range(num_s):
        # la recompensa es -1 en todas las transiciones hasta que el estado terminal es alcanzado
        for a in range(num_a):
            # si s != 0 la transición ocurre desde un estado no terminal, y por tanto la recompensa es -1
            if s != 0:
                R[s,a,dynamics_sxa[s,a]] = -1
    return R # retornamos la matriz de recompensas rellena


def state_value_initializacion(num_s):
    # inicializamos a cero los valores de la state-value function
    V = np.zeros((num_s, 1))
    return V # retornamos el vector de state-values


def maxV(gamma, p, r, v, s, num_a):
    v_aux = np.zeros((num_a,1))
    for a in range(num_a):
        # evaluamos la state-value function para todas las posibles acciones
        v_aux[a] = np.dot(p[s,a,:],r[s,a,:]) + np.dot(p[s,a,:],gamma*v)
    # seguimos aquella acción que maximiza la state-value function, lo cual se traduce en
    # quedarnos el máximo valor de los state-values obtenidos (ecuación de optimalidad de
    # Bellman; seguimos una greedy policy respecto al valor de la state-value function)
    v_final = max(v_aux)
    return v_final; # retornamos el state-value máximo


def argAmaxV(gamma, p, r, v, s, num_a, order = 0):
    # función para seleccionar la acción que maximiza la state value-function. El parámetro order
    # permite elegir, en caso de empate, qué acción tomar (es decir, si elegimos la primera,
    # la segunda, la tercera etc... de entre las acciones que empatan).
    v_aux = np.zeros((num_a, 1)) # vector de v's auxiliar para cada acción
    for a in range(num_a):
        # evaluamos la state-value function para cada acción posible
        v_aux[a] = np.dot(p[s,a,:],r[s,a,:]) + np.dot(p[s,a,:],gamma*v)
    # obtenemos la/las acción/acciones con la cual/las cuales se maximiza v:
    m = max(v_aux) # conocemos el valor v máximo
    # en caso de empate de acciones, buscamos todas aquellas que maximizan el state-value:
    actmax_aux = [i for i, j in enumerate(v_aux) if j == m] # vector con las acciones que empatan
    actmax_final = actmax_aux[order % np.size(actmax_aux,0)] # en caso de empate, elegir la acción número |order|
    # con order % np.size(actmax_aux,0) nos aseguramos de que si pedimos elegir la acción |order| y el |número de acciones
    # que empatan| < |order|, python no lance error, sino que coja la siguiente opción posible siguiendo aritmética modular.
    return actmax_final;


# dynamics_SxA define una matriz de transiciones en la que las filas son los estados s0:s14 y las columnas
# las acciones a0:a3 donde a0 = izquierda, a1 = arriba, a2 = derecha y a3 = abajo.
# El valor de cada elemento de de la matriz nos indica el número de estado al que pasaremos al aplicar una
# determinada acción.
# Así por ejemplo, si estamos en el estado s0 (estado terminal), cualquier acción nos llevará
# a él mismo. Por el contrario, si estamos en el estado s2 y aplicamos la acción a3 (movernos abajo), pasaremos
# al estado s6.
dynamics_SxA = np.array([[0, 0, 0, 0],
                        [0, 1, 2, 5],
                        [1, 2, 3, 6],
                        [2, 3, 3, 7],
                        [4, 0, 5, 8],
                        [4, 1, 6, 9],
                        [5, 2, 7, 10],
                        [6, 3, 7, 11],
                        [8, 4, 9, 12],
                        [8, 5, 10, 13],
                        [9, 6, 11, 14],
                        [10, 7, 11, 0],
                        [12, 8, 13, 12],
                        [12, 9, 14, 13],
                        [13, 10, 0, 14],
                        ])

numS = 15 # número de estados posibles (14 + estado terminal = 15)
numA = 4 # número de acciones posibles (izquierda, arriba, derecha, abajo)
gamma = 1 # discount rate

P = get_dynamics(numS, numA, dynamics_SxA) # p(s'|s,a)
policy = policy_initialization(numS, numA) # pi(a|s);
R = reward_initialization(numS, numA, dynamics_SxA) # r(s,a,s')
V = state_value_initializacion(numS) # v(s)

theta = 1e-4 # condición de convergencia
while True: # evaluar lo de dentro del bucle hasta que el algoritmo haya convergido
    inc=0
    for s in range(numS):
        v = deepcopy(V[s,0]) # copia profunda necesaria para no copiar la referencia (la dirección de memoria de V[s]). Con esto conseguimos que al cambiar V[s] no se cambie también v

        V[s] = maxV(gamma, P, R, V, s, numA) # ecuación de optimalidad de Bellman

        # buscamos de entre todos los estados, aquel cuya diferencia entre el state-value anterior y actual
        # es máxima. Si se garantiza la convergencia para este estado, se garantiza para los demás por ser la
        # diferencia máxima.
        inc = np.maximum(inc,np.abs(v-V[s]))
    if inc < theta:
        # cuando el algoritmo ha convergido, paramos el bucle de value iteration
        break

# tras haber convergido el algoritmo, obtenemos la policy que maximiza la state-value function:
for s in range(numS):
    policy[s] = argAmaxV(gamma, P, R, V, s, numA)

print(V) # mostramos por pantalla el valor final de la state-value function
plt.figure(1)
plt.stem(np.arange(numS),policy) # ploteamos la policy óptima encontrada
plt.xlabel("Casilla"), plt.ylabel("Política final (movimiento)")
plt.show()
