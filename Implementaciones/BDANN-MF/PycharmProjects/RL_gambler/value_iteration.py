import numpy as np
import matplotlib.pyplot as plt


class Environment:
    r0 = 0 # Reward when reaching 0$
    r100 = 1 # Reward when reaching 100$
    ph = 0.4 # Probability of the coin coming up heads
    gamma = 1

    def __init__(self, goal):
        self.goal = goal
        self.numS = (self.goal-1)+2 # Number of possible states (gambler's capital); +2 due to the terminal states (0$ and 100$)
        self.states = 1 + np.arange(self.numS-2) # Possible states s={1,2,...,99}
        self.numA = 1 + np.max(np.minimum(self.states,self.goal-self.states)) # Maximum number of possible actions a={0,1,...,min(s,100-s}; +1 due to action a=0$


def get_dynamics(play):
    P = np.zeros((play.numS, play.numA, play.numS))
    for s in range(play.numS):
        for a in range(np.minimum(s,play.goal-s)+1):
            P[s,a,s+a] = play.ph
            P[s,a,s-a] += 1-play.ph
            # Con el += conseguimos que si una misma acción produce una transición al mismo estado
            # tanto si ganas como si pierdes, se acumulen las probabilidades de transición (caso a=0$).
    return P


def reward_initialization(play):
    R = np.zeros((play.numS, play.numA, play.numS))
    for s in play.states:
        for a in range(np.minimum(s,play.goal-s)+1):
            if s+a == play.goal:
                R[s,a,s+a] = 1
    return R


def state_value_initializacion(num_s):
    V = np.zeros((num_s, 1)) # inicializamos a cero los valores de la state value function
    return V


def maxV(play, p, r, v, s):
    v_aux = np.zeros((play.numA,1))
    possible_actions = np.arange(np.minimum(s,play.goal-s)+1)
    for a in range(play.numA):
        v_aux[a] = np.dot(p[s,a,:],r[s,a,:]) + np.dot(p[s,a,:],play.gamma*v)
    v_final = max(v_aux[possible_actions])
    return v_final;


def argAmaxV(play, p, r, v, order = 3):
    v_aux = np.zeros((play.numA, 1)) # vector de v's auxiliar para cada acción
    actmax_final = np.zeros((play.numS, 1)) # vector en el que se devuelve la policy
    for s in range(play.numS):
        possible_actions = np.arange(np.minimum(s,play.goal-s)+1)
        for a in range(play.numA):
            v_aux[a] = np.dot(p[s,a,:],r[s,a,:]) + np.dot(p[s,a,:],play.gamma*v)
        # obtenemos la acción con la cual se maximiza v de entre el set de acciones posibles
        v_aux_possible = v_aux[possible_actions]
        m = max(v_aux_possible) # conocemos el valor v máximo
        actmax_aux = [i for i, j in enumerate(v_aux_possible) if j == m] # vector con las acciones que empatan
        actmax_final[s] = actmax_aux[order % np.size(actmax_aux,0)] # en caso de empate, elegir la acción número "order"
    return actmax_final;


goal = 100 # Dollars ($) to win
play = Environment(goal)

P = get_dynamics(play) # p(s'|s,a);
R = reward_initialization(play) # r(s,a,s');
V = state_value_initializacion(play.numS) # v(s)

theta = 1e-5
while True:
    inc=0
    for s in range(play.numS):
        v = V[s,0] # slicing necesario (en lugar de poner V[s], para no copiar la referencia y que al cambiar V[s] no se cambie también v)
        V[s] = maxV(play, P, R, V, s)
        inc = np.maximum(inc,np.abs(v-V[s]))
    if inc < theta:
        break

policy = argAmaxV(play, P, R, V)

plt.figure(1)
plt.stem(np.arange(play.numS),policy)
plt.xlabel("Capital"), plt.ylabel("Final policy (stake)"), plt.xticks(np.linspace(0,goal,5))

plt.figure(2)
plt.plot(np.arange(play.numS),V)
plt.xlabel("Capital"), plt.ylabel("Value estimates"), plt.xticks(np.linspace(0,goal,5)), plt.grid()
plt.show()
