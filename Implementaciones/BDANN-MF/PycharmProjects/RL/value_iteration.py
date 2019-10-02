import numpy as np
import matplotlib.pyplot as plt

# Initialization of constants
goal = 100 # Dollars ($) to win
numS = (goal-1)+2 # Number of possible states (gambler's capital); +2 due to the terminal states (0$ and 100$)
states = 1 + np.arange(numS-2) # Possible states s={1,2,...,99}
numA_max = 1 + np.max(np.minimum(states,goal-states)) # Maximum number of possible actions a={0,1,...,min(s,100-s}; +1 due to action a=0
r0 = 0 # Reward when reaching 0$
r100 = 1 # Reward when reaching 100$
ph = 0.4 # Probability of the coin coming up heads

# Initialization of auxiliary arrays and matrices
V = np.zeros((numS, 1)) # V(s) state-value function
V[goal] = 0 # ¿¿PONER A UNO SEGÚN ENUNCIADO DE p.89 ex. 4.9??
pi = np.zeros((numS, numA_max)) # pi(a|s);
p = np.zeros((numS, numA_max, numS)) # p(s'|s,a);
p_mat = np.zeros((numS, numA_max, numS*numA_max))
r = np.zeros((numS, numA_max, numS)) # r(s,a,s');
r_mat = np.zeros((numS, 1, numS*numA_max))

for s in range(numS):#for s in states:
    for a in range(np.minimum(s,goal-s)+1):
        # Con el += conseguimos que si una misma acción produce una transición al mismo estado
        # tanto si ganas como si pierdes, se acumulen las probabilidades de transición (caso a=0$).
        p[s,a,s+a] = ph
        p[s,a,s-a] += 1-ph

for s in states:
    for a in range(np.minimum(s,goal-s)+1):
        if s+a == goal:
            r[s,a,s+a] = 1

for s in range(numS):
    for a in range(numA_max):
        p_mat[s,a,a*numS:(a+1)*numS] = p[s,a,:]
        r_mat[s,0,a*numS:(a+1)*numS] = r[s,a,:]

g_mat = np.empty((0,numS))
for a in range(numA_max):
    g_mat = np.concatenate((g_mat,np.eye(numS,numS)))

# "In place" implementation of iterative policy evaluation (policy = choose that action with which we maximize V)
theta = 1e-4 # tantos ceros después de la coma como exponente-1
v = 0
gamma = 1
while True:
    inc=0
    for s in range(numS):
        v = V[s,0] # slicing necesario (en lugar de poner V[s] para no copiar la referencia y que al cambiar V[s] no se cambie también v
        V[s] = np.max(np.dot(p_mat[s,:,:],r_mat[s,:,:].transpose()) + np.dot(np.dot(p_mat[s,:,:],g_mat),gamma*V))
        inc = np.maximum(inc,np.abs(v-V[s]))
    if inc < theta:
        break

# Output a deterministic policy
order = 1 # In tie case, choose the "order" option.
policy = np.zeros((numS,1))
for s in range(numS):
    # In case of multiple occurrences of the maximum values, the index corresponding to the number "order" occurrence are returned.
    V_final = np.dot(p_mat[s,:,:],r_mat[s,:,:].transpose()) + np.dot(np.dot(p_mat[s,:,:],g_mat),gamma*V)
    m = max(V_final)
    actmax = [i for i, j in enumerate(V_final) if j == m] # vector con las acciones que empatan
    if m != 0: # si hay empate de V_final con valor distinto de cero...
        policy[s] = actmax[order % np.size(actmax,0)]
    else: # si hay empate de V_final, pero con valor 0, es porque estamos en los estados terminales...
        policy[s] = actmax[0 % np.size(actmax,0)] # ...y la única acción que podemos tomar es apostar 0$

plt.figure(1)
plt.stem(np.arange(numS),policy)
plt.xlabel("Capital"), plt.ylabel("Final policy (stake)"), plt.xticks(np.linspace(0,goal,5))

plt.figure(2)
plt.plot(np.arange(numS),V)
plt.xlabel("Capital"), plt.ylabel("Value estimates"), plt.xticks(np.linspace(0,goal,5)), plt.grid()
plt.show()
