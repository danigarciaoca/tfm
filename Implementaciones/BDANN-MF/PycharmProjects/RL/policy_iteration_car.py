import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
poisson.pmf(x, mu)
# Initialization of constants
car_states = 20+1 # +1 due to 0 cars state
car_move = 5 + 1 # a maximum of five cars can be moved from one location to another; +1 due to 0 cars moved
lambda1req = 3 # expected number of rental requests at 1st location
lambda2req = 4 # expected number of rental requests at 2nd location
lambda1ret = 3 # expected number of return requests at 1st location
lambda2ret = 2 # expected number of return requests at 2nd location

# Initialization of auxiliary arrays and matrices
V = np.zeros((car_states, 1)) # V(s) state-value function
pi1 = np.zeros((car_states, car_move)) # pi(a|s);
pi1 = np.zeros((car_states, car_move)) # pi(a|s);
p = np.zeros((car_states, car_move, car_states)) # p(s'|s,a);
p_mat = np.zeros((car_states, car_move, car_states*car_move))
r = np.zeros((car_states, car_move, car_states)) # r(s,a,s');
r_mat = np.zeros((car_states, 1, car_states*car_move))

for s in states:
    pi[s,:np.minimum(s,goal-s)+1] = 1/(np.minimum(s,goal-s)+1)

for s in range(car_states):#for s in states:
    for a in range(np.minimum(s,goal-s)+1):
        # Con el += conseguimos que si una misma acción produce una transición al mismo estado
        # tanto si ganas como si pierdes, se acumulen las probabilidades de transición (caso a=0$).
        p[s,a,s+a] += ph
        p[s,a,s-a] += 1-ph

for s in states:
    for a in range(np.minimum(s,goal-s)+1):
        # Con el += conseguimos que si una misma acción produce una transición al mismo estado
        # tanto si ganas como si pierdes, se acumulen las probabilidades de transición (caso a=0$).
        if s+a == goal:
            r[s,a,s+a] = 1

for s in range(car_states):
    for a in range(car_move):
        p_mat[s,a,a*car_states:(a+1)*car_states] = p[s,a,:]
        r_mat[s,0,a*car_states:(a+1)*car_states] = r[s,a,:]

g_mat = np.empty((0,car_states))
for a in range(car_move):
    g_mat = np.concatenate((g_mat,np.eye(car_states,car_states)))

# "In place" implementation of iterative policy evaluation (policy = choose that action with which we maximize V)
theta = 1e-4 # tantos ceros después de la coma como exponente-1
v = 0
gamma = 0.9
while True:
    inc=0
    for s in range(car_states):
        v = V[s,0] # slicing necesario (en lugar de poner V[s] para no copiar la referencia y que al cambiar V[s] no se cambie también v
        V[s] = np.max(np.dot(p_mat[s,:,:],r_mat[s,:,:].transpose()) + np.dot(np.dot(p_mat[s,:,:],g_mat),gamma*V))
        inc = np.maximum(inc,np.abs(v-V[s]))
    if inc < theta:
        break

# Output a deterministic policy
order = 0 # In tie case, choose the "order" option.
policy = np.zeros((car_states,1))
for s in range(car_states):
    # In case of multiple occurrences of the maximum values, the index corresponding to the number "order" occurrence are returned.
    V_final = np.dot(p_mat[s,:,:],r_mat[s,:,:].transpose()) + np.dot(np.dot(p_mat[s,:,:],g_mat),gamma*V)
    m = max(V_final)
    actmax = [i for i, j in enumerate(V_final) if j == m] # vector con las acciones que empatan
    if m != 0: # si hay empate de V_final con valor distinto de cero...
        policy[s] = actmax[order % np.size(actmax,0)]
    else: # si hay empate de V_final, pero con valor 0, es porque estamos en los estados terminales...
        policy[s] = actmax[0 % np.size(actmax,0)] # ...y la única acción que podemos tomar es apostar 0$

plt.figure(1)
plt.stem(np.arange(car_states),policy)
plt.xlabel("Capital"), plt.ylabel("Final policy (stake)")

plt.figure(2)
plt.stem(np.arange(car_states),V)
plt.xlabel("Capital"), plt.ylabel("Value estimates")
plt.show()
