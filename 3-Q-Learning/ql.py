import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

#lista invertida para usar con Greedy y EGreedy
actions_list_inv = {0: "UP",
                    1: "RIGHT",
                    2: "DOWN",
                    3: "LEFT"
                }
# Discount factor
discount = 0.8

movEpisod = 0
recompEpisod = 0

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension

def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

def getBestActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1 and np.amax(Q[state]) == Q[state][1]:
        actions.append("RIGHT")
    if x > 0 and np.amax(Q[state]) == Q[state][3]:
        actions.append("LEFT")
    if y < height - 1 and np.amax(Q[state]) == Q[state][2]:
        actions.append("DOWN")
    if y > 0 and np.amax(Q[state]) == Q[state][0]:
        actions.append("UP")
    return actions


def getRndAction(state):
    return random.choice(getActions(state))

# Greedy cuando haya otro valor de recompensa contiguo mayor que 0.Devolviendo la mejor recompensa, si no, aleatorio
def greedy(state):
    if(np.amax(Q[state]) > 0 ): 
        return actions_list_inv[np.argmax(Q[state])]
    else:
        return getRndAction(state)

# EGreedy cuando el random generado mayor que el ratio, en tal caso explora aleatoriamente, si no sigue en explotacion
def eGreedy(state, rate):
    if(random.random() > rate):
        return getRndAction(state)
    else:
        return greedy(state)

def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))

def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return


# Episodes
for i in xrange(100):
    state = getRndState()

    while state != final_state:
        rate = 0.8  #ratio para continuar explotacion
        #-------------Apartado 1 Random--------------
        action = getRndAction(state)
        #-------------Apartado 2 Greedy--------------
        #action = greedy(state)
        #-------------Apartado 2 EGreedy-------------
        action = eGreedy(state, rate)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]

        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state

        movEpisod = movEpisod + 1
        recompEpisod = recompEpisod + Rewards[new_state]

print Q
# Q matrix plot


print "Recompensa promedia por accion: ", recompEpisod/movEpisod
print "Acciones promedio por episodio: ", movEpisod/100


s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()
