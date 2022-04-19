from cmath import exp


import numpy as np
import math
import random


Temperture = 10000
dT = 0.98
eps = 1e-6
y = 66


def fitness(x):
    return math.log(x)/x+y


def simulated_annealing():
    global Temperture
    x = random.uniform(0, 100)
    n = fitness(x)
    answer = n
    while(Temperture > eps):
        new_x = x+Temperture*random.uniform(-100, 100)
        if new_x >= 0 and new_x <= 100:
            new_y = fitness(new_x)
            answer = max(n, new_y)
            if(new_y-n > eps):
                n = new_y
                x = new_x
            elif(random.random() < math.exp((new_y-n)/Temperture)):
                n = new_y
                x = new_x
        Temperture *= dT
    return answer


print(simulated_annealing())
